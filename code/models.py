"""모델 공통 코드: LightGBM, 2단계(보드 이웃의 1단계 점수를 특징으로), 묶음 나누기.

누출 방지 원칙
- 교차검증은 묶음(보드·패널·같은 시각) 단위로 나눈다.
- 2단계의 훈련 특징은 중첩 교차검증으로 만든다(바깥 검증 묶음의 라벨이 어떤 경로로도 들어가지 않게).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

LGB_PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=63, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
                  deterministic=True, force_row_wise=True, n_jobs=16)


def lgb_fit_predict(Xtr, ytr, Xte_list, seed, params=None):
    p = dict(LGB_PARAMS if params is None else params)
    pos = max(int(ytr.sum()), 1)
    w = (len(ytr) - pos) / pos
    if p.pop("spw", "full") == "sqrt":  # 29번 탐색용: 양성 가중을 비율의 제곱근으로
        w = float(np.sqrt(w))
    m = lgb.LGBMClassifier(scale_pos_weight=w, random_state=seed, **p)
    m.fit(Xtr, ytr)
    return [m.predict_proba(X)[:, 1] for X in Xte_list]


def folds(y, groups, seed, n=5, grouped=True):
    if grouped:
        return list(StratifiedGroupKFold(n, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))
    return list(StratifiedKFold(n, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def oof_scores(X, y, groups, seed, grouped=True, params=None):
    oof = np.zeros(len(y))
    for tr, va in folds(y, groups, seed, grouped=grouped):
        oof[va] = lgb_fit_predict(X.iloc[tr], y[tr], [X.iloc[va]], seed, params)[0]
    return oof


def neighbour_features(score, group_levels: dict):
    """같은 묶음의 '다른' 항목 점수 요약. group_levels = {이름: 묶음 번호 배열}.

    점수는 1단계 모델의 예측(라벨 아님). 자기 자신은 뺀다.
    """
    s = pd.Series(score)
    out = {"s1_self": score}
    for name, g in group_levels.items():
        g = pd.Series(g)
        grp = s.groupby(g)
        n = grp.transform("size").values
        tot = grp.transform("sum").values
        top1 = grp.transform("max").values
        # 두 번째로 큰 값: 자기 자신이 최대일 때 쓰기 위해
        rank_desc = grp.rank(ascending=False, method="first").values
        second = s.where(rank_desc == 2).groupby(g).transform("max").values
        mx_other = np.where(score >= top1, second, top1)
        mean_other = np.where(n > 1, (tot - score) / np.maximum(n - 1, 1), np.nan)
        cnt_hi = (s > 0.5).groupby(g).transform("sum").values - (score > 0.5)
        out[f"nb_{name}_n"] = n
        out[f"nb_{name}_max_other"] = np.where(n > 1, mx_other, np.nan)
        out[f"nb_{name}_mean_other"] = mean_other
        out[f"nb_{name}_cnt_hi_other"] = cnt_hi
        out[f"nb_{name}_rank"] = rank_desc / n
    return pd.DataFrame(out)


def crossfit(X, y, groups, seed, X_eval, grouped=True, params=None):
    """교차적합: 5겹 각 모델로 (자기 검증 겹의 OOF 점수, 평가 기간 점수)를 낸다.

    운영 모델 = 이 다섯 모델. 문턱값 보정도 같은 다섯 모델의 OOF로 하므로 점수 눈금이 일치한다.
    반환: oof(len y), fold_id(len y), eval_k(5 x len X_eval)
    """
    fl = folds(y, groups, seed, grouped=grouped)
    oof = np.zeros(len(y))
    fold_id = np.full(len(y), -1)
    ev = np.zeros((len(fl), len(X_eval)))
    for k, (tr, va) in enumerate(fl):
        p_va, p_ev = lgb_fit_predict(X.iloc[tr], y[tr], [X.iloc[va], X_eval], seed, params)
        oof[va], fold_id[va], ev[k] = p_va, k, p_ev
    return oof, fold_id, ev


def two_stage_crossfit(X_dev, y_dev, groups_dev, levels_dev, X_eval, levels_eval, seed, params=None):
    """2단계 모델의 교차적합. 바깥 겹마다 안쪽 5겹으로 1단계 OOF를 만들어 2단계 훈련 특징으로 쓴다."""
    def stack(X, s1, levels):
        nb = neighbour_features(s1, levels)
        nb.index = X.index
        return pd.concat([X, nb], axis=1)

    fl = folds(y_dev, groups_dev, seed)
    oof2 = np.zeros(len(y_dev))
    fold_id = np.full(len(y_dev), -1)
    ev2 = np.zeros((len(fl), len(X_eval)))
    for k, (tr, va) in enumerate(fl):
        Xtr, ytr = X_dev.iloc[tr], y_dev[tr]
        lv_tr = {n: v[tr] for n, v in levels_dev.items()}
        lv_va = {n: v[va] for n, v in levels_dev.items()}
        s1_tr = oof_scores(Xtr, ytr, groups_dev[tr], seed, grouped=True, params=params)
        s1_va, s1_ev = lgb_fit_predict(Xtr, ytr, [X_dev.iloc[va], X_eval], seed, params)
        p_va, p_ev = lgb_fit_predict(stack(Xtr, s1_tr, lv_tr), ytr,
                                     [stack(X_dev.iloc[va], s1_va, lv_va), stack(X_eval, s1_ev, levels_eval)],
                                     seed, params)
        oof2[va], fold_id[va], ev2[k] = p_va, k, p_ev
    return oof2, fold_id, ev2


def two_stage(X_dev, y_dev, groups_dev, levels_dev, X_eval, levels_eval, seed, params=None):
    """2단계 모델. 반환: (개발 기간 OOF 점수, 평가 기간 점수).

    levels_* : neighbour_features에 줄 묶음 번호들(dict). groups_dev: 교차검증 묶음.
    바깥 5겹마다 안쪽 5겹으로 1단계 OOF를 만들어 2단계 훈련 특징으로 쓴다.
    """
    def stage1_oof(X, y, g, s):
        return oof_scores(X, y, g, s, grouped=True, params=params)

    def stack(X, s1, levels):
        nb = neighbour_features(s1, levels)
        nb.index = X.index
        return pd.concat([X, nb], axis=1)

    oof2 = np.zeros(len(y_dev))
    for tr, va in folds(y_dev, groups_dev, seed):
        Xtr, ytr = X_dev.iloc[tr], y_dev[tr]
        lv_tr = {k: v[tr] for k, v in levels_dev.items()}
        lv_va = {k: v[va] for k, v in levels_dev.items()}
        s1_tr = stage1_oof(Xtr, ytr, groups_dev[tr], seed)
        s1_va = lgb_fit_predict(Xtr, ytr, [X_dev.iloc[va]], seed, params)[0]
        oof2[va] = lgb_fit_predict(stack(Xtr, s1_tr, lv_tr), ytr,
                                   [stack(X_dev.iloc[va], s1_va, lv_va)], seed, params)[0]
    s1_dev = stage1_oof(X_dev, y_dev, groups_dev, seed)
    s1_eval = lgb_fit_predict(X_dev, y_dev, [X_eval], seed, params)[0]
    s2_eval = lgb_fit_predict(stack(X_dev, s1_dev, levels_dev), y_dev,
                              [stack(X_eval, s1_eval, levels_eval)], seed, params)[0]
    return oof2, s2_eval
