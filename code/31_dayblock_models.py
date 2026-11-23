"""날짜 단위 교차적합을 2단계 모델·BoardSet에도 (2026-11-23, 2판 검토 반영).

27번은 GBDT만 날짜 단위로 돌렸다. 본문이 "보드·날짜 단위로 보정하라"고 권하므로 주 모델 둘도 같은 묶음으로
보정했을 때의 실제 놓침·물량을 잰다. 모델·특징·설정은 26번(2단계)·22번(BoardSet)과 같고 교차검증 묶음만 날짜다.
BoardSet의 집합(주의가 보는 범위)은 그대로 패널(PHME)·타임스탬프 묶음(Siemens)이고, 조기 종료용 검증 집합도
날짜 단위로 뗀다.

저장: <ds>_<split>_{two_stage|boardset_ctx}_day_seed<s>.npz (oof, fold, ev) — 27번과 같은 형식
사용: python -I 31_dayblock_models.py <phme|siemens> <tables 폴더> <결과 폴더> <씨앗 개수> <boardset 설정 json> <모델들(쉼표)>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fcr_eval import siemens_half  # noqa: E402
from boardset import fit_predict, prep_numeric  # noqa: E402
from models import folds, lgb_fit_predict, neighbour_features, oof_scores  # noqa: E402
from phme_data import feature_columns  # noqa: E402


def two_stage_crossfit(X_dev, y_dev, outer_groups, inner_groups, levels_dev, X_eval, levels_eval, seed, params=None):
    """models.two_stage_crossfit과 같되 바깥 겹(보정에 쓰는 보류 불량)은 날짜, 안쪽 겹(2단계 훈련 특징을
    만드는 1단계 OOF)은 원래대로 패널·타임스탬프 묶음. PHME 7월은 날짜가 5일뿐이라 안쪽까지 날짜로 나눌 수 없다."""
    def stack(X, s1, levels):
        nb = neighbour_features(s1, levels)
        nb.index = X.index
        return pd.concat([X, nb], axis=1)

    fl = folds(y_dev, outer_groups, seed)
    oof2 = np.zeros(len(y_dev))
    fold_id = np.full(len(y_dev), -1)
    ev2 = np.zeros((len(fl), len(X_eval)))
    for k, (tr, va) in enumerate(fl):
        Xtr, ytr = X_dev.iloc[tr], y_dev[tr]
        lv_tr = {n: v[tr] for n, v in levels_dev.items()}
        lv_va = {n: v[va] for n, v in levels_dev.items()}
        s1_tr = oof_scores(Xtr, ytr, inner_groups[tr], seed, grouped=True, params=params)
        s1_va, s1_ev = lgb_fit_predict(Xtr, ytr, [X_dev.iloc[va], X_eval], seed, params)
        p_va, p_ev = lgb_fit_predict(stack(Xtr, s1_tr, lv_tr), ytr,
                                     [stack(X_dev.iloc[va], s1_va, lv_va), stack(X_eval, s1_ev, levels_eval)],
                                     seed, params)
        oof2[va], fold_id[va], ev2[k] = p_va, k, p_ev
    return oof2, fold_id, ev2

DS, TAB, OUT, N_SEEDS = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
CFG = json.loads(Path(sys.argv[5]).read_text())
MODELS = sys.argv[6].split(",")
OUT.mkdir(parents=True, exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save(name, g):
    np.savez_compressed(OUT / f"{name}.npz", oof=g[0], fold=g[1], ev=g[2])


def cat_codes(df_cat, fit_rows, rows_list):
    """22번과 같다: 훈련 행에서 본 값에 1..K, 처음 보는 값은 0."""
    cards, outs = [], [np.zeros((len(r), df_cat.shape[1]), np.int64) for r in rows_list]
    for j, c in enumerate(df_cat.columns):
        vals = pd.Index(pd.unique(df_cat[c].iloc[fit_rows].astype(str)))
        cards.append(len(vals))
        for o, r in zip(outs, rows_list):
            o[:, j] = vals.get_indexer(df_cat[c].iloc[r].astype(str)) + 1
    return cards, outs


def boardset_day(num, catdf, y, sets, day, dev, ev, seed):
    fl = folds(y[dev], day[dev], seed)
    oof = np.zeros(len(dev), np.float32)
    fold_id = np.full(len(dev), -1)
    evk = np.zeros((len(fl), len(ev)), np.float32)
    for k, (tr, va) in enumerate(fl):
        rtr, rva = dev[tr], dev[va]
        XN_tr, XN_va, XN_ev = prep_numeric(num[rtr], [num[rva], num[ev]])
        cards, (XC_tr, XC_va, XC_ev) = cat_codes(catdf, rtr, [rtr, rva, ev])
        p_va, p_ev = fit_predict(XN_tr, XC_tr, y[rtr], sets[rtr], [XN_va, XN_ev], [XC_va, XC_ev],
                                 [sets[rva], sets[ev]], cards, seed, val_groups=day[rtr], **CFG)
        oof[va], fold_id[va], evk[k] = p_va, k, p_ev
    return oof, fold_id, evk


if DS == "phme":
    PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, verbose=-1, deterministic=True, force_row_wise=True,
                  n_jobs=16)
    tab = pd.read_parquet(TAB / "phme_components.parquet")
    tab = tab[tab["date"].notna()].reset_index(drop=True)
    day = pd.to_datetime(tab["date"]).dt.strftime("%Y-%m-%d").factorize()[0]
    panel = tab["PanelID"].factorize()[0]
    board = (tab["PanelID"] + "_" + tab["FigureID"]).factorize()[0]
    pos = (tab["PanelID"] + "_" + tab["ComponentID"].astype(str)).factorize()[0]
    y = tab["y"].values.astype(int)
    splits = {"official": (np.where(tab["split"] == "train")[0], np.where(tab["split"] == "test")[0]),
              "temporal": (np.where(tab["date"] < "2019-08-01")[0], np.where(tab["date"] >= "2019-08-01")[0])}
    catdf = tab[["ComponentID", "comp_type", "FigureID"]]
    num = tab[[c for c in feature_columns(tab, use_ctx=True) if c not in ("ComponentID", "comp_type")]] \
        .astype(float).values
    tabc = tab.copy()
    for c in ["ComponentID", "comp_type"]:
        tabc[c] = tabc[c].astype("category")
    Xc = tabc[feature_columns(tabc, use_ctx=True)]
    lv = {"board": board, "panel": panel, "pos": pos}
    for split, (dev, ev) in splits.items():
        for seed in range(N_SEEDS):
            if "two_stage" in MODELS:
                g = two_stage_crossfit(Xc.iloc[dev], y[dev], day[dev], panel[dev], {k: v[dev] for k, v in lv.items()},
                                       Xc.iloc[ev], {k: v[ev] for k, v in lv.items()}, seed, PARAMS)
                save(f"phme_{split}_two_stage_day_seed{seed}", g)
                log(split, "two_stage day", seed)
            if "boardset_ctx" in MODELS:
                save(f"phme_{split}_boardset_ctx_day_seed{seed}", boardset_day(num, catdf, y, panel, day, dev, ev, seed))
                log(split, "boardset_ctx day", seed)
else:
    df = pd.read_parquet(TAB / "siemens_rows.parquet")
    y = df["class"].values.astype(int)
    grp = df["group"].values
    day = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d").factorize()[0]
    CAT = ["inspection_type", "meta_feat1", "meta_feat2", "meta_feat3", "meta_feat4"]
    insp = [c for c in df.columns if c.startswith("inspection_feat")]
    ctx = [c for c in df.columns if c.startswith("grp_")]
    Xc = df[CAT + insp].copy()
    for c in CAT:
        Xc[c] = Xc[c].astype("category")
    Xc = pd.concat([Xc, df[ctx]], axis=1)
    half = siemens_half(df)
    dev, ev = np.arange(half), np.arange(half, len(df))
    num = df[insp + ctx].astype(float).values
    num_noctx = df[insp].astype(float).values
    catdf = df[CAT]
    for seed in range(N_SEEDS):
        if "two_stage" in MODELS:
            g = two_stage_crossfit(Xc.iloc[dev], y[dev], day[dev], grp[dev], {"grp": grp[dev]}, Xc.iloc[ev],
                                   {"grp": grp[ev]}, seed)
            save(f"siemens_eval_two_stage_day_seed{seed}", g)
            log("two_stage day", seed)
        if "two_stage_dd" in MODELS:  # 심사 의견(3판): 안쪽 겹도 날짜 단위(Siemens는 날짜가 64일이라 가능)
            g = two_stage_crossfit(Xc.iloc[dev], y[dev], day[dev], day[dev], {"grp": grp[dev]}, Xc.iloc[ev],
                                   {"grp": grp[ev]}, seed)
            save(f"siemens_eval_two_stage_dd_day_seed{seed}", g)
            log("two_stage_dd day", seed)
        if "boardset_ctx" in MODELS:
            save(f"siemens_eval_boardset_ctx_day_seed{seed}", boardset_day(num, catdf, y, grp, day, dev, ev, seed))
            log("boardset_ctx day", seed)
        if "boardset_noctx" in MODELS:
            save(f"siemens_eval_boardset_noctx_day_seed{seed}",
                 boardset_day(num_noctx, catdf, y, grp, day, dev, ev, seed))
            log("boardset_noctx day", seed)
(OUT / f"{DS}_daymodels_{'_'.join(MODELS)}_done").write_text("done")
