"""Siemens AOI 가성 데이터 기준 모델과 Pfab·Rothering(2025) 평가 절차의 재현 (2026-11-23).

Pfab 절차: 시간순으로 앞 절반 = 모델링(그 안에서 층화 무작위 80:20), 뒤 절반 = 시간순 다섯 구간.
문턱값은 모델링 80%의 5겹 교차검증 예측(OOF)으로 정한다(놓침 1%).
변형
  base: 측정값과 범주 열만(Pfab과 같은 입력)
  ctx : base + 같은 시각(같은 보드 검사로 보이는) 묶음의 맥락(묶음 크기, 검사 유형별 개수,
        측정값 몇 개의 묶음 평균과의 차이). 다른 행의 라벨은 쓰지 않는다.
모델: LightGBM, 씨앗 5개.

사용: python -I 02_baseline_siemens.py <데이터 폴더> <결과 폴더>
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import evaluate, threshold_for_f1, threshold_for_slip  # noqa: E402

DATA = Path(sys.argv[1]) / "siemens"
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
SLIP = 0.01
SEEDS = [0, 1, 2, 3, 4]
SPLIT = "grouped"  # 같은 시각 묶음(한 보드로 보이는 것)을 훈련과 시험에 나눠 넣지 않는다
CAT = ["inspection_type", "meta_feat1", "meta_feat2", "meta_feat3", "meta_feat4"]


def log(*a):
    print(*a, flush=True)


df = pd.read_csv(DATA / "dataset.csv", low_memory=False)
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
y_all = df["class"].values.astype(int)
groups_all = df["timestamp"].factorize()[0]
insp = [c for c in df.columns if c.startswith("inspection_feat")]

# 맥락 특징
g = df.groupby("timestamp")
ctx = pd.DataFrame(index=df.index)
ctx["grp_size"] = g["class"].transform("size")
for t in sorted(df["inspection_type"].unique()):
    ctx[f"grp_type{t}"] = g["inspection_type"].transform(lambda v, t=t: (v == t).sum())
for c in insp[:10]:
    ctx[f"grp_dev_{c}"] = df[c] - g[c].transform("mean")

X_base = df[CAT + insp].copy()
for c in CAT:
    X_base[c] = X_base[c].astype("category")
X_ctx = pd.concat([X_base, ctx], axis=1)

n = len(df)
half = n // 2
model_idx = np.arange(half)
eval_slices = np.array_split(np.arange(half, n), 5)


def fit_predict(Xtr, ytr, Xte, seed):
    pos = max(ytr.sum(), 1)
    m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=63, min_child_samples=40,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           scale_pos_weight=(len(ytr) - pos) / pos, random_state=seed, verbose=-1, n_jobs=16,
                           deterministic=True, force_row_wise=True)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


out = {}
for name, X in [("base", X_base), ("ctx", X_ctx)]:
    res = []
    for seed in SEEDS:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        a, b = next(gss.split(model_idx, y_all[model_idx], groups_all[model_idx]))
        hp_idx, te_idx = model_idx[a], model_idx[b]
        Xh, yh = X.iloc[hp_idx], y_all[hp_idx]
        oof = np.zeros(len(hp_idx))
        for f_tr, f_va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(Xh, yh, groups_all[hp_idx]):
            oof[f_va] = fit_predict(Xh.iloc[f_tr], yh[f_tr], Xh.iloc[f_va], seed)
        t_f1 = threshold_for_f1(yh, oof)
        t_slip = threshold_for_slip(yh, oof, SLIP)
        # 모델링 데이터 전체(80%)로 다시 학습
        scores_te = fit_predict(Xh, yh, X.iloc[te_idx], seed)
        r = {"seed": seed, "t_slip": t_slip, "cv": evaluate(yh, oof, t_f1, t_slip),
             "random_test": evaluate(y_all[te_idx], scores_te, t_f1, t_slip), "slices": []}
        scores_ev = fit_predict(Xh, yh, X.iloc[np.arange(half, n)], seed)
        off = 0
        for i, sl in enumerate(eval_slices):
            sc = scores_ev[off: off + len(sl)]
            off += len(sl)
            e = evaluate(y_all[sl], sc, t_f1, t_slip)
            e["start"] = str(df["timestamp"].iloc[sl[0]])
            r["slices"].append(e)
        res.append(r)
        log(name, seed, "random_test", json.dumps(r["random_test"]),
            "slices slip", [round(s["slip"], 3) for s in r["slices"]],
            "vol", [round(s["volume_reduction"], 3) for s in r["slices"]])
    out[name] = res


def mean_sd(vals):
    return [round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)]


summ = {}
for name, res in out.items():
    s = {"random_test": {k: mean_sd([r["random_test"][k] for r in res])
                         for k in ["pr_auc", "f1_bad", "slip", "volume_reduction"]}}
    s["slices"] = [{k: mean_sd([r["slices"][i][k] for r in res]) for k in ["pr_auc", "slip", "volume_reduction"]}
                   | {"n_bad": res[0]["slices"][i]["n_bad"], "n": res[0]["slices"][i]["n"],
                      "start": res[0]["slices"][i]["start"]}
                   for i in range(5)]
    summ[name] = s
out["summary"] = summ
(OUT / f"baseline_siemens_{SPLIT}.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
log(json.dumps(summ, indent=1))
