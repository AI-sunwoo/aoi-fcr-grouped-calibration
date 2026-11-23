"""GBDT도 BoardSet과 같은 탐색 예산(8가지)을 준다 (2026-11-23, 심사 의견 반영).

1) 탐색: 개발 기간(PHME는 official 훈련 공개분, Siemens는 앞 절반) 묶음 단위 5겹 OOF AP로
   num_leaves {31,127} x min_child_samples {20,100} x 양성 가중 {비율, 비율의 제곱근}, 맥락 특징 포함, 씨앗 0.
2) 고른 설정으로 26번과 같은 교차적합을 다시 한다(묶음 단위만): 맥락 없음, 맥락 포함, 2단계.
   저장 이름: <ds>_<split>_{lgbt_noctx|lgbt_base|lgbt_ctx|two_stage_t}_seed<s>.npz

사용: python -I 29_gbdt_tuned.py <phme|siemens> <tables 폴더> <결과 폴더> <씨앗 개수>
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fcr_eval import siemens_half  # noqa: E402
from models import crossfit, oof_scores, two_stage_crossfit  # noqa: E402
from phme_data import feature_columns  # noqa: E402

DS, TAB, OUT, N_SEEDS = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
OUT.mkdir(parents=True, exist_ok=True)
BASE = dict(n_estimators=500, learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            verbose=-1, deterministic=True, force_row_wise=True, n_jobs=16)
GRID = [dict(BASE, num_leaves=nl, min_child_samples=mc, spw=w)
        for nl, mc, w in itertools.product([31, 127], [20, 100], ["full", "sqrt"])]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save(name, g):
    np.savez_compressed(OUT / f"{name}.npz", oof=g[0], fold=g[1], ev=g[2])


if DS == "phme":
    tab = pd.read_parquet(TAB / "phme_components.parquet")
    tab = tab[tab["date"].notna()].reset_index(drop=True)
    for c in ["ComponentID", "comp_type"]:
        tab[c] = tab[c].astype("category")
    panel = tab["PanelID"].factorize()[0]
    board = (tab["PanelID"] + "_" + tab["FigureID"]).factorize()[0]
    pos = (tab["PanelID"] + "_" + tab["ComponentID"].astype(str)).factorize()[0]
    y = tab["y"].values.astype(int)
    splits = {"official": (np.where(tab["split"] == "train")[0], np.where(tab["split"] == "test")[0]),
              "temporal": (np.where(tab["date"] < "2019-08-01")[0], np.where(tab["date"] >= "2019-08-01")[0])}
    Xn, Xc = tab[feature_columns(tab, use_ctx=False)], tab[feature_columns(tab, use_ctx=True)]
    sel_dev, groups, levels = splits["official"][0], panel, {"board": board, "panel": panel, "pos": pos}
    names = {"noctx": "lgbt_noctx"}
else:
    df = pd.read_parquet(TAB / "siemens_rows.parquet")
    y = df["class"].values.astype(int)
    grp = df["group"].values
    CAT = ["inspection_type", "meta_feat1", "meta_feat2", "meta_feat3", "meta_feat4"]
    insp = [c for c in df.columns if c.startswith("inspection_feat")]
    ctx = [c for c in df.columns if c.startswith("grp_")]
    Xn = df[CAT + insp].copy()
    for c in CAT:
        Xn[c] = Xn[c].astype("category")
    Xc = pd.concat([Xn, df[ctx]], axis=1)
    half = siemens_half(df)
    splits = {"eval": (np.arange(half), np.arange(half, len(df)))}
    sel_dev, groups, levels = splits["eval"][0], grp, {"grp": grp}
    names = {"noctx": "lgbt_base"}

scores = []
for i, prm in enumerate(GRID):
    oof = oof_scores(Xc.iloc[sel_dev], y[sel_dev], groups[sel_dev], 0, grouped=True, params=prm)
    ap = average_precision_score(y[sel_dev], oof)
    scores.append(ap)
    log("config", i, {k: prm[k] for k in ["num_leaves", "min_child_samples", "spw"]}, "OOF AP %.4f" % ap)
best = GRID[int(np.argmax(scores))]
(OUT / f"gbdt_tuned_{DS}.json").write_text(json.dumps({"grid_ap": scores, "best": {
    k: best[k] for k in ["num_leaves", "min_child_samples", "spw"]}}, indent=1))
log("best", {k: best[k] for k in ["num_leaves", "min_child_samples", "spw"]})

for split, (dev, ev) in splits.items():
    for seed in range(N_SEEDS):
        for name, X in [(names["noctx"], Xn), ("lgbt_ctx", Xc)]:
            g = crossfit(X.iloc[dev], y[dev], groups[dev], seed, X.iloc[ev], grouped=True, params=best)
            save(f"{DS}_{split}_{name}_seed{seed}", g)
            log(split, name, seed)
        g = two_stage_crossfit(Xc.iloc[dev], y[dev], groups[dev], {k: v[dev] for k, v in levels.items()},
                               Xc.iloc[ev], {k: v[ev] for k, v in levels.items()}, seed, best)
        save(f"{DS}_{split}_two_stage_t_seed{seed}", g)
        log(split, "two_stage_t", seed)
(OUT / f"{DS}_tuned_done").write_text("done")
