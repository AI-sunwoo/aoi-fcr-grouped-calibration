"""PHME 시간순 분할의 설정 탐색을 7월 패널만으로 다시 한다 (2026-11-23, 3차 검토 반영).

23번(BoardSet)과 29번(GBDT)은 대회 훈련 자료(official dev)로 설정을 골랐는데, 그 자료에는 9월 패널(시간순
분할의 평가 기간)이 들어 있다. 그래서 시간순 분할에 쓰는 설정을 7월 패널(시간순 개발 기간)의 묶음 단위
5겹 OOF AP로만 다시 고르고, 같은 교차적합을 되풀이한다. 탐색 범위는 23·29번과 같다.

저장(results 폴더): gbdt_tuned_phme_july.json, boardset_cfg_phme_july.json,
  phme_temporal_{lgbj_noctx|lgbj_ctx|two_stage_j}_seed<s>.npz (씨앗 3),
  phme_temporal_{boardsetj_ctx|boardsetj_noctx}_seed<s>.npz (씨앗 5)
사용: python -I 35_july_tuning.py <tables 폴더> <결과 폴더> <스레드 수>
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
from boardset import fit_predict, prep_numeric  # noqa: E402
from models import crossfit, folds, oof_scores, two_stage_crossfit  # noqa: E402
from phme_data import feature_columns  # noqa: E402

TAB, OUT, NJ = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save(name, g):
    np.savez_compressed(OUT / f"{name}.npz", oof=g[0], fold=g[1], ev=g[2])


def cat_codes(df_cat, fit_rows, rows_list):
    cards, outs = [], [np.zeros((len(r), df_cat.shape[1]), np.int64) for r in rows_list]
    for j, c in enumerate(df_cat.columns):
        vals = pd.Index(pd.unique(df_cat[c].iloc[fit_rows].astype(str)))
        cards.append(len(vals))
        for o, r in zip(outs, rows_list):
            o[:, j] = vals.get_indexer(df_cat[c].iloc[r].astype(str)) + 1
    return cards, outs


tab = pd.read_parquet(TAB / "phme_components.parquet")
tab = tab[tab["date"].notna()].reset_index(drop=True)
y = tab["y"].values.astype(int)
panel = tab["PanelID"].factorize()[0]
board = (tab["PanelID"] + "_" + tab["FigureID"]).factorize()[0]
pos = (tab["PanelID"] + "_" + tab["ComponentID"].astype(str)).factorize()[0]
dev, ev = np.where(tab["date"] < "2019-08-01")[0], np.where(tab["date"] >= "2019-08-01")[0]
tabc = tab.copy()
for c in ["ComponentID", "comp_type"]:
    tabc[c] = tabc[c].astype("category")
Xn, Xc = tabc[feature_columns(tabc, use_ctx=False)], tabc[feature_columns(tabc, use_ctx=True)]
lv = {"board": board, "panel": panel, "pos": pos}

# ---------- A. GBDT: 29번과 같은 8가지
BASE = dict(n_estimators=500, learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            verbose=-1, deterministic=True, force_row_wise=True, n_jobs=NJ)
GRID = [dict(BASE, num_leaves=nl, min_child_samples=mc, spw=w)
        for nl, mc, w in itertools.product([31, 127], [20, 100], ["full", "sqrt"])]
aps = []
for i, prm in enumerate(GRID):
    oof = oof_scores(Xc.iloc[dev], y[dev], panel[dev], 0, grouped=True, params=prm)
    aps.append(float(average_precision_score(y[dev], oof)))
    log("gbdt config", i, {k: prm[k] for k in ["num_leaves", "min_child_samples", "spw"]}, "OOF AP %.4f" % aps[-1])
best = GRID[int(np.argmax(aps))]
(OUT / "gbdt_tuned_phme_july.json").write_text(json.dumps({"grid_ap": aps, "best": {
    k: best[k] for k in ["num_leaves", "min_child_samples", "spw"]}}, indent=1))
for seed in range(3):
    for name, X in [("lgbj_noctx", Xn), ("lgbj_ctx", Xc)]:
        save(f"phme_temporal_{name}_seed{seed}", crossfit(X.iloc[dev], y[dev], panel[dev], seed, X.iloc[ev],
                                                          grouped=True, params=best))
    save(f"phme_temporal_two_stage_j_seed{seed}",
         two_stage_crossfit(Xc.iloc[dev], y[dev], panel[dev], {k: v[dev] for k, v in lv.items()}, Xc.iloc[ev],
                            {k: v[ev] for k, v in lv.items()}, seed, best))
    log("gbdt temporal seed", seed)

# ---------- B. BoardSet: 23번과 같은 8가지, 7월 패널만
catdf = tab[["ComponentID", "comp_type", "FigureID"]]
num_c = tab[[c for c in feature_columns(tab, use_ctx=True) if c not in ("ComponentID", "comp_type")]].astype(float).values
num_n = tab[[c for c in feature_columns(tab, use_ctx=False) if c not in ("ComponentID", "comp_type")]].astype(float).values
base = dict(max_epochs=80, patience=12, val_frac=0.15, lr=1e-3, pw="sqrt")
grid = [dict(base, d=d, layers=L, dropout=p) for d, L, p in itertools.product([64, 128], [1, 2], [0.1, 0.3])]
fl = folds(y[dev], panel[dev], 0)
prepped = []
for tr, va in fl:
    rtr, rva = dev[tr], dev[va]
    XN_tr, XN_va = prep_numeric(num_c[rtr], [num_c[rva]])
    cards, (XC_tr, XC_va) = cat_codes(catdf, rtr, [rtr, rva])
    prepped.append((tr, va, rtr, rva, XN_tr, XN_va, cards, XC_tr, XC_va))
results = []
for cfg in grid:
    oof = np.zeros(len(dev))
    for tr, va, rtr, rva, XN_tr, XN_va, cards, XC_tr, XC_va in prepped:
        oof[va] = fit_predict(XN_tr, XC_tr, y[rtr], panel[rtr], [XN_va], [XC_va], [panel[rva]], cards, 0, **cfg)[0]
    results.append({"cfg": cfg, "oof_ap": float(average_precision_score(y[dev], oof))})
    log("boardset", json.dumps(results[-1]))
cfg = dict(max(results, key=lambda r: r["oof_ap"])["cfg"], n_init=3)
(OUT / "boardset_cfg_phme_july.json").write_text(json.dumps({"best": cfg, "grid": results}, indent=1))


def run(name, num, seed):
    fl = folds(y[dev], panel[dev], seed)
    oof, fold_id = np.zeros(len(dev), np.float32), np.full(len(dev), -1)
    evk = np.zeros((len(fl), len(ev)), np.float32)
    for k, (tr, va) in enumerate(fl):
        rtr, rva = dev[tr], dev[va]
        XN_tr, XN_va, XN_ev = prep_numeric(num[rtr], [num[rva], num[ev]])
        cards, (XC_tr, XC_va, XC_ev) = cat_codes(catdf, rtr, [rtr, rva, ev])
        p_va, p_ev = fit_predict(XN_tr, XC_tr, y[rtr], panel[rtr], [XN_va, XN_ev], [XC_va, XC_ev],
                                 [panel[rva], panel[ev]], cards, seed, **cfg)
        oof[va], fold_id[va], evk[k] = p_va, k, p_ev
    save(f"phme_temporal_{name}_seed{seed}", (oof, fold_id, evk))


for seed in range(5):
    run("boardsetj_ctx", num_c, seed)
    run("boardsetj_noctx", num_n, seed)
    log("boardset temporal seed", seed)
(OUT / "phme_july_done").write_text("done")
