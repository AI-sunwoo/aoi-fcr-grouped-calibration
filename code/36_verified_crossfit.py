"""수리 확인 라벨로 모든 맥락 모델을 교차적합한다 (2026-11-23, 3차 검토 반영).

24번은 GBDT 두 개를 전체 개발 자료로 다시 학습해 확인했다. 심사 의견에 따라 2단계 모델과 BoardSet까지
26·22번과 같은 교차적합(패널 단위 5겹)으로 돌려, 같은 분석(AP, 정답 기준 물량, 보정 놓침, 부트스트랩)에 넣는다.
라벨: 첫 작업자 Bad 가운데 수리 작업자가 NotPossibleToRepair로 본 것만 1, Good·FalseScrap은 0,
NotYetClassified만 있는 불량 부품은 뺀다(24번과 같다). 공식 분할만.

저장: phme_verified_meta.npz (y, dev, ev, panel, board), phme_verified_<model>_seed<s>.npz
사용: python -I 36_verified_crossfit.py <데이터 폴더> <tables 폴더> <결과 폴더> <boardset 설정 json>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from boardset import fit_predict, prep_numeric  # noqa: E402
from models import crossfit, folds, two_stage_crossfit  # noqa: E402
from phme_data import IDS, K, feature_columns  # noqa: E402

DATA, TAB, OUT = Path(sys.argv[1]) / "phme", Path(sys.argv[2]), Path(sys.argv[3])
CFG = json.loads(Path(sys.argv[4]).read_text())
OUT.mkdir(parents=True, exist_ok=True)
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, verbose=-1, deterministic=True, force_row_wise=True, n_jobs=4)


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


lab = []
for name in ["training", "testing"]:
    a = pd.read_csv(DATA / f"AOI_{name}.csv.zip", dtype=IDS)
    bad = a[a["OperatorLabel"] == "Bad"]
    g = bad.groupby(K)["RepairLabel"].agg(lambda v: "NPR" if (v == "NotPossibleToRepair").any()
                                         else ("FS" if (v == "FalseScrap").any() else "NYC")).rename("rep")
    lab.append(g.reset_index())
lab = pd.concat(lab, ignore_index=True)
tab = pd.read_parquet(TAB / "phme_components.parquet")
tab = tab[tab["date"].notna()].reset_index(drop=True)
tab = tab.merge(lab, on=K, how="left")
tab["y_ver"] = np.where(tab["rep"] == "NPR", 1, 0)
tab = tab[~((tab["y"] == 1) & (tab["rep"] == "NYC"))].reset_index(drop=True)
y = tab["y_ver"].values.astype(int)
panel = tab["PanelID"].factorize()[0]
board = (tab["PanelID"] + "_" + tab["FigureID"]).factorize()[0]
pos = (tab["PanelID"] + "_" + tab["ComponentID"].astype(str)).factorize()[0]
dev, ev = np.where(tab["split"] == "train")[0], np.where(tab["split"] == "test")[0]
np.savez_compressed(OUT / "phme_verified_meta.npz", y=y, dev=dev, ev=ev, panel=panel, board=board)
log("dev", len(dev), int(y[dev].sum()), "eval", len(ev), int(y[ev].sum()))
drop = ["rep", "y_ver"]
tabc = tab.copy()
for c in ["ComponentID", "comp_type"]:
    tabc[c] = tabc[c].astype("category")
Xn = tabc[feature_columns(tabc, use_ctx=False)].drop(columns=drop, errors="ignore")
Xc = tabc[feature_columns(tabc, use_ctx=True)].drop(columns=drop, errors="ignore")
lv = {"board": board, "panel": panel, "pos": pos}
for seed in range(5):
    save(f"phme_verified_lgb_noctx_seed{seed}", crossfit(Xn.iloc[dev], y[dev], panel[dev], seed, Xn.iloc[ev],
                                                         grouped=True, params=PARAMS))
    save(f"phme_verified_lgb_ctx_seed{seed}", crossfit(Xc.iloc[dev], y[dev], panel[dev], seed, Xc.iloc[ev],
                                                       grouped=True, params=PARAMS))
    save(f"phme_verified_two_stage_seed{seed}",
         two_stage_crossfit(Xc.iloc[dev], y[dev], panel[dev], {k: v[dev] for k, v in lv.items()}, Xc.iloc[ev],
                            {k: v[ev] for k, v in lv.items()}, seed, PARAMS))
    log("gbdt seed", seed)
catdf = tab[["ComponentID", "comp_type", "FigureID"]]
num = tab[[c for c in feature_columns(tab, use_ctx=True) if c not in ("ComponentID", "comp_type", *drop)]] \
    .astype(float).values
for seed in range(3):
    fl = folds(y[dev], panel[dev], seed)
    oof, fold_id = np.zeros(len(dev), np.float32), np.full(len(dev), -1)
    evk = np.zeros((len(fl), len(ev)), np.float32)
    for k, (tr, va) in enumerate(fl):
        rtr, rva = dev[tr], dev[va]
        XN_tr, XN_va, XN_ev = prep_numeric(num[rtr], [num[rva], num[ev]])
        cards, (XC_tr, XC_va, XC_ev) = cat_codes(catdf, rtr, [rtr, rva, ev])
        p_va, p_ev = fit_predict(XN_tr, XC_tr, y[rtr], panel[rtr], [XN_va, XN_ev], [XC_va, XC_ev],
                                 [panel[rva], panel[ev]], cards, seed, **CFG)
        oof[va], fold_id[va], evk[k] = p_va, k, p_ev
    save(f"phme_verified_boardset_ctx_seed{seed}", (oof, fold_id, evk))
    log("boardset seed", seed)
(OUT / "phme_verified_done").write_text("done")
