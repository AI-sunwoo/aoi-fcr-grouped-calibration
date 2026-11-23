"""BoardSet 하이퍼파라미터를 개발 자료의 묶음 단위 5겹 OOF PR-AUC로만 고른다 (평가 자료는 보지 않는다).

PHME는 대회 훈련 자료(official dev), Siemens는 앞 절반(dev)만 쓴다. 씨앗 0, 초기화 1개.
고른 설정은 <결과 폴더>/boardset_cfg_<ds>.json 에 쓰고, 본 실험에서는 n_init=3으로 돌린다.

사용: python -I 23_boardset_hpsearch.py <phme|siemens> <tables 폴더> <결과 폴더>
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
from models import folds  # noqa: E402
from phme_data import feature_columns  # noqa: E402

DS, TAB, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)


def cat_codes(df_cat, fit_rows, rows_list):
    cards, outs = [], [np.zeros((len(r), df_cat.shape[1]), np.int64) for r in rows_list]
    for j, c in enumerate(df_cat.columns):
        vals = pd.Index(pd.unique(df_cat[c].iloc[fit_rows].astype(str)))
        cards.append(len(vals))
        for o, r in zip(outs, rows_list):
            o[:, j] = vals.get_indexer(df_cat[c].iloc[r].astype(str)) + 1
    return cards, outs


if DS == "phme":
    tab = pd.read_parquet(TAB / "phme_components.parquet")
    tab = tab[tab["date"].notna()].reset_index(drop=True)
    y = tab["y"].values.astype(int)
    sets = tab["PanelID"].factorize()[0]
    dev = np.where(tab["split"] == "train")[0]
    catdf = tab[["ComponentID", "comp_type", "FigureID"]]
    cols = [c for c in feature_columns(tab, use_ctx=True) if c not in ("ComponentID", "comp_type")]
    num = tab[cols].astype(float).values
else:
    df = pd.read_parquet(TAB / "siemens_rows.parquet")
    y = df["class"].values.astype(int)
    sets = df["group"].values
    dev = np.arange(len(df) // 2)
    catdf = df[["inspection_type", "meta_feat1", "meta_feat2", "meta_feat3", "meta_feat4"]]
    num = df[[c for c in df.columns if c.startswith(("inspection_feat", "grp_"))]].astype(float).values

base = dict(max_epochs=80, patience=12, val_frac=0.15, lr=1e-3, pw="sqrt")
grid = [dict(base, d=d, layers=L, dropout=p) for d, L, p in itertools.product([64, 128], [1, 2], [0.1, 0.3])]
fold_list = folds(y[dev], sets[dev], 0)
prepped = []
for tr, va in fold_list:
    rtr, rva = dev[tr], dev[va]
    XN_tr, XN_va = prep_numeric(num[rtr], [num[rva]])
    cards, (XC_tr, XC_va) = cat_codes(catdf, rtr, [rtr, rva])
    prepped.append((tr, va, rtr, rva, XN_tr, XN_va, cards, XC_tr, XC_va))
results = []
for cfg in grid:
    t0 = time.time()
    oof = np.zeros(len(dev))
    for tr, va, rtr, rva, XN_tr, XN_va, cards, XC_tr, XC_va in prepped:
        oof[va] = fit_predict(XN_tr, XC_tr, y[rtr], sets[rtr], [XN_va], [XC_va], [sets[rva]], cards, 0, **cfg)[0]
    ap = float(average_precision_score(y[dev], oof))
    results.append({"cfg": cfg, "oof_ap": ap, "sec": round(time.time() - t0, 1)})
    print(time.strftime("%H:%M:%S"), json.dumps(results[-1]), flush=True)
best = max(results, key=lambda r: r["oof_ap"])
cfg = dict(best["cfg"], n_init=3)
(OUT / f"boardset_cfg_{DS}.json").write_text(json.dumps(cfg), encoding="utf-8")
(OUT / f"boardset_hpsearch_{DS}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
print("best", json.dumps(best))
