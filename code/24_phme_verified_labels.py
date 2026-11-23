"""견고성 확인: 수리 단계에서 확인된 라벨로 바꿔도 맥락 특징의 효과가 남는가 (2026-11-23).

첫 작업자가 한 보드를 불량으로 보면 여러 부품을 함께 Bad로 적는 습관이 있다면, 맥락 특징의 효과 일부는
'라벨 습관'을 배운 것일 수 있다. 두 번째 작업자(수리)가 NotPossibleToRepair로 판정한 것만 진짜 불량(1),
Good 또는 FalseScrap은 불량 아님(0)으로 두고, NotYetClassified만 있는 부품은 뺀다.

사용: python -I 24_phme_verified_labels.py <데이터 폴더> <tables 폴더> <결과 폴더>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import slip_and_volume, threshold_for_f1, threshold_for_slip  # noqa: E402
from models import lgb_fit_predict, oof_scores  # noqa: E402
from phme_data import IDS, K, feature_columns  # noqa: E402

DATA, TAB, OUT = Path(sys.argv[1]) / "phme", Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, verbose=-1, deterministic=True, force_row_wise=True, n_jobs=16)

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
keep = ~((tab["y"] == 1) & (tab["rep"] == "NYC"))
tab = tab[keep].reset_index(drop=True)
for c in ["ComponentID", "comp_type"]:
    tab[c] = tab[c].astype("category")
panel = tab["PanelID"].factorize()[0]
y = tab["y_ver"].values.astype(int)
dev, ev = np.where(tab["split"] == "train")[0], np.where(tab["split"] == "test")[0]
out = {"n_dev": [int(len(dev)), int(y[dev].sum())], "n_eval": [int(len(ev)), int(y[ev].sum())]}
for name, use_ctx in [("lgb_noctx", False), ("lgb_ctx", True)]:
    X = tab[feature_columns(tab, use_ctx=use_ctx)].drop(columns=["rep", "y_ver"], errors="ignore")
    rows = []
    for seed in range(5):
        oof = oof_scores(X.iloc[dev], y[dev], panel[dev], seed, grouped=True, params=PARAMS)
        sc = lgb_fit_predict(X.iloc[dev], y[dev], [X.iloc[ev]], seed, PARAMS)[0]
        t = threshold_for_f1(y[dev], oof)
        r = {"eval_ap": average_precision_score(y[ev], sc), "eval_f1": f1_score(y[ev], (sc >= t).astype(int)),
             "dev_ap": average_precision_score(y[dev], oof)}
        for a in [0.02, 0.05, 0.10]:
            r[f"oracle_vol@{a}"] = slip_and_volume(y[ev], sc, threshold_for_slip(y[ev], sc, a))[1]
        rows.append(r)
    df = pd.DataFrame(rows)
    out[name] = {c: [float(df[c].mean()), float(df[c].std(ddof=0))] for c in df.columns}
    print(name, {c: round(v[0], 4) for c, v in out[name].items()}, flush=True)
print(out["n_dev"], out["n_eval"])
(OUT / "phme_verified_labels.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
