"""29번의 교차적합 단계만 따로 돌린다 (2026-11-23). 탐색은 29번이 이미 끝냈고 고른 설정을 json에서 읽는다.
나눔 하나, 씨앗 하나만 돌려서 여러 프로세스·여러 장비에 나눌 수 있게 한다. 결과 파일 이름과 내용 형식은 29번과 같다.

사용: python -I 29b_tuned_one.py phme <tables 폴더> <결과 폴더> <official|temporal> <씨앗> <스레드 수>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import crossfit, two_stage_crossfit  # noqa: E402
from phme_data import feature_columns  # noqa: E402

DS, TAB, OUT, SPLIT, SEED, NJ = (sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], int(sys.argv[5]),
                                 int(sys.argv[6]))
assert DS == "phme"
BASE = dict(n_estimators=500, learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            verbose=-1, deterministic=True, force_row_wise=True, n_jobs=NJ)
best = dict(BASE, **json.loads((OUT / f"gbdt_tuned_{DS}.json").read_text())["best"])


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save(name, g):
    np.savez_compressed(OUT / f"{name}.npz", oof=g[0], fold=g[1], ev=g[2])


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
groups, levels = panel, {"board": board, "panel": panel, "pos": pos}
dev, ev = splits[SPLIT]
log("config", {k: best[k] for k in ["num_leaves", "min_child_samples", "spw", "n_jobs"]})
for name, X in [("lgbt_noctx", Xn), ("lgbt_ctx", Xc)]:
    g = crossfit(X.iloc[dev], y[dev], groups[dev], SEED, X.iloc[ev], grouped=True, params=best)
    save(f"{DS}_{SPLIT}_{name}_seed{SEED}", g)
    log(SPLIT, name, SEED)
g = two_stage_crossfit(Xc.iloc[dev], y[dev], groups[dev], {k: v[dev] for k, v in levels.items()},
                       Xc.iloc[ev], {k: v[ev] for k, v in levels.items()}, SEED, best)
save(f"{DS}_{SPLIT}_two_stage_t_seed{SEED}", g)
log(SPLIT, "two_stage_t", SEED, "done")
