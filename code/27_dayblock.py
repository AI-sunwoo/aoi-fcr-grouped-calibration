"""날짜 단위 교차적합 (2026-11-23). 행 → 보드 → 날짜로 묶음을 키우면 남은 초과 놓침이 줄어드는지 본다.

보드(타임스탬프·패널) 단위로 나눠도 같은 날의 이웃 보드는 로트·피더·스텐실 상태를 공유한다.
날짜 단위로 나눈 교차검증으로 문턱값을 보정했을 때 실제 놓침이 목표에 더 가까워지면, 남은 초과의 일부는
'시간 이웃 누출'이고, 그대로면 평가 기간의 변화(drift) 몫이다.

저장: <ds>_<split>_<model>_day_seed<s>.npz (oof, fold, ev)  — 26번과 같은 형식
사용: python -I 27_dayblock.py <phme|siemens> <tables 폴더> <결과 폴더> <씨앗 개수>
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fcr_eval import siemens_half  # noqa: E402
from models import crossfit  # noqa: E402
from phme_data import feature_columns  # noqa: E402

DS, TAB, OUT, N_SEEDS = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
OUT.mkdir(parents=True, exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save(name, g):
    np.savez_compressed(OUT / f"{name}.npz", oof=g[0], fold=g[1], ev=g[2])


if DS == "phme":
    PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, verbose=-1, deterministic=True, force_row_wise=True,
                  n_jobs=16)
    tab = pd.read_parquet(TAB / "phme_components.parquet")
    tab = tab[tab["date"].notna()].reset_index(drop=True)
    for c in ["ComponentID", "comp_type"]:
        tab[c] = tab[c].astype("category")
    day = pd.to_datetime(tab["date"]).dt.strftime("%Y-%m-%d").factorize()[0]
    y = tab["y"].values.astype(int)
    splits = {"official": (np.where(tab["split"] == "train")[0], np.where(tab["split"] == "test")[0]),
              "temporal": (np.where(tab["date"] < "2019-08-01")[0], np.where(tab["date"] >= "2019-08-01")[0])}
    for split, (dev, ev) in splits.items():
        log(split, "days in dev", len(np.unique(day[dev])))
        for seed in range(N_SEEDS):
            for name, use_ctx in [("lgb_noctx", False), ("lgb_ctx", True)]:
                X = tab[feature_columns(tab, use_ctx=use_ctx)]
                g = crossfit(X.iloc[dev], y[dev], day[dev], seed, X.iloc[ev], grouped=True, params=PARAMS)
                save(f"phme_{split}_{name}_day_seed{seed}", g)
                log(split, name, seed)
else:
    df = pd.read_parquet(TAB / "siemens_rows.parquet")
    y = df["class"].values.astype(int)
    day = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d").factorize()[0]
    CAT = ["inspection_type", "meta_feat1", "meta_feat2", "meta_feat3", "meta_feat4"]
    insp = [c for c in df.columns if c.startswith("inspection_feat")]
    ctx = [c for c in df.columns if c.startswith("grp_")]
    Xb = df[CAT + insp].copy()
    for c in CAT:
        Xb[c] = Xb[c].astype("category")
    Xc = pd.concat([Xb, df[ctx]], axis=1)
    half = siemens_half(df)
    dev, ev = np.arange(half), np.arange(half, len(df))
    log("days in dev", len(np.unique(day[dev])), "days in eval", len(np.unique(day[ev])),
        "days split across halves", len(np.intersect1d(day[dev], day[ev])))
    for seed in range(N_SEEDS):
        for name, X in [("lgb_base", Xb), ("lgb_ctx", Xc)]:
            g = crossfit(X.iloc[dev], y[dev], day[dev], seed, X.iloc[ev], grouped=True)
            save(f"siemens_eval_{name}_day_seed{seed}", g)
            log(name, seed)
(OUT / f"{DS}_day_done").write_text("done")
