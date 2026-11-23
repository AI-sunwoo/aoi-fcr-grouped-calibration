"""Siemens: 더 강한 손 맥락 기준선 — 묶음 전체 집계 특징의 GBDT (2026-11-23, 3차 검토 반영).

심사 의견: BoardSet이 "우리가 설계하지 못한 맥락"을 배웠다고 쓰려면, 사람이 만든 맥락을 더 넉넉히 준
GBDT와 견줘야 한다. 그래서 타임스탬프 묶음마다 70개 측정값 전부의 평균·최소·최대·표준편차,
(묶음, 검사 종류)별 평균, 묶음 안 백분위 순위를 만들어 기존 맥락 특징에 더한다(라벨은 쓰지 않는다).
설정은 기본 GBDT(models.LGB_PARAMS)와 같고, 보드 단위·날짜 단위 교차적합을 씨앗 3개로 한다.

저장: siemens_eval_lgb_agg_seed<s>.npz (보드 단위), siemens_eval_lgb_agg_day_seed<s>.npz (날짜 단위)
사용: python -I 34_siemens_groupagg.py <tables 폴더> <결과 폴더> <씨앗 개수> <스레드 수>
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fcr_eval import siemens_half  # noqa: E402
from models import LGB_PARAMS, crossfit  # noqa: E402

TAB, OUT, N_SEEDS, NJ = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
PARAMS = dict(LGB_PARAMS, n_jobs=NJ)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


df = pd.read_parquet(TAB / "siemens_rows.parquet")
y = df["class"].values.astype(int)
grp = df["group"].values
day = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d").factorize()[0]
CAT = ["inspection_type", "meta_feat1", "meta_feat2", "meta_feat3", "meta_feat4"]
insp = [c for c in df.columns if c.startswith("inspection_feat")]
ctx = [c for c in df.columns if c.startswith("grp_")]
meas = df[insp].astype("float32")
g = meas.groupby(df["group"].values)
agg = pd.concat([g.transform("mean").add_suffix("_gmean"), g.transform("min").add_suffix("_gmin"),
                 g.transform("max").add_suffix("_gmax"), g.transform("std").add_suffix("_gstd"),
                 meas.groupby([df["group"].values, df["inspection_type"].values]).transform("mean").add_suffix("_gtmean"),
                 g.rank(pct=True).add_suffix("_grank")], axis=1).astype("float32")
X = df[CAT + insp].copy()
for c in CAT:
    X[c] = X[c].astype("category")
X = pd.concat([X, df[ctx], agg], axis=1)
log("features", X.shape)
half = siemens_half(df)
dev, ev = np.arange(half), np.arange(half, len(df))
for seed in range(N_SEEDS):
    for name, gr in [("lgb_agg", grp), ("lgb_agg_day", day)]:
        oof, fold, evs = crossfit(X.iloc[dev], y[dev], gr[dev], seed, X.iloc[ev], grouped=True, params=PARAMS)
        np.savez_compressed(OUT / f"siemens_eval_{name}_seed{seed}.npz", oof=oof, fold=fold, ev=evs)
        log(name, seed)
(OUT / "siemens_agg_done").write_text("done")
