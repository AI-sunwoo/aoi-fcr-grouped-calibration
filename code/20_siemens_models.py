"""Siemens AOI: 모델 비교 실험 (2026-11-23). 점수만 저장하고 지표는 40_analyze.py에서 계산한다.

기간: 시각순으로 앞 절반 = 개발(dev), 뒤 절반 = 평가(eval). Pfab·Rothering(2025)과 같은 나눔.
모델
  lgb_base : 측정값 + 범주 열(Pfab과 같은 입력)
  lgb_ctx  : + 같은 시각 묶음의 맥락(묶음 크기, 검사 유형별 개수, 측정값의 묶음 평균과의 차이)
  two_stage: lgb_ctx의 1단계 점수를 같은 묶음의 다른 행들에 대해 요약해 2단계 특징으로(중첩 교차검증)
개발 기간 OOF: 묶음 단위 5겹(grouped). lgb_base·lgb_ctx는 행 단위 5겹(leaky)도 함께 저장해 누출 효과를 잰다.
평가 기간 점수: 개발 기간 전체로 학습한 모델.

사용: python -I 20_siemens_models.py <tables 폴더> <결과 폴더> [씨앗 개수]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import lgb_fit_predict, oof_scores, two_stage  # noqa: E402

TAB = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
N_SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
CAT = ["inspection_type", "meta_feat1", "meta_feat2", "meta_feat3", "meta_feat4"]

df = pd.read_parquet(TAB / "siemens_rows.parquet")
y = df["class"].values.astype(int)
groups = df["group"].values
insp = [c for c in df.columns if c.startswith("inspection_feat")]
ctx = [c for c in df.columns if c.startswith("grp_")]
X_base = df[CAT + insp].copy()
for c in CAT:
    X_base[c] = X_base[c].astype("category")
X_ctx = pd.concat([X_base, df[ctx]], axis=1)

n = len(df)
half = n // 2
dev, ev = np.arange(half), np.arange(half, n)
np.savez_compressed(OUT / "siemens_meta.npz", y=y, groups=groups, half=half,
                    ts=df["timestamp"].astype("int64").values, insp_type=df["inspection_type"].values)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


for seed in range(N_SEEDS):
    for name, X in [("lgb_base", X_base), ("lgb_ctx", X_ctx)]:
        oof_g = oof_scores(X.iloc[dev], y[dev], groups[dev], seed, grouped=True)
        oof_l = oof_scores(X.iloc[dev], y[dev], groups[dev], seed, grouped=False)
        sc_ev = lgb_fit_predict(X.iloc[dev], y[dev], [X.iloc[ev]], seed)[0]
        np.savez_compressed(OUT / f"siemens_{name}_seed{seed}.npz", oof_grouped=oof_g, oof_leaky=oof_l, eval=sc_ev)
        log(name, seed, "done")
    lv_dev = {"grp": groups[dev]}
    lv_ev = {"grp": groups[ev]}
    oof2, sc2 = two_stage(X_ctx.iloc[dev], y[dev], groups[dev], lv_dev, X_ctx.iloc[ev], lv_ev, seed)
    np.savez_compressed(OUT / f"siemens_two_stage_seed{seed}.npz", oof_grouped=oof2, eval=sc2)
    log("two_stage", seed, "done")
(OUT / "siemens_models_done.json").write_text(json.dumps({"seeds": N_SEEDS}), encoding="utf-8")
