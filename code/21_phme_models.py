"""PHME 2022 과제 2: 모델 비교 실험 (2026-11-23). 점수만 저장하고 지표는 40_analyze.py에서 계산한다.

분할
  official : 대회 훈련 → 대회 시험(같은 날짜들, 패널 무작위)
  temporal : 훈련+시험을 합쳐 7월 → 9월
모델
  lgb_noctx : AOI 판정 개수 + SPI 요약 + 부품 이름
  lgb_ctx   : + 같은 보드·패널의 AOI 개수
  two_stage : lgb_ctx 1단계 점수를 같은 보드·같은 패널·같은 패널의 같은 부품 자리에 대해 요약(중첩 교차검증)
개발(훈련) 기간 OOF: 패널 단위 5겹(grouped). lgb_noctx·lgb_ctx는 부품 단위 5겹(leaky)도 저장.

사용: python -I 21_phme_models.py <tables 폴더> <결과 폴더> [씨앗 개수]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import lgb_fit_predict, oof_scores, two_stage  # noqa: E402
from phme_data import feature_columns  # noqa: E402

TAB = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
N_SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, verbose=-1, deterministic=True, force_row_wise=True,
              n_jobs=16)

tab = pd.read_parquet(TAB / "phme_components.parquet")
tab = tab[tab["date"].notna()].reset_index(drop=True)  # SPI와 이어지지 않는 9개 제외
for c in ["ComponentID", "comp_type"]:
    tab[c] = tab[c].astype("category")
tab["panel_code"] = tab["PanelID"].factorize()[0]
tab["board_code"] = (tab["PanelID"] + "_" + tab["FigureID"]).factorize()[0]
tab["pos_code"] = (tab["PanelID"] + "_" + tab["ComponentID"].astype(str)).factorize()[0]
y = tab["y"].values.astype(int)

SPLITS = {
    "official": (np.where(tab["split"] == "train")[0], np.where(tab["split"] == "test")[0]),
    "temporal": (np.where(tab["date"] < "2019-08-01")[0], np.where(tab["date"] >= "2019-08-01")[0]),
}
np.savez_compressed(OUT / "phme_meta.npz", y=y, panel=tab["panel_code"].values, board=tab["board_code"].values,
                    date=tab["date"].astype("int64").values,
                    **{f"{k}_dev": v[0] for k, v in SPLITS.items()}, **{f"{k}_eval": v[1] for k, v in SPLITS.items()})


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


for split, (dev, ev) in SPLITS.items():
    g = tab["panel_code"].values
    for seed in range(N_SEEDS):
        for name, cols in [("lgb_noctx", feature_columns(tab, use_ctx=False)),
                           ("lgb_ctx", feature_columns(tab, use_ctx=True))]:
            cols = [c for c in cols if c not in ("panel_code", "board_code", "pos_code")]
            X = tab[cols]
            oof_g = oof_scores(X.iloc[dev], y[dev], g[dev], seed, grouped=True, params=PARAMS)
            oof_l = oof_scores(X.iloc[dev], y[dev], g[dev], seed, grouped=False, params=PARAMS)
            sc = lgb_fit_predict(X.iloc[dev], y[dev], [X.iloc[ev]], seed, PARAMS)[0]
            np.savez_compressed(OUT / f"phme_{split}_{name}_seed{seed}.npz", oof_grouped=oof_g, oof_leaky=oof_l,
                                eval=sc)
            log(split, name, seed, "done")
        cols = [c for c in feature_columns(tab, use_ctx=True) if c not in ("panel_code", "board_code", "pos_code")]
        X = tab[cols]
        lv = {"board": tab["board_code"].values, "panel": tab["panel_code"].values, "pos": tab["pos_code"].values}
        oof2, sc2 = two_stage(X.iloc[dev], y[dev], g[dev], {k: v[dev] for k, v in lv.items()},
                              X.iloc[ev], {k: v[ev] for k, v in lv.items()}, seed, PARAMS)
        np.savez_compressed(OUT / f"phme_{split}_two_stage_seed{seed}.npz", oof_grouped=oof2, eval=sc2)
        log(split, "two_stage", seed, "done")
(OUT / "phme_models_done.json").write_text(json.dumps({"seeds": N_SEEDS}), encoding="utf-8")
