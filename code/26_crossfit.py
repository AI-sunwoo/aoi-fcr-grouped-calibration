"""교차적합 실험(LightGBM 계열) — 20·21번을 대체한다 (2026-11-23).

운영 모델 = 개발 기간 5겹 교차검증의 다섯 모델. 각 모델의 OOF 불량 점수로 불량 기준 conformal p값을 만들고
평가 기간 점수는 다섯 모델의 p값 평균으로 판정한다(40번 분석에서). 그래서 보정과 운영의 점수 눈금이 같다.
lgb 계열은 행 단위 5겹(누출) 교차적합도 함께 저장해 누출 효과를 잰다.

저장: <ds>_<split>_<model>_seed<s>.npz
  oof, fold, ev(5 x n_eval)                  : 묶음 단위
  oof_l, fold_l, ev_l                        : 행 단위(lgb 계열만)

사용: python -I 26_crossfit.py <phme|siemens> <tables 폴더> <결과 폴더> <씨앗 개수>
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fcr_eval import siemens_half  # noqa: E402
from models import crossfit, two_stage_crossfit  # noqa: E402
from phme_data import feature_columns  # noqa: E402

DS, TAB, OUT, N_SEEDS = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
OUT.mkdir(parents=True, exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save(name, g, l=None):
    d = {"oof": g[0], "fold": g[1], "ev": g[2]}
    if l is not None:
        d |= {"oof_l": l[0], "fold_l": l[1], "ev_l": l[2]}
    np.savez_compressed(OUT / f"{name}.npz", **d)


if DS == "phme":
    PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, verbose=-1, deterministic=True, force_row_wise=True,
                  n_jobs=16)
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
    np.savez_compressed(OUT / "phme_meta.npz", y=y, panel=panel, board=board,
                        **{f"{k}_dev": v[0] for k, v in splits.items()},
                        **{f"{k}_eval": v[1] for k, v in splits.items()})
    for split, (dev, ev) in splits.items():
        for seed in range(N_SEEDS):
            for name, use_ctx in [("lgb_noctx", False), ("lgb_ctx", True)]:
                X = tab[feature_columns(tab, use_ctx=use_ctx)]
                g = crossfit(X.iloc[dev], y[dev], panel[dev], seed, X.iloc[ev], grouped=True, params=PARAMS)
                l = crossfit(X.iloc[dev], y[dev], panel[dev], seed, X.iloc[ev], grouped=False, params=PARAMS)
                save(f"phme_{split}_{name}_seed{seed}", g, l)
                log(split, name, seed)
            X = tab[feature_columns(tab, use_ctx=True)]
            lv = {"board": board, "panel": panel, "pos": pos}
            g = two_stage_crossfit(X.iloc[dev], y[dev], panel[dev], {k: v[dev] for k, v in lv.items()},
                                   X.iloc[ev], {k: v[ev] for k, v in lv.items()}, seed, PARAMS)
            save(f"phme_{split}_two_stage_seed{seed}", g)
            log(split, "two_stage", seed)
else:
    df = pd.read_parquet(TAB / "siemens_rows.parquet")
    y = df["class"].values.astype(int)
    grp = df["group"].values
    CAT = ["inspection_type", "meta_feat1", "meta_feat2", "meta_feat3", "meta_feat4"]
    insp = [c for c in df.columns if c.startswith("inspection_feat")]
    ctx = [c for c in df.columns if c.startswith("grp_")]
    Xb = df[CAT + insp].copy()
    for c in CAT:
        Xb[c] = Xb[c].astype("category")
    Xc = pd.concat([Xb, df[ctx]], axis=1)
    half = siemens_half(df)
    dev, ev = np.arange(half), np.arange(half, len(df))
    np.savez_compressed(OUT / "siemens_meta.npz", y=y, groups=grp, half=half,
                        ts=df["timestamp"].astype("int64").values, insp_type=df["inspection_type"].values)
    for seed in range(N_SEEDS):
        for name, X in [("lgb_base", Xb), ("lgb_ctx", Xc)]:
            g = crossfit(X.iloc[dev], y[dev], grp[dev], seed, X.iloc[ev], grouped=True)
            l = crossfit(X.iloc[dev], y[dev], grp[dev], seed, X.iloc[ev], grouped=False)
            save(f"siemens_eval_{name}_seed{seed}", g, l)
            log(name, seed)
        g = two_stage_crossfit(Xc.iloc[dev], y[dev], grp[dev], {"grp": grp[dev]}, Xc.iloc[ev], {"grp": grp[ev]}, seed)
        save(f"siemens_eval_two_stage_seed{seed}", g)
        log("two_stage", seed)
(OUT / f"{DS}_crossfit_done").write_text("done")
