"""PHME 과제 2: 맥락 특징을 묶음별로 하나씩 더하는 실험 (2026-11-23).

대회 1위(Gaffet 외 2022)가 과제 2에 쓴 맥락은 '같은 부품에서 AOI가 잡은 핀 수'(n_rows)와
'같은 보드에서 AOI가 잡은 핀 수'(board_rows)다. 우리가 새로 더한 것이 얼마나 보태는지 따로 잰다.
  F0: 맥락 없음(n_rows는 부품 자체 특징이라 포함)
  F1: + board_rows                      (대회 1위와 같은 보드 맥락)
  F2: + board_comps, board_labels       (보드의 잡힌 부품 수, AOI 판정 종류 수)
  F3: + panel_rows, panel_boards        (패널 단위)
  F4: + samecomp_boards                 (같은 패널의 다른 보드에서 같은 부품 자리가 잡혔는가) = 전체
분할: official, temporal. 씨앗 5개. 문턱값은 개발 기간 패널 단위 OOF로만 정한다.

사용: python -I 25_phme_ctx_ablation.py <tables 폴더> <결과 폴더>
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
from phme_data import CTX_COLS, feature_columns  # noqa: E402

TAB, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, verbose=-1, deterministic=True, force_row_wise=True, n_jobs=16)
STEPS = [("F0_none", []), ("F1_board_rows", ["board_rows"]),
         ("F2_board_all", ["board_rows", "board_comps", "board_labels"]),
         ("F3_panel", ["board_rows", "board_comps", "board_labels", "panel_rows", "panel_boards"]),
         ("F4_full", CTX_COLS)]

tab = pd.read_parquet(TAB / "phme_components.parquet")
tab = tab[tab["date"].notna()].reset_index(drop=True)
for c in ["ComponentID", "comp_type"]:
    tab[c] = tab[c].astype("category")
panel = tab["PanelID"].factorize()[0]
y = tab["y"].values.astype(int)
base_cols = feature_columns(tab, use_ctx=False)
splits = {"official": (np.where(tab["split"] == "train")[0], np.where(tab["split"] == "test")[0]),
          "temporal": (np.where(tab["date"] < "2019-08-01")[0], np.where(tab["date"] >= "2019-08-01")[0])}
out = {}
for split, (dev, ev) in splits.items():
    out[split] = {}
    for name, extra in STEPS:
        X = tab[base_cols + extra]
        rows = []
        for seed in range(5):
            oof = oof_scores(X.iloc[dev], y[dev], panel[dev], seed, grouped=True, params=PARAMS)
            sc = lgb_fit_predict(X.iloc[dev], y[dev], [X.iloc[ev]], seed, PARAMS)[0]
            t = threshold_for_f1(y[dev], oof)
            r = {"eval_ap": average_precision_score(y[ev], sc),
                 "eval_f1": f1_score(y[ev], (sc >= t).astype(int), zero_division=0)}
            for a in [0.02, 0.05, 0.10]:
                r[f"oracle_vol@{a}"] = slip_and_volume(y[ev], sc, threshold_for_slip(y[ev], sc, a))[1]
            rows.append(r)
        df = pd.DataFrame(rows)
        out[split][name] = {c: [float(df[c].mean()), float(df[c].std(ddof=0))] for c in df.columns}
        print(split, name, {c: round(v[0], 4) for c, v in out[split][name].items()}, flush=True)
(OUT / "phme_ctx_ablation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
