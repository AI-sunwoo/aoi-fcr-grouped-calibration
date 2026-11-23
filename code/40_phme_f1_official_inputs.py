"""PHME 대회 비교용 F1을 대회 입력 규칙으로 다시 잰다 (2026-11-23, 모의 심사 5번 반영).

대회 문서(Task 2): "teams can use all the features available in the SPI dataset as well as the output label of the AOI
(AOILabel)". 우리 특징의 machine_B(MachineID)는 이 목록에 없다. 그래서 machine_B를 뺀 특징으로 공식 나눔(훈련 릴리스
→ 시험 릴리스)을 26번과 같은 설정으로 교차적합하고, 48번과 같은 방식으로 F1을 잰다:
씨앗마다 개발 OOF 확률에서 F1을 가장 크게 하는 문턱값을 정해 시험 확률(다섯 겹 평균)에 적용.
모델: GBDT 맥락 없음 / 맥락, 2단계(기본 설정). 씨앗 5개.

사용: python -I 40_phme_f1_official_inputs.py <tables 폴더> <출력 json>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import threshold_for_f1  # noqa: E402
from models import crossfit, two_stage_crossfit  # noqa: E402
from phme_data import feature_columns  # noqa: E402

TAB, OUT = Path(sys.argv[1]), Path(sys.argv[2])
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, verbose=-1, deterministic=True, force_row_wise=True, n_jobs=6)
tab = pd.read_parquet(TAB / "phme_components.parquet")
tab = tab[tab["date"].notna()].reset_index(drop=True)
for c in ["ComponentID", "comp_type"]:
    tab[c] = tab[c].astype("category")
panel = tab["PanelID"].factorize()[0]
board = (tab["PanelID"] + "_" + tab["FigureID"]).factorize()[0]
pos = (tab["PanelID"] + "_" + tab["ComponentID"].astype(str)).factorize()[0]
y = tab["y"].values.astype(int)
dev, ev = np.where(tab["split"] == "train")[0], np.where(tab["split"] == "test")[0]
yd, ye = y[dev], y[ev]
DROP = ["machine_B"]
assert all(c in tab.columns for c in DROP)
res = {"dropped": DROP, "n_dev": [int(len(dev)), int(yd.sum())], "n_eval": [int(len(ev)), int(ye.sum())], "models": {}}
for name in ["lgb_noctx", "lgb_ctx", "two_stage"]:
    f1s, aps = [], []
    for seed in range(5):
        cols = [c for c in feature_columns(tab, use_ctx=(name != "lgb_noctx")) if c not in DROP]
        X = tab[cols]
        if name == "two_stage":
            lv = {"board": board, "panel": panel, "pos": pos}
            oof, fold, evs = two_stage_crossfit(X.iloc[dev], yd, panel[dev], {k: v[dev] for k, v in lv.items()},
                                                X.iloc[ev], {k: v[ev] for k, v in lv.items()}, seed, PARAMS)
        else:
            oof, fold, evs = crossfit(X.iloc[dev], yd, panel[dev], seed, X.iloc[ev], grouped=True, params=PARAMS)
        raw_ev = evs.mean(0)
        t = threshold_for_f1(yd, oof)
        f1s.append(float(f1_score(ye, (raw_ev >= t).astype(int), zero_division=0)))
        aps.append(float(average_precision_score(ye, raw_ev)))
        print(name, seed, round(f1s[-1], 3), round(aps[-1], 3), flush=True)
    res["models"][name] = {"f1_mean": float(np.mean(f1s)), "f1_sd": float(np.std(f1s, ddof=1)), "f1": f1s,
                           "ap_mean": float(np.mean(aps)), "ap_sd": float(np.std(aps, ddof=1))}
Path(OUT).write_text(json.dumps(res, indent=1))
print(json.dumps({k: (round(v["f1_mean"], 3), round(v["f1_sd"], 3), round(v["ap_mean"], 3)) for k, v in res["models"].items()}))
