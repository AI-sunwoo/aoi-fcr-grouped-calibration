"""Siemens 대조 나눔을 여러 번 되풀이한다 (2026-11-23, 3차 검토 반영).

28번(controls)은 보드 무작위·날짜 무작위 나눔을 한 번씩만 했다. 나눔 하나의 우연을 빼려고 각 나눔을
서로 다른 난수로 여러 번 만들고(난수 5000+i), GBDT(맥락 없음)를 행 / 보드 / 날짜 단위로 교차적합해
실제 놓침·물량의 분포를 본다. 씨앗 = i. 지표만 json으로 쓴다.

사용: python -I 33_controls_repeat.py <tables 폴더> <결과 폴더> <randboard|randday> <시작 i> <개수> <스레드 수>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fcr_eval import summarise  # noqa: E402
from models import LGB_PARAMS, crossfit  # noqa: E402

TAB, OUT, KIND = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
I0, N, NJ = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
OUT.mkdir(parents=True, exist_ok=True)
PARAMS = dict(LGB_PARAMS, n_jobs=NJ)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


df = pd.read_parquet(TAB / "siemens_rows.parquet")
y = df["class"].values.astype(int)
grp = df["group"].values
day = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d").factorize()[0]
CAT = ["inspection_type", "meta_feat1", "meta_feat2", "meta_feat3", "meta_feat4"]
insp = [c for c in df.columns if c.startswith("inspection_feat")]
Xb = df[CAT + insp].copy()
for c in CAT:
    Xb[c] = Xb[c].astype("category")
out_file = OUT / f"controls_repeat_{KIND}.json"
res = json.loads(out_file.read_text()) if out_file.exists() else {}
for i in range(I0, I0 + N):
    rng = np.random.default_rng(5000 + i)
    unit = grp if KIND == "randboard" else day
    u = np.unique(unit)
    is_ev = np.isin(unit, rng.choice(u, len(u) // 2, replace=False))
    dev, ev = np.flatnonzero(~is_ev), np.flatnonzero(is_ev)
    yd, ye = y[dev], y[ev]
    info = {"n_dev": [int(len(dev)), int(yd.sum())], "n_eval": [int(len(ev)), int(ye.sum())]}
    for level in ["row", "board", "day"]:
        key = f"{i}|{level}"
        if key in res:
            continue
        g = day if level == "day" else grp
        oof, fold, evs = crossfit(Xb.iloc[dev], yd, g[dev], i, Xb.iloc[ev], grouped=(level != "row"), params=PARAMS)
        res[key] = {**info, **summarise(yd, ye, grp[ev], oof, fold, evs)}
        out_file.write_text(json.dumps(res, indent=1))
        log(KIND, key, "slip@1/5/10%", [round(100 * res[key][f"pooled_slip@{a}"], 1) for a in [0.01, 0.05, 0.1]])
log("done")
