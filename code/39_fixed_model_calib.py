"""고정 모델 보정 대조 (2026-11-23, 모의 심사 2번 반영).

심사 의견: 교차검증 묶음을 바꾸면 보정 표본뿐 아니라 다섯 점수 모델의 학습 표본도 바뀐다. 모델을 하나로 고정하고
보정 표본만 바꾸는 대조가 필요하다.

설계(반복 r마다, 개발 부분 안에서):
  D_c = 개발 날짜의 20%를 통째로 뺀다(학습에 없는 날짜)
  B_b = 남은 날짜의 보드(타임스탬프 묶음) 20%를 통째로 뺀다(학습한 날짜의 새 보드)
  R_a = 남은 보드의 행 20%를 뺀다(같은 보드의 다른 호출은 학습에 있음)
  T   = 나머지로 GBDT(맥락 없음, 본문과 같은 설정·클래스 가중치) 하나를 학습한다.
세 보정 묶음의 불량 점수를 같은 개수(n = 세 묶음 중 최소)로 무작위로 줄인 뒤, 분할 conformal 기대 놓침 규칙
p(z) = (1 + #{c ≤ s(z)}) / (n + 1) ≤ α 로 문턱값을 정하고 평가 부분의 실제 놓침·물량을 잰다. 모델·평가 자료는 같고
보정 표본만 다르다.
kind: chrono(날짜 경계 시간순 반분: 평가 = 뒤 절반) / randday(33·37번과 같은 날짜 무작위 20 나눔, 시간 변화 없음)

사용: python -I 39_fixed_model_calib.py <tables 폴더> <출력 json> <chrono|randday> <반복 수> <스레드 수>
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fcr_eval import siemens_half  # noqa: E402
from models import LGB_PARAMS, lgb_fit_predict  # noqa: E402

TAB, OUT, KIND, NREP, NJ = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
PARAMS = dict(LGB_PARAMS, n_jobs=NJ)
A = [0.01, 0.02, 0.05, 0.10]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


df = pd.read_parquet(TAB / "siemens_rows.parquet")
y = df["class"].values.astype(int)
grp = df["group"].values
day = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d").factorize()[0]
CAT = ["inspection_type", "meta_feat1", "meta_feat2", "meta_feat3", "meta_feat4"]
insp = [c for c in df.columns if c.startswith("inspection_feat")]
X = df[CAT + insp].copy()
for c in CAT:
    X[c] = X[c].astype("category")

res = json.loads(Path(OUT).read_text()) if Path(OUT).exists() else {}
for r in range(NREP):
    key = f"{KIND}|{r}"
    if key in res:
        continue
    t0 = time.time()
    if KIND == "chrono":
        os.environ["SIEMENS_SPLIT"] = "day"
        h = siemens_half(df)
        dev, ev = np.arange(h), np.arange(h, len(df))
    else:
        rng0 = np.random.default_rng(5000 + r)
        u = np.unique(day)
        is_ev = np.isin(day, rng0.choice(u, len(u) // 2, replace=False))
        dev, ev = np.flatnonzero(~is_ev), np.flatnonzero(is_ev)
    rng = np.random.default_rng(7000 + r)
    dd = np.unique(day[dev])
    out_days = rng.choice(dd, int(round(0.2 * len(dd))), replace=False)
    D_c = dev[np.isin(day[dev], out_days)]
    rest = dev[~np.isin(day[dev], out_days)]
    gb = np.unique(grp[rest])
    out_boards = rng.choice(gb, int(round(0.2 * len(gb))), replace=False)
    B_b = rest[np.isin(grp[rest], out_boards)]
    rest = rest[~np.isin(grp[rest], out_boards)]
    R_a = rng.choice(rest, int(round(0.2 * len(rest))), replace=False)
    T = np.setdiff1d(rest, R_a)
    # R_a의 행 가운데 같은 보드의 다른 호출이 T에 있는 비율(누출 조건 확인)
    share_mates = float(np.isin(grp[R_a], grp[T]).mean())
    s_pools = lgb_fit_predict(X.iloc[T], y[T], [X.iloc[R_a], X.iloc[B_b], X.iloc[D_c], X.iloc[ev]], r, PARAMS)
    pools = {"rows_of_training_boards": (s_pools[0], y[R_a]), "new_boards_training_days": (s_pools[1], y[B_b]),
             "new_days": (s_pools[2], y[D_c])}
    s_ev, ye = s_pools[3], y[ev]
    n_eq = min(int(lab.sum()) for _, lab in pools.values())
    e = {"n_train": int(len(T)), "n_def_pools": {k: int(lab.sum()) for k, (_, lab) in pools.items()}, "n_eq": n_eq,
         "share_rows_with_mates_in_training": share_mates}
    for k, (s, lab) in pools.items():
        c = np.sort(rng.choice(s[lab == 1], n_eq, replace=False))
        p = (1 + np.searchsorted(c, s_ev, side="right")) / (n_eq + 1)
        for a in A:
            ps = p <= a
            e[f"{k}|slip@{a}"] = float(ps[ye == 1].mean())
            e[f"{k}|vol@{a}"] = float(ps[ye == 0].mean())
    res[key] = e
    Path(OUT).write_text(json.dumps(res))
    log(key, "n_eq", n_eq, "slip@5%", *[round(100 * e[f"{k}|slip@0.05"], 2) for k in pools], "sec", round(time.time() - t0))
log("done")
