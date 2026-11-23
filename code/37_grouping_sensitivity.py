"""보정 묶음 정의의 민감도 + 보정 곡선 + 분할 conformal 대조 (2026-11-23, 3판 심사 의견 반영).

심사 의견: Siemens의 '보드' = 같은 타임스탬프. 한 보드가 여러 타임스탬프에 걸치면, 날짜 단위가 없애는 것의 일부는
'날짜 안의 가까운 시각 누출'일 수 있다. 그래서 같은 나눔(33번과 같은 난수 5000+i)에서 보정 묶음만 바꾼다:
  row(묶음 없음) / ts(같은 타임스탬프) / w1m·w10m·w1h·w8h(고정 시간 창) / day(UTC 날짜) / day_p2h(2시간 민 날짜, 현지 시각 가정)
각 교차적합에서
  - 기대 놓침 규칙(합친 순위)의 실제 놓침·물량: α = 0.5% … 10% (0.5% 간격) → 보정 곡선
  - 높은 확신 규칙: α = 1, 2, 5, 10%
  - 분할 conformal 대조: 겹 모델 하나와 그 겹의 보류 불량만으로 정한 규칙의 실제 놓침(다섯 겹 평균) — 교차적합 때문에
    보수적인지 가리려는 것
kind: randboard / randday (20 나눔, 33번과 같은 나눔), chrono (앞 절반 → 뒤 절반, i=0만).
모델: GBDT 맥락 없음(models.LGB_PARAMS), 씨앗 = i.

사용: python -I 37_grouping_sensitivity.py <tables 폴더> <결과 폴더> <randboard|randday|chrono> <시작 i> <개수> <스레드 수>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fcr_eval import siemens_half  # noqa: E402
from fcr_eval import pac_r, pooled_rank  # noqa: E402
from models import LGB_PARAMS, crossfit  # noqa: E402

TAB, OUT, KIND = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
I0, N, NJ = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
OUT.mkdir(parents=True, exist_ok=True)
PARAMS = dict(LGB_PARAMS, n_jobs=NJ)
GRID = [round(0.005 * k, 3) for k in range(1, 21)]
MAIN = [0.01, 0.02, 0.05, 0.10]
NS = 1_000_000_000


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


df = pd.read_parquet(TAB / "siemens_rows.parquet")
y = df["class"].values.astype(int)
grp = df["group"].values
ts = pd.to_datetime(df["timestamp"]).astype("int64").values
day_str = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d").factorize()[0]   # 33번과 같은 날짜 번호
GROUPS = {"row": grp, "ts": grp, "w1m": ts // (60 * NS), "w10m": ts // (600 * NS), "w1h": ts // (3600 * NS),
          "w8h": ts // (8 * 3600 * NS), "day": ts // (86400 * NS), "day_p2h": (ts + 2 * 3600 * NS) // (86400 * NS)}
CAT = ["inspection_type", "meta_feat1", "meta_feat2", "meta_feat3", "meta_feat4"]
insp = [c for c in df.columns if c.startswith("inspection_feat")]
Xb = df[CAT + insp].copy()
for c in CAT:
    Xb[c] = Xb[c].astype("category")


def metrics(yd, ye, oof, fold, ev):
    R, n = pooled_rank(oof, fold, yd, ev)
    p = (1 + R) / (n + 1)
    pos, neg = ye == 1, ye == 0
    r = {"n_def_dev": int(n)}
    for a in GRID:
        ps = p <= a
        r[f"slip@{a}"], r[f"vol@{a}"] = float(ps[pos].mean()), float(ps[neg].mean())
    for a in MAIN:
        ps = R <= pac_r(n, a)
        r[f"pac_slip@{a}"], r[f"pac_vol@{a}"] = float(ps[pos].mean()), float(ps[neg].mean())
        sl, vo = [], []
        for k in range(ev.shape[0]):
            c = np.sort(oof[(fold == k) & (yd == 1)])
            pk = (1 + np.searchsorted(c, ev[k], side="right")) / (len(c) + 1)
            sl.append((pk <= a)[pos].mean())
            vo.append((pk <= a)[neg].mean())
        r[f"split_slip@{a}"], r[f"split_vol@{a}"] = float(np.mean(sl)), float(np.mean(vo))
    return r


out_file = OUT / f"grouping_sens_{KIND}_{I0}.json"
res = json.loads(out_file.read_text()) if out_file.exists() else {}
for i in range(I0, I0 + N):
    if KIND == "chrono":
        half = siemens_half(df)
        dev, ev = np.arange(half), np.arange(half, len(df))
    else:
        rng = np.random.default_rng(5000 + i)
        unit = grp if KIND == "randboard" else day_str
        u = np.unique(unit)
        is_ev = np.isin(unit, rng.choice(u, len(u) // 2, replace=False))
        dev, ev = np.flatnonzero(~is_ev), np.flatnonzero(is_ev)
    yd, ye = y[dev], y[ev]
    for g, gv in GROUPS.items():
        key = f"{i}|{g}"
        if key in res:
            continue
        t0 = time.time()
        oof, fold, evs = crossfit(Xb.iloc[dev], yd, gv[dev], i, Xb.iloc[ev], grouped=(g != "row"), params=PARAMS)
        res[key] = {"n_groups_dev": int(len(np.unique(gv[dev]))), **metrics(yd, ye, oof, fold, evs),
                    "sec": round(time.time() - t0, 1)}
        out_file.write_text(json.dumps(res))
        log(KIND, key, "groups", res[key]["n_groups_dev"], "slip@1/5%", round(100 * res[key]["slip@0.01"], 2),
            round(100 * res[key]["slip@0.05"], 2), "split@5%", round(100 * res[key]["split_slip@0.05"], 2))
log("done")
