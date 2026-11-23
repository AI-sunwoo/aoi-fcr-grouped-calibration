"""묶음 수준 보정 규칙 대조 (2026-11-23, 4판 심사 의견 D3).

심사 의견: 불량을 하나씩 세는 규칙(기대 놓침 규칙)은 불량이 묶여 있으면 보장이 없다. 인용한 묶음 수준 방법
(Dunn et al. 2023 등)을 같은 교차적합 위에서 시험해 보라. 37번과 같은 나눔·씨앗으로 교차적합을 다시 만들고
(그래서 기대 놓침 규칙 값은 37번과 같아야 한다) 세 규칙의 실제 놓침·물량을 잰다.
  pooled : 기대 놓침 규칙(본문) — p = (1 + R) / (n + 1), R = 보류 불량 중 점수가 z 이하인 수(각자 겹 모델 기준)
  gw     : 묶음 가중 — 보류 불량이 있는 묶음 G개가 한 표씩: p = (1 + Σ_g R_g / n_g) / (1 + G)
           (Dunn et al. 2023의 CDF 풀링과 같은 생각; 목표는 '새 묶음에서 고른 불량 하나'의 놓침)
  gboot  : 묶음 부트스트랩 높은 확신 — 보류 불량을 가진 묶음을 복원 추출(B = 1000)해
           p_b = (1 + Σ_g m_bg R_g) / (1 + Σ_g m_bg n_g)를 만들고, 그 (1 − δ) 분위가 α 이하일 때만 통과(δ = 0.05).
           불량 수가 아니라 묶음 수가 정하는 불확실성을 반영한다(근사, 묶음 수가 많아질 때).
보정 묶음(교차적합 겹을 나누는 단위이자 위 규칙의 묶음): ts(타임스탬프 = 보드), day(UTC 날짜).
kind: randboard / randday (20 나눔, 33·37번과 같은 난수 5000+i), chrono(앞 절반 → 뒤 절반, i=0만).

사용: python -I 38_group_rules.py <tables 폴더> <결과 폴더> <randboard|randday|chrono> <시작 i> <개수> <스레드 수>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fcr_eval import siemens_half  # noqa: E402
from fcr_eval import pooled_rank  # noqa: E402
from models import LGB_PARAMS, crossfit  # noqa: E402

TAB, OUT, KIND = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
I0, N, NJ = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
OUT.mkdir(parents=True, exist_ok=True)
PARAMS = dict(LGB_PARAMS, n_jobs=NJ)
MAIN = [0.01, 0.02, 0.05, 0.10]
NS = 1_000_000_000
B, DELTA = 1000, 0.05


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


df = pd.read_parquet(TAB / "siemens_rows.parquet")
y = df["class"].values.astype(int)
grp = df["group"].values
ts = pd.to_datetime(df["timestamp"]).astype("int64").values
day_str = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d").factorize()[0]   # 33·37번과 같은 날짜 번호(나눔용)
GROUPS = {"ts": grp, "day": ts // (86400 * NS)}                                     # 37번과 같은 보정 묶음
CAT = ["inspection_type", "meta_feat1", "meta_feat2", "meta_feat3", "meta_feat4"]
insp = [c for c in df.columns if c.startswith("inspection_feat")]
Xb = df[CAT + insp].copy()
for c in CAT:
    Xb[c] = Xb[c].astype("category")


def group_counts(oof, fold, yd, gdev, ev):
    """R_g(z): 묶음 g의 보류 불량 중 점수가 z 이하인 수(그 묶음이 속한 겹의 모델 기준). 반환 (E×G float32, n_g)."""
    d = np.flatnonzero(yd == 1)
    gu, ginv = np.unique(gdev[d], return_inverse=True)
    Rg = np.zeros((ev.shape[1], len(gu)), np.float32)
    ng = np.zeros(len(gu))
    for j in range(len(gu)):
        idx = d[ginv == j]
        ks = np.unique(fold[idx])
        assert len(ks) == 1, "묶음이 두 겹에 걸침"
        c = np.sort(oof[idx])
        Rg[:, j] = np.searchsorted(c, ev[ks[0]], side="right")
        ng[j] = len(idx)
    return Rg, ng


def metrics(yd, ye, oof, fold, ev, gdev, seed):
    pos, neg = ye == 1, ye == 0
    R, n = pooled_rank(oof, fold, yd, ev)
    Rg, ng = group_counts(oof, fold, yd, gdev, ev)
    assert int(ng.sum()) == n and np.allclose(Rg.sum(1), R)
    G = len(ng)
    p_pool = (1 + R) / (n + 1)
    p_gw = (1 + Rg @ (1 / ng)) / (1 + G)
    rng = np.random.default_rng(10_000 + seed)
    M = rng.multinomial(G, np.full(G, 1 / G), size=B).astype(np.float32)        # B×G
    den = 1 + M @ ng.astype(np.float32)                                         # B
    kq = int(np.ceil((1 - DELTA) * B)) - 1
    q_boot = np.empty(len(ye), np.float32)
    for s in range(0, len(ye), 40_000):
        pb = (1 + Rg[s:s + 40_000] @ M.T) / den                                  # chunk×B
        q_boot[s:s + 40_000] = np.partition(pb, kq, axis=1)[:, kq]
    r = {"n_def_dev": int(n), "n_groups_def": int(G)}
    for a in MAIN:
        for name, ps in [("pooled", p_pool <= a), ("gw", p_gw <= a), ("gboot", q_boot <= a)]:
            r[f"{name}_slip@{a}"], r[f"{name}_vol@{a}"] = float(ps[pos].mean()), float(ps[neg].mean())
    return r


out_file = OUT / f"group_rules_{KIND}_{I0}.json"
res = json.loads(out_file.read_text()) if out_file.exists() else {}
for i in range(I0, I0 + N):
    if KIND == "chrono":
        half = siemens_half(df)
        dev, ev = np.arange(half), np.arange(half, len(df))
    else:
        rng0 = np.random.default_rng(5000 + i)
        unit = grp if KIND == "randboard" else day_str
        u = np.unique(unit)
        is_ev = np.isin(unit, rng0.choice(u, len(u) // 2, replace=False))
        dev, ev = np.flatnonzero(~is_ev), np.flatnonzero(is_ev)
    yd, ye = y[dev], y[ev]
    for g, gv in GROUPS.items():
        key = f"{i}|{g}"
        if key in res:
            continue
        t0 = time.time()
        oof, fold, evs = crossfit(Xb.iloc[dev], yd, gv[dev], i, Xb.iloc[ev], grouped=True, params=PARAMS)
        res[key] = {**metrics(yd, ye, oof, fold, evs, gv[dev], i), "sec": round(time.time() - t0, 1)}
        out_file.write_text(json.dumps(res))
        log(KIND, key, "G", res[key]["n_groups_def"], "slip@5% pooled/gw/gboot",
            *[round(100 * res[key][f"{m}_slip@0.05"], 2) for m in ["pooled", "gw", "gboot"]],
            "vol@5%", *[round(100 * res[key][f"{m}_vol@0.05"], 1) for m in ["pooled", "gw", "gboot"]])
log("done")
