"""분석 4판 (2026-11-23, 3판 심사 의견 반영).

(a) 보정 묶음 정의의 민감도(37번): 나눔별 실제 놓침/목표의 평균과 나눔 사이 95% 구간(나눔 부트스트랩),
    목표별 초과 나눔 수(20개 중), 보정 곡선(α = 0.5…10%), 분할 conformal 대조, 물량.
(b) 날짜 단위 보정에서 모델 사이 짝 차이(실제 물량·놓침, 날짜 부트스트랩 2,000회) — Siemens 시간순.
(c) 이동 원점의 원점별 표(부록용).
(d) 타임스탬프 간격과 시간 거리별 불량 보드 군집.
(e) 나눔별 불량 보드 수·불량 있는 날 수.

사용: python -I 54_analysis_v4.py <cf 폴더> <v3 폴더> <v4 폴더> <tables 폴더> <출력 json>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fcr_eval import passed  # noqa: E402

RES, V3, V4, TAB, OUT = (Path(a) for a in sys.argv[1:6])
rng = np.random.default_rng(2028)
NB = 2000
DAY_NS = 86_400_000_000_000
GRID = [round(0.005 * k, 3) for k in range(1, 21)]
MAIN = [0.01, 0.02, 0.05, 0.10]
report = {}


def q(v):
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


# =============================================================== (a)
gs = {}
for f in sorted(V4.glob("grouping_sens_*.json")):
    kind = f.stem.split("_")[2]
    for k, v in json.loads(f.read_text()).items():
        i, g = k.split("|")
        gs.setdefault(kind, {}).setdefault(g, {})[int(i)] = v
sens = {}
for kind, d in gs.items():
    sens[kind] = {}
    for g, parts in d.items():
        P = list(parts.values())
        e = {"n_partitions": len(P), "n_groups_dev": float(np.mean([p["n_groups_dev"] for p in P]))}
        for a in GRID:
            ratios = np.array([p[f"slip@{a}"] / a for p in P])
            e[f"ratio@{a}"] = float(ratios.mean())
            e[f"ratio_ci@{a}"] = q([ratios[rng.integers(0, len(P), len(P))].mean() for _ in range(NB)]) if len(P) > 1 else [float(ratios[0])] * 2
            e[f"vol@{a}"] = float(np.mean([p[f"vol@{a}"] for p in P]))
        for a in MAIN:
            e[f"exceed@{a}"] = int(sum(p[f"slip@{a}"] > a for p in P))
            e[f"pac_exceed@{a}"] = int(sum(p[f"pac_slip@{a}"] > a for p in P))
            e[f"split_exceed@{a}"] = int(sum(p[f"split_slip@{a}"] > a for p in P))
            e[f"split_ratio@{a}"] = float(np.mean([p[f"split_slip@{a}"] / a for p in P]))
            e[f"pac_ratio@{a}"] = float(np.mean([p[f"pac_slip@{a}"] / a for p in P]))
            e[f"split_vol@{a}"] = float(np.mean([p[f"split_vol@{a}"] for p in P]))
        sens[kind][g] = e
report["grouping_sensitivity"] = sens
print("(a)", {k: list(v) for k, v in sens.items()}, flush=True)

# =============================================================== (b)
meta = dict(np.load(RES / "siemens_meta.npz"))
y, h, grp = meta["y"], int(meta["half"]), meta["groups"]
yd, ye = y[:h], y[h:]
dayE = (meta["ts"][h:] // DAY_NS).astype(np.int64)
_, dinv = np.unique(dayE, return_inverse=True)
nD = dinv.max() + 1
Dd = np.bincount(dinv, weights=(ye == 1).astype(float), minlength=nD)
Fd = np.bincount(dinv, weights=(ye == 0).astype(float), minlength=nD)
W = rng.multinomial(nD, np.full(nD, 1 / nD), size=NB).astype(float)


def per_day(pattern, a, rule="pooled"):
    fl = sorted(RES.glob(pattern)) or sorted(V4.glob(pattern))
    if not fl:
        return None
    A, B = np.zeros(nD), np.zeros(nD)
    for f in fl:
        d = dict(np.load(f))
        ps = passed(rule, d["oof"], d["fold"], yd, d["ev"], a)
        A += np.bincount(dinv, weights=(ps & (ye == 1)).astype(float), minlength=nD)
        B += np.bincount(dinv, weights=(ps & (ye == 0)).astype(float), minlength=nD)
    return A / len(fl), B / len(fl), len(fl)


MODELS = {"two_stage": "siemens_eval_two_stage_day_seed*.npz", "two_stage_dd": "siemens_eval_two_stage_dd_day_seed*.npz",
          "lgb_base": "siemens_eval_lgb_base_day_seed*.npz", "lgb_ctx": "siemens_eval_lgb_ctx_day_seed*.npz",
          "lgb_agg": "siemens_eval_lgb_agg_day_seed*.npz", "boardset_ctx": "siemens_eval_boardset_ctx_day_seed*.npz",
          "boardset_noctx": "siemens_eval_boardset_noctx_day_seed*.npz"}
md = {}
for a in [0.02, 0.05]:
    cache = {m: per_day(p, a) for m, p in MODELS.items()}
    for m, c in cache.items():
        if c is not None:
            md[f"{m}|{a}"] = {"slip": float(c[0].sum() / Dd.sum()), "vol": float(c[1].sum() / Fd.sum()), "n_seeds": c[2],
                              "slip_ci": q((W @ c[0]) / (W @ Dd)), "vol_ci": q((W @ c[1]) / (W @ Fd))}
    for m in ["lgb_base", "lgb_ctx", "lgb_agg", "boardset_ctx", "boardset_noctx", "two_stage_dd"]:
        if cache.get(m) is None:
            continue
        A1, B1, _ = cache["two_stage"]
        A2, B2, _ = cache[m]
        md[f"two_stage - {m}|{a}"] = {"d_vol": float((B1.sum() - B2.sum()) / Fd.sum()), "d_vol_ci": q((W @ B1 - W @ B2) / (W @ Fd)),
                                      "d_slip": float((A1.sum() - A2.sum()) / Dd.sum()), "d_slip_ci": q((W @ A1 - W @ A2) / (W @ Dd))}
report["dayblocked_model_pairs"] = md
print("(b)", {k: v for k, v in md.items() if " - " in k}, flush=True)

# =============================================================== (c)
r3 = json.loads((V3.parent / "analysis_v3" / "analysis_v3.json").read_text()) if (V3.parent / "analysis_v3" / "analysis_v3.json").exists() else None
ro = {}
for f in sorted(V3.glob("rolling_origin_*.json")):
    ro.update(json.loads(f.read_text()))
report["rolling_origin_table"] = {k: {kk: v[kk] for kk in ["dev_days", "eval_days", "n_dev", "n_eval", "eval_def_boards", "eval_ap", "dev_ap_fold"]
                                      + [f"pooled_slip@{a}" for a in MAIN] + [f"pooled_vol@{a}" for a in MAIN]}
                                  for k, v in ro.items()}

# =============================================================== (d)
df = pd.read_parquet(TAB / "siemens_rows.parquet", columns=["timestamp", "group", "class"])
b = df.assign(ts=pd.to_datetime(df["timestamp"]).astype("int64")).groupby("group").agg(ts=("ts", "first"), y=("class", "max"))
b = b.sort_values("ts")
gap = np.diff(b["ts"].values) / 1e9
gq = np.quantile(gap, [0.1, 0.25, 0.5, 0.75, 0.9])
bins = [0, 10, 60, 600, 3600, 8 * 3600, np.inf]
labels = ["<10 s", "10 s–1 min", "1–10 min", "10 min–1 h", "1–8 h", "≥8 h"]
prev, nxt = b["y"].values[:-1], b["y"].values[1:]
lag = {}
for lo, hi, lab in zip(bins[:-1], bins[1:], labels):
    m = (gap >= lo) & (gap < hi)
    lag[lab] = {"pairs": int(m.sum()), "p_next_def_given_def": float(nxt[m & (prev == 1)].mean()) if (m & (prev == 1)).any() else None,
                "p_next_def_given_ok": float(nxt[m & (prev == 0)].mean()) if (m & (prev == 0)).any() else None}
report["timestamp_gaps"] = {"boards": int(len(b)), "gap_quantiles_s": [float(x) for x in gq],
                            "share_gap_lt_10s": float((gap < 10).mean()), "share_gap_lt_60s": float((gap < 60).mean()),
                            "consecutive_board_clustering": lag, "base_rate": float(b["y"].mean())}
print("(d)", report["timestamp_gaps"], flush=True)

# =============================================================== (e)
day_all = (meta["ts"] // DAY_NS).astype(np.int64)
sp = {}
for name, idx in [("siemens_dev", np.arange(h)), ("siemens_eval", np.arange(h, len(y)))]:
    t = pd.DataFrame({"g": grp[idx], "y": y[idx], "d": day_all[idx]})
    gb = t.groupby("g").agg(y=("y", "max"), d=("d", "first"))
    sp[name] = {"defect_calls": int(t.y.sum()), "defective_boards": int(gb.y.sum()), "days": int(t.d.nunique()),
                "days_with_defect": int(t[t.y == 1].d.nunique())}
pm = dict(np.load(RES / "phme_meta.npz"))
for split in ["official", "temporal"]:
    for part in ["dev", "eval"]:
        idx = pm[f"{split}_{part}"]
        t = pd.DataFrame({"b": pm["board"][idx], "p": pm["panel"][idx], "y": pm["y"][idx]})
        sp[f"phme_{split}_{part}"] = {"defect_calls": int(t.y.sum()), "defective_boards": int(t.groupby("b").y.max().sum()),
                                     "defective_panels": int(t.groupby("p").y.max().sum())}
report["split_counts"] = sp
print("(e)", sp, flush=True)
Path(OUT).write_text(json.dumps(report, indent=1), encoding="utf-8")
print("written", OUT)
