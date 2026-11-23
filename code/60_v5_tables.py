"""5판(모의 심사 반영) 부록 표: S19 고정 모델 보정 대조(39번), S20 간격 둔 이동 원점(32번 ROLL_GAP=7과 간격 0 비교),
PHME 대회 입력 규칙 F1(40번). 요약 JSON과 Markdown을 쓴다.

사용: python3 -I 60_v5_tables.py <v5 폴더> <v3 폴더(간격 0 이동 원점)> <출력 json>
"""
import json
import sys
from pathlib import Path

import numpy as np

V5, V3, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
A = [0.01, 0.02, 0.05, 0.1]
POOL = {"rows_of_training_boards": "held-out calls of training boards", "new_boards_training_days": "new boards from training days",
        "new_days": "days absent from training"}
rep = {}
print("## Table S19. Fixed-model calibration control, Siemens data, GBDT without context\n")
print("One GBDT is trained per repetition; only the calibration sample changes. Within the development part, 20% of the days "
      "are held out whole, then 20% of the boards of the remaining days, then 20% of the calls of the remaining boards (98% of "
      "which have board-mates in training). The defect scores of each held-out pool are subsampled to the same number n "
      "(smallest pool) and a split-conformal expected-slip threshold is set from them. Ratio: realized slip on the evaluation "
      "part divided by the target, mean over 20 repetitions; Exceed: repetitions above the target; Vol.: volume (%) at 5%. "
      "n < 99: repetitions whose pools held fewer than 99 defects, so that no call can pass at a 1% target (realized slip 0); "
      "Ratio @1% (n ≥ 99): mean over the other repetitions.\n")
print("| Evaluation | Calibration sample | n (median) | n < 99 | Ratio @1% | @1% (n ≥ 99) | @2% | @5% | @10% | Exceed (1/2/5/10%) | Vol. @5% |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for kind, lab in [("chrono", "Chronological (day boundary)"), ("randday", "Random days (20 partitions)")]:
    f = V5 / f"fixed_model_{kind}.json"
    if not f.exists():
        continue
    R = list(json.loads(f.read_text()).values())
    rep[kind] = {}
    for j, (k, name) in enumerate(POOL.items()):
        nn = np.array([r["n_eq"] for r in R])
        e = {"n_eq": float(np.median(nn)), "reps": len(R), "n_lt_99": int((nn < 99).sum()),
             "ratio_ge99@0.01": float(np.mean([r[f"{k}|slip@0.01"] / 0.01 for r in R if r["n_eq"] >= 99]))}
        for a in A:
            rat = np.array([r[f"{k}|slip@{a}"] / a for r in R])
            e[f"ratio@{a}"], e[f"exceed@{a}"] = float(rat.mean()), int((rat > 1).sum())
            e[f"vol@{a}"] = float(np.mean([r[f"{k}|vol@{a}"] for r in R]))
        rep[kind][k] = e
        print(f"| {lab if j == 0 else ''} | {name} | {e['n_eq']:.0f} | {e['n_lt_99']} | {e['ratio@0.01']:.2f} | {e['ratio_ge99@0.01']:.2f} | "
              + " | ".join(f"{e[f'ratio@{a}']:.2f}" for a in A[1:])
              + " | " + " / ".join(str(e[f'exceed@{a}']) for a in A) + f" | {100 * e['vol@0.05']:.1f} |")

g7 = sorted(V5.glob("rolling_origin_*_gap7.json"))
g0 = sorted(V3.glob("rolling_origin_*.json"))
if g7:
    d7, d0 = {}, {}
    for f in g7:
        d7.update(json.loads(f.read_text()))
    for f in g0:
        d0.update(json.loads(f.read_text()))
    print("\n## Table S20. Rolling origins with the evaluation window shifted by 7 production days, Siemens data, expected-slip rule, one seed\n")
    print("Sensitivity of the rolling origins to shifting the evaluation window. Same development data, models and thresholds "
          "as Table S7; each evaluation window starts 7 production days with data after the end of development instead of "
          "immediately, so the evaluated boards differ and the last window shortens from 17 to 10 days. Met: origin–target "
          "combinations (of 16) at or below the target; Ratio: realized slip / target, median [max].\n")
    print("| Model | Folds | Met, no gap | Met, 7-day gap | Ratio, no gap | Ratio, 7-day gap |")
    print("|---|---|---|---|---|---|")
    rep["gap"] = {}
    for m, lab in [("lgb_base", "GBDT, no context"), ("lgb_ctx", "GBDT + context")]:
        for i, lv in enumerate(["row", "board", "day"]):
            row = {}
            for tag, d in [("gap0", d0), ("gap7", d7)]:
                rs = [v for k, v in d.items() if k.split("|")[1] == m and k.split("|")[2] == lv]
                rat = np.array([r[f"pooled_slip@{a}"] / a for r in rs for a in A])
                row[tag] = {"met": int((rat <= 1).sum()), "n": int(len(rat)), "median": float(np.median(rat)) if len(rat) else None,
                            "max": float(rat.max()) if len(rat) else None}
            rep["gap"][f"{m}|{lv}"] = row
            f0, f7 = row["gap0"], row["gap7"]
            print(f"| {lab if i == 0 else ''} | {lv} | {f0['met']} / {f0['n']} | {f7['met']} / {f7['n']} | "
                  f"{f0['median']:.2f} [{f0['max']:.2f}] | {f7['median']:.2f} [{f7['max']:.2f}] |")
f1 = V5 / "phme_f1_official_inputs.json"
if f1.exists():
    rep["f1_official_inputs"] = json.loads(f1.read_text())
Path(OUT).write_text(json.dumps(rep, indent=1), encoding="utf-8")
