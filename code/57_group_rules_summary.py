"""38번(묶음 수준 보정 규칙 대조) 결과를 모아 요약 JSON과 부록 표 S11(Markdown)을 만든다 (2026-11-23, 4판).

사용: python3 -I 57_group_rules_summary.py <group_rules_*.json 폴더> <출력 json>
"""
import json
import sys
from pathlib import Path

import numpy as np

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
A = [0.01, 0.02, 0.05, 0.1]
RULES = {"pooled": "expected-slip (defects)", "gw": "group-weighted", "gboot": "group-bootstrap"}
KIND = {"randboard": "Random boards (20)", "randday": "Random days (20)", "chrono": "Chronological halves (1)"}
FOLD = {"ts": "board-wise", "day": "day-blocked"}
rng = np.random.default_rng(2030)

d = {}
for f in sorted(p for p in SRC.glob("group_rules_*.json") if "summary" not in p.name):
    kind = f.stem.split("_")[2]
    for k, v in json.loads(f.read_text()).items():
        i, g = k.split("|")
        d.setdefault(kind, {}).setdefault(g, {})[int(i)] = v

summ = {}
for kind, gs in d.items():
    for g, parts in gs.items():
        P = list(parts.values())
        for r in RULES:
            e = {"n_partitions": len(P), "groups_with_defects": float(np.mean([p["n_groups_def"] for p in P]))}
            for a in A:
                rat = np.array([p[f"{r}_slip@{a}"] / a for p in P])
                e[f"ratio@{a}"] = float(rat.mean())
                e[f"max_ratio@{a}"] = float(rat.max())
                if len(P) > 1:
                    bm = [rat[rng.integers(0, len(P), len(P))].mean() for _ in range(2000)]
                    e[f"ratio_ci@{a}"] = [float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975))]
                e[f"exceed@{a}"] = int((rat > 1).sum())
                e[f"vol@{a}"] = float(np.mean([p[f"{r}_vol@{a}"] for p in P]))
            summ[f"{kind}|{g}|{r}"] = e
Path(OUT).write_text(json.dumps(summ, indent=1), encoding="utf-8")

print("## Table S11. Group-level calibration rules, Siemens data, GBDT without context\n")
print("Same partitions and cross-fits as Table 3 (one seed per partition) and the chronological halves (one seed). Rules "
      "(Section 4.3): expected-slip rule counting defects (main rule), group-weighted rule (every calibration group with "
      "held-out defects counts once) and group-bootstrap rule (95th percentile of the p-value over 1,000 resamples of the "
      "groups). Groups: mean number of calibration groups with held-out defects (boards for board-wise, days for "
      "day-blocked folds). Ratio: realised slip divided by the target, mean over partitions. Exceed: partitions whose "
      "realised slip exceeded the target at 1 / 2 / 5 / 10%. Vol.: inspection-volume reduction (%), mean over "
      "partitions.\n")
print("| Split | Folds (groups) | Rule | Ratio @1% | @2% | @5% | @10% | Exceed | Vol. @2% | Vol. @5% |")
print("|---|---|---|---|---|---|---|---|---|---|")
for kind in ["randboard", "randday", "chrono"]:
    first = True
    for g in ["ts", "day"]:
        for j, r in enumerate(RULES):
            e = summ.get(f"{kind}|{g}|{r}")
            if e is None:
                continue
            print(f"| {KIND[kind] if first else ''} | {FOLD[g] + ' (' + format(e['groups_with_defects'], '.0f') + ')' if j == 0 else ''} | "
                  f"{RULES[r]} | " + " | ".join(f"{e[f'ratio@{a}']:.2f}" for a in A) + " | "
                  + " / ".join(str(e[f"exceed@{a}"]) for a in A) + f" | {100 * e['vol@0.02']:.1f} | {100 * e['vol@0.05']:.1f} |")
            first = False
