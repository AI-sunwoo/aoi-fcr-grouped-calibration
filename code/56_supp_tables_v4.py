"""부록 표 S8(보정 묶음 폭 민감도)·S9(날짜 단위 보정의 모델 짝 차이)를 Markdown으로 만든다 (2026-11-23, 4판).

사용: python3 -I 56_supp_tables_v4.py <analysis_v4.json>
"""
import json
import sys
from pathlib import Path

R = json.loads(Path(sys.argv[1]).read_text())
G = ["row", "ts", "w1m", "w10m", "w1h", "w8h", "day", "day_p2h"]
NAME = {"row": "row-wise", "ts": "timestamp", "w1m": "1-min window", "w10m": "10-min window", "w1h": "1-h window",
        "w8h": "8-h window", "day": "calendar day", "day_p2h": "day, boundary +2 h"}
KIND = {"randboard": "Random boards (20)", "randday": "Random days (20)", "chrono": "Chronological halves (1)"}
A = [0.01, 0.02, 0.05, 0.1]

print("## Table S8. Width of the calibration groups, Siemens data, GBDT without context\n")
print("Same partitions as Table II of the main paper (random boards and random days, 20 each; one seed per partition) and the chronological "
      "halves (one seed, so its timestamp-fold row differs from the three-seed board-wise row of Table I of the main paper). Groups: mean number of calibration groups in the development part. Ratio: realized slip divided by the target, "
      "mean over partitions, for the expected-slip rule. Exceed: partitions (of 20; of 1 for the chronological halves) whose "
      "realized slip exceeded the target at 1 / 2 / 5 / 10%, for the expected-slip rule (E) and the high-confidence rule (H). "
      "Split: per-fold split-conformal variant (Section V of the main paper), ratio at a 5% target and exceedances at 1 / 2 / 5 / 10%. "
      "Vol.: inspection-volume reduction (%) of the expected-slip rule at a 5% target, mean over partitions.\n")
print("| Split | Folds | Groups | Ratio @1% | @2% | @5% [95% interval] | @10% | Exceed E | Exceed H | Split ratio @5% | "
      "Split exceed | Vol. @5% |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
S = R["grouping_sensitivity"]
for k in ["randboard", "randday", "chrono"]:
    for i, g in enumerate(G):
        e = S[k][g]
        ci = e["ratio_ci@0.05"]
        ci_s = f" [{ci[0]:.2f}–{ci[1]:.2f}]" if k != "chrono" else ""
        grp = "—" if g == "row" else f"{e['n_groups_dev']:,.0f}"
        print(f"| {KIND[k] if i == 0 else ''} | {NAME[g]} | {grp} | {e['ratio@0.01']:.2f} | {e['ratio@0.02']:.2f} | "
              f"{e['ratio@0.05']:.2f}{ci_s} | {e['ratio@0.1']:.2f} | "
              f"{' / '.join(str(e[f'exceed@{a}']) for a in A)} | {' / '.join(str(e[f'pac_exceed@{a}']) for a in A)} | "
              f"{e['split_ratio@0.05']:.2f} | {' / '.join(str(e[f'split_exceed@{a}']) for a in A)} | "
              f"{100 * e['vol@0.05']:.1f} |")

MD = R["dayblocked_model_pairs"]
MN = {"two_stage": "Two-stage neighbor", "two_stage_dd": "Two-stage, day-blocked inner folds", "lgb_base": "GBDT, no context",
      "lgb_ctx": "GBDT + context", "lgb_agg": "GBDT + full group aggregates", "boardset_ctx": "BoardSet + context",
      "boardset_noctx": "BoardSet, no hand-made context"}
print("\n## Table S9. Day-blocked calibration, Siemens chronological evaluation half: realized operating points and paired "
      "differences from the two-stage model\n")
dd2, dd5 = MD["two_stage_dd|0.02"], MD["two_stage_dd|0.05"]
ci = lambda x: f"{100 * x[0]:.1f}–{100 * x[1]:.1f}"  # noqa: E731
print("Expected-slip rule, decisions averaged over three seeds; intervals for the operating points themselves are in Table S3; "
      f"for the variant with day-blocked inner folds they are {ci(dd2['slip_ci'])}% slip and {ci(dd2['vol_ci'])}% volume at 2%, "
      f"and {ci(dd5['slip_ci'])}% and {ci(dd5['vol_ci'])}% at 5% (same resamples as the differences). "
      f"Δ: 95% percentile intervals from 2,000 bootstrap resamples of the {R.get('split_counts', {}).get('siemens_eval', {}).get('days', 38)} "
      "evaluation days, the same resamples for all "
      "models, so differences are paired. Δ is the two-stage "
      "model minus the row's model, in percentage points; a positive Δ volume means the two-stage model saved more "
      "inspections.\n")
print("| Model | Target | Slip (%) | Volume (%) | Δ volume | Δ slip |")
print("|---|---|---|---|---|---|")
for m in ["two_stage", "two_stage_dd", "lgb_base", "lgb_ctx", "lgb_agg", "boardset_ctx", "boardset_noctx"]:
    for a in [0.02, 0.05]:
        r = MD.get(f"{m}|{a}")
        if r is None:
            continue
        d = MD.get(f"two_stage - {m}|{a}")
        def f(x, n):
            return f"{x:.{n}f}".replace("-", "−")
        dv = f"{f(100 * d['d_vol'], 1)} ({f(100 * d['d_vol_ci'][0], 1)} to {f(100 * d['d_vol_ci'][1], 1)})" if d else "—"
        ds = f"{f(100 * d['d_slip'], 2)} ({f(100 * d['d_slip_ci'][0], 2)} to {f(100 * d['d_slip_ci'][1], 2)})" if d else "—"
        print(f"| {MN[m] if a == 0.02 else ''} | {int(100 * a)}% | {100 * r['slip']:.1f} | {100 * r['vol']:.1f} | {dv} | {ds} |")
