"""TCPMT 보충자료 표 S3·S4·S13~S17을 날짜 경계 분할 결과로 다시 만든다 (2026-11-23, 모의 심사 1번 반영).

입력: analysis_v2_db.json(48번), analysis_v3_db.json(51번), board_workload_db.json(59번), tuned_siemens_db.json(63번)
— 모두 results/analysis_v5/.
출력: 표별 Markdown 조각을 <출력 폴더>/S*.md로 쓴다.

사용: python3 -I 61_supp_tcpmt_db.py <analysis_v5 폴더> <출력 폴더>
"""
import json
import sys
from pathlib import Path

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
A2 = json.loads((SRC / "analysis_v2_db.json").read_text())
A3 = json.loads((SRC / "analysis_v3_db.json").read_text())
BW = json.loads((SRC / "board_workload_db.json").read_text())
TS = json.loads((SRC / "tuned_siemens_db.json").read_text())   # 조율 Siemens GBDT (날짜 경계 분할), 63번
AL = [0.01, 0.02, 0.05, 0.1]
NAME = {"lgb_base": "GBDT, no context", "lgb_noctx": "GBDT, no context", "lgb_ctx": "GBDT + context",
        "lgb_agg": "GBDT + full group aggregates", "two_stage": "Two-stage neighbor", "two_stage_dd": "Two-stage, day-blocked inner folds",
        "boardset_single": "BoardSet, single (no other calls visible)", "boardset_noctx": "BoardSet, no hand-made context",
        "boardset_ctx": "BoardSet + context", "boardsetj_noctx": "BoardSet, no hand-made context, July-selected configuration",
        "boardsetj_ctx": "BoardSet + context, July-selected configuration", "lgbt_noctx": "Tuned GBDT, no context",
        "lgbt_base": "Tuned GBDT, no context", "lgbt_ctx": "Tuned GBDT + context", "two_stage_t": "Two-stage, tuned base learners"}
LEV = {"row": "row", "board": "board", "day": "day"}
pct = lambda x: f"{100 * x:.1f}"  # noqa: E731
DAYS = int(A2["siemens"]["eval"].get("n_eval_days", 37)) if isinstance(A2["siemens"]["eval"], dict) else 37


def write(name, text):
    (OUT / f"{name}.md").write_text(text.strip() + "\n", encoding="utf-8")


# ---------------- S3
R = A3["realised"]
lines = ["## Table S3. Realized slip and volume with day-block bootstrap intervals, Siemens chronological evaluation half", "",
         "Day-boundary split (evaluation half: 37 days). Seed-averaged decisions; 95% percentile intervals from 2,000 bootstrap "
         "resamples of the evaluation days. P(>α): share of resamples in which the realized slip exceeded the target.", "",
         "| Model | Folds | Rule | α | Slip (95% interval) | Volume (95% interval) | P(>α) |", "|---|---|---|---|---|---|---|"]
for m in ["lgb_base", "lgb_ctx", "two_stage", "boardset_ctx", "boardset_noctx", "lgb_agg"]:
    for lv in ["row", "board", "day"]:
        for rule, rn in [("pooled", "expected-slip"), ("pac", "high-confidence")]:
            for a in AL:
                k = f"{m}|{lv}|{rule}|{a}"
                if k not in R:
                    continue
                v = R[k]
                lines.append(f"| {NAME[m]} | {lv} | {rn} | {int(100 * a)}% | {pct(v['slip'])} ({pct(v['slip_ci'][0])}–{pct(v['slip_ci'][1])}) | "
                             f"{pct(v['vol'])} ({pct(v['vol_ci'][0])}–{pct(v['vol_ci'][1])}) | {v['p_slip_gt_target']:.2f} |")
write("S3", "\n".join(lines))

# ---------------- S4
P = A3["grouping_pairs"]
lines = ["## Table S4. Paired differences between groupings (expected-slip rule), Siemens chronological evaluation half", "",
         "Day-boundary split. Difference in realized slip and volume (percentage points) between two groupings of the same model, "
         "95% day-block bootstrap interval.", "", "| Model | Groupings | α | Δ slip | Δ volume |", "|---|---|---|---|---|"]
f = lambda x: f"{100 * x:.1f}".replace("-", "−")  # noqa: E731
for k, v in P.items():
    m, g, rule, a = k.split("|")
    if rule != "pooled":
        continue
    lines.append(f"| {NAME[m]} | {g.replace('-', ' − ')} | {int(round(100 * float(a)))}% | {f(v['d_slip'])} ({f(v['d_slip_ci'][0])} to {f(v['d_slip_ci'][1])}) | "
                 f"{f(v['d_vol'])} ({f(v['d_vol_ci'][0])} to {f(v['d_vol_ci'][1])}) |")
write("S4", "\n".join(lines))

# ---------------- S13 (Table I, all data)
lines = ["## Table S13. Grouping of the cross-fitting folds, all data", "",
         "Grouping of the cross-fitting folds and realized operating points, GBDT models, expected-slip rule. Dev. AP: "
         "development AP within held-out folds. Eval. AP: AP on the evaluation data. Slip / volume: realized slip and volume (%) at "
         "target α. Means over seeds (Siemens 3, PHME 5); Siemens: day-boundary split, day-block intervals in Table S3.", "",
         "| Data, model | Folds | Dev. AP | Eval. AP | α = 1% | α = 2% | α = 5% | α = 10% |", "|---|---|---|---|---|---|---|---|"]
for ds, split, models, lab in [("siemens", "eval", ["lgb_base", "lgb_ctx"], "Siemens"), ("phme", "temporal", ["lgb_noctx", "lgb_ctx"], "PHME temporal"),
                               ("phme", "official", ["lgb_noctx", "lgb_ctx"], "PHME official")]:
    L = A2[ds][split]["leakage"]
    for m in models:
        for i, lv in enumerate(["row", "board", "day"]):
            e = L[m][lv]
            fl = "panel-wise" if (ds == "phme" and lv == "board") else {"row": "row-wise", "board": "board-wise", "day": "day-blocked"}[lv]
            cells = " | ".join(f"{pct(e[f'pooled_slip@{a}'][0])} / {pct(e[f'pooled_vol@{a}'][0])}" for a in AL)
            lines.append(f"| {lab + ', ' + NAME[m].replace('GBDT, no context', 'GBDT') if i == 0 else ''} | {fl} | "
                         f"{e['dev_ap_fold'][0]:.3f} | {e['eval_ap'][0]:.3f} | {cells} |")
write("S13", "\n".join(lines))

# ---------------- S14 (AP)
order = ["lgb_noctx", "lgb_ctx", "lgb_agg", "two_stage", "boardset_single", "boardset_noctx", "boardset_ctx", "boardsetj_noctx",
         "boardsetj_ctx", "lgbt_noctx", "lgbt_ctx", "two_stage_t"]
lines = ["## Table S14. Average precision with standard deviations", "",
         "Average precision on the evaluation data, mean ± standard deviation over seeds (PHME 5, Siemens 3; tuned "
         "models 3). Board-wise (PHME: panel-wise) cross-fitting. Siemens: day-boundary split; the tuned Siemens configuration "
         "was re-selected on its development half ("
         f"{TS['config']['best']['num_leaves']} leaves, at least {TS['config']['best']['min_child_samples']} samples per leaf, "
         f"{'square-root' if TS['config']['best']['spw'] == 'sqrt' else 'full'} class weight).", "",
         "| Model | PHME official | PHME temporal | Siemens |", "|---|---|---|---|"]
for m in order:
    cells = []
    for ds, split in [("phme", "official"), ("phme", "temporal"), ("siemens", "eval")]:
        key = {"lgb_noctx": "lgb_base", "lgbt_noctx": "lgbt_base"}.get(m, m) if ds == "siemens" else m
        e = A2[ds][split]["models"].get(key)
        if ds == "siemens" and key in TS["models"]:
            e = TS["models"][key]
        cells.append(f"{e['eval_ap'][0]:.3f} ± {e['eval_ap'][1]:.3f}" if e else "—")
    if all(c == "—" for c in cells):
        continue
    lines.append(f"| {NAME[m]} | " + " | ".join(cells) + " |")
write("S14", "\n".join(lines))

# ---------------- S15 (oracle and calibrated, board-wise)
lines = ["## Table S15. Oracle and calibrated inspection-volume reduction", "",
         "Inspection-volume reduction (%) on the evaluation data. Oracle: threshold set on evaluation labels "
         "(seed-averaged scores). Calibrated: realized slip / volume (%) of the expected-slip rule with board-wise (PHME: "
         "panel-wise) folds, mean over seeds. Siemens: day-boundary split.", "",
         "| Data | Model | Oracle @1% | @2% | @5% | @10% | Calibrated α = 2% | Calibrated α = 5% |", "|---|---|---|---|---|---|---|---|"]
for ds, split, lab, models in [("phme", "official", "PHME official", ["lgb_noctx", "lgb_ctx", "two_stage", "two_stage_t", "boardset_ctx"]),
                               ("phme", "temporal", "PHME temporal", ["lgb_noctx", "lgb_ctx", "two_stage", "two_stage_t", "boardsetj_ctx"]),
                               ("siemens", "eval", "Siemens", ["lgb_base", "lgb_ctx", "lgb_agg", "two_stage", "boardset_noctx", "boardset_ctx"])]:
    for i, m in enumerate(models):
        e = A2[ds][split]["models"].get(m)
        if not e:
            continue
        orc = " | ".join(pct(e[f"ens_oracle@{a}"]) for a in AL)
        cal = " | ".join(f"{pct(e[f'pooled_slip@{a}'][0])} / {pct(e[f'pooled_vol@{a}'][0])}" for a in [0.02, 0.05])
        lines.append(f"| {lab if i == 0 else ''} | {NAME[m]} | {orc} | {cal} |")
write("S15", "\n".join(lines))

# ---------------- S16 (matched volume)
mv = A3["matched_volume_siemens"]
lines = ["## Table S16. Board escapes at matched volume, Siemens", "",
         f"Day-boundary split. Most defects sit on boards with several defects (87.7% of the {BW['n_def']:,} evaluation defects, on "
         f"{BW['n_def_boards']} defective boards). Thresholds are set on the evaluation labels so that each model passes 20%, 40% or "
         "60% of the false calls (seed-averaged scores). Slip: call slip (%); boards: defective boards whose defect calls were all "
         "passed; Δ: difference in the share of such boards from the GBDT without context (percentage points, 95% day-block interval).", "",
         "| Model | 20%: slip / boards | Δ | 40%: slip / boards | Δ | 60%: slip / boards | Δ |", "|---|---|---|---|---|---|---|"]
for m in ["lgb_base", "lgb_ctx", "lgb_agg", "two_stage", "boardset_noctx", "boardset_ctx"]:
    cells = []
    for v in ["0.2", "0.4", "0.6"]:
        e = mv[m][v]
        cells.append(f"{pct(e['slip'])} / {e['esc_n']}")
        cells.append("—" if "d_esc_vs_ref" not in e else f"{f(e['d_esc_vs_ref'])} ({f(e['d_esc_vs_ref_ci'][0])} to {f(e['d_esc_vs_ref_ci'][1])})")
    lines.append(f"| {NAME[m]} | " + " | ".join(cells) + " |")
write("S16", "\n".join(lines))

# ---------------- S17 (board measures and workload of calibrated rules)
lines = ["## Table S17. Board-level measures and workload of the calibrated rules, Siemens", "",
         f"Day-boundary split; expected-slip rule; means over three seeds. Evaluation half: {BW['n_eval']:,} calls, {BW['n_def']:,} "
         f"defects ({BW['per_10k_def']:.1f} per 10,000 calls), {BW['n_boards']:,} boards, {BW['n_def_boards']} defective boards. "
         "≥1 passed: defective boards with at least one auto-passed defect call (an upper bound on boards that can leave with an "
         "unnoticed defect); all passed: defective boards whose defect calls were all auto-passed; touch-free: share of the boards "
         "with at least one AOI call in the evaluation data (timestamp groups; boards without calls never reach the manual station) "
         "whose calls were all auto-passed. Checks avoided and defects passed per 10,000 "
         "calls; the labor saved depends on the time per check and on board handling.", "",
         "| Model | Folds | Target | Slip (%) | Volume (%) | ≥1 passed (%) | All passed (%) | Touch-free (%) | Checks avoided | Defects passed |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for m in ["lgb_base", "lgb_ctx", "lgb_agg", "two_stage", "two_stage_dd", "boardset_noctx", "boardset_ctx"]:
    for lv in ["board", "day"]:
        for a in [0.02, 0.05]:
            v = BW.get(f"{m}|{lv}|{a}")
            if not v:
                continue
            lines.append(f"| {NAME[m]} | {'board-wise' if lv == 'board' else 'day-blocked'} | {int(100 * a)}% | {pct(v['slip'])} | "
                         f"{pct(v['vol'])} | {pct(v['partial_escape'])} | {pct(v['full_escape'])} | {pct(v['touch_free'])} | "
                         f"{v['saved_per_10k']:,.0f} | {v['passed_def_per_10k']:.1f} |")
write("S17", "\n".join(lines))
print("written", sorted(p.name for p in OUT.glob("S*.md")))
