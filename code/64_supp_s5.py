"""보충자료 표 S5(PHME 맥락 특징 단계별 추가)를 25_phme_ctx_ablation.py의 결과 JSON에서 만든다 (2026-11-23, 재심사 반영).

사용: python3 -I 64_supp_s5.py <phme_ctx_ablation.json> <출력 .md>
"""
import json
import sys
from pathlib import Path

R = json.loads(Path(sys.argv[1]).read_text())
LAB = [("F0_none", "F0: no context (component count only)"), ("F1_board_rows", "F1: + flagged rows on the board"),
       ("F2_board_all", "F2: + flagged components and AOI classes on the board"),
       ("F3_panel", "F3: + flagged rows and boards on the panel"), ("F4_full", "F4: + same position flagged on sibling boards")]
lines = ["## Table S5. Context-feature ablation, PHME", "",
         "GBDT refitted on all development data. Each row adds features to the row above. Oracle volume: inspection-volume "
         "reduction (%) at 5% / 10% slip with the threshold set on the evaluation labels, mean over five seeds of the per-seed value.", "",
         "| Features | Official AP | Official vol. @5% / @10% | Temporal AP | Temporal vol. @5% / @10% |", "|---|---|---|---|---|"]
for k, lab in LAB:
    cells = []
    for s in ["official", "temporal"]:
        e = R[s][k]
        cells += [f"{e['eval_ap'][0]:.3f} ± {e['eval_ap'][1]:.3f}",
                  f"{100 * e['oracle_vol@0.05'][0]:.1f} / {100 * e['oracle_vol@0.1'][0]:.1f}"]
    lines.append(f"| {lab} | " + " | ".join(cells) + " |")
Path(sys.argv[2]).write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
