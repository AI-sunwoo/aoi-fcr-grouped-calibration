"""그림 4 (2026-11-23, 3판 심사 의견 반영): 보정 곡선과 묶음 폭 민감도. 54번 결과를 그린다.

(a)(b) 보드 무작위·날짜 무작위 20개 나눔에서 기대 놓침 규칙의 실제 놓침(나눔 평균)을 목표 α(0.5~10%)에 대해,
       묶음 정의별로. 대각선 = 목표대로.
(c) 실제 놓침/목표(목표 5%, 나눔 평균과 95% 구간)를 묶음 폭에 대해: 행 → 타임스탬프 → 1분 → 10분 → 1시간 → 8시간 → 날짜.

사용: python -I 55_fig4.py <analysis_v4.json> <그림 폴더>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

R = json.loads(Path(sys.argv[1]).read_text())["grouping_sensitivity"]
FIG = Path(sys.argv[2])
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans"})
GRID = [round(0.005 * k, 3) for k in range(1, 21)]
ORDER = ["row", "ts", "w1m", "w10m", "w1h", "w8h", "day", "day_p2h"]
NAMES = {"row": "row", "ts": "timestamp", "w1m": "1 min", "w10m": "10 min", "w1h": "1 h", "w8h": "8 h", "day": "day",
         "day_p2h": "day (+2 h)"}
COL = {"row": "#d62728", "ts": "#1f77b4", "w1h": "#9467bd", "day": "#2ca02c"}
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
for ax, kind, title in [(axes[0], "randboard", "(a) Random boards, 20 partitions"),
                        (axes[1], "randday", "(b) Random days, 20 partitions")]:
    for g in ["row", "ts", "w1h", "day"]:
        e = R[kind][g]
        ax.plot([100 * a for a in GRID], [100 * a * e[f"ratio@{a}"] for a in GRID], color=COL[g], lw=1.6,
                label=f"{NAMES[g]} folds")
    ax.plot([0, 10], [0, 10], color="k", lw=0.8, ls=":")
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 25)
    ax.set_xlabel("Target slip α (%)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Mean realised slip (%)")
axes[0].legend(fontsize=7.5, frameon=False, loc="upper left")
ax = axes[2]
x = np.arange(len(ORDER))
for kind, col, lab, dx in [("randboard", "#1f77b4", "random boards", -0.12), ("randday", "#2ca02c", "random days", 0.0),
                           ("chrono", "#000000", "chronological halves", 0.12)]:
    if kind not in R:
        continue
    m = np.array([R[kind][g]["ratio@0.05"] for g in ORDER])
    ci = np.array([R[kind][g]["ratio_ci@0.05"] for g in ORDER])
    ax.errorbar(x + dx, m, yerr=[m - ci[:, 0], ci[:, 1] - m], marker="o", ms=3.5, capsize=2, color=col, lw=1.4, label=lab)
ax.axhline(1, color="k", lw=0.8, ls=":")
ax.set_xticks(x)
ax.set_xticklabels([NAMES[g] for g in ORDER], rotation=40, ha="right")
ax.set_ylabel("Realised slip / target at α = 5%")
ax.set_title("(c) Width of the calibration groups")
ax.set_yscale("log")
ax.set_yticks([0.5, 1, 2, 4])
ax.set_yticklabels(["0.5", "1", "2", "4"])
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5, frameon=False)
fig.tight_layout()
fig.savefig(FIG / "fig3_calibration_width.png", dpi=300)
fig.savefig(FIG / "fig3_calibration_width.pdf")
print("written")
