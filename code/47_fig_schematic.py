"""그림 1: 공정 흐름과 누출 원리 도식 (2026-11-23, 3판에서 날짜 단위 (d) 추가). 데이터는 쓰지 않는다(개념도).

(a) SPI → 리플로 → AOI → FCR 판정 → 자동 통과 / 수작업 확인 → 수리
(b) 패널 → 보드 → AOI 호출. 행 단위 교차검증은 한 보드의 호출을 훈련·검증에 나눠 넣고,
    보드 단위 교차검증은 보드를 통째로 한 겹에 넣는다.

사용: python -I 47_fig_schematic.py <그림 폴더>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

FIG = Path(sys.argv[1])
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
INK, MUTED, TRAIN, VAL, DEF = "#222222", "#777777", "#c9c9c9", "#1f77b4", "#d62728"

fig = plt.figure(figsize=(7.4, 4.6))
# (a) 공정 흐름
ax = fig.add_axes([0.02, 0.60, 0.96, 0.36])
ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")
ax.text(0, 29, "(a) Where false-call reduction sits", fontsize=9, weight="bold", va="top")
boxes = [(1, "Solder paste\ninspection"), (17, "Placement\n+ reflow"), (33, "AOI\n(all calls of a\nboard together)"),
         (51, "FCR model\nscore → p-value")]
for x, t in boxes:
    ax.add_patch(FancyBboxPatch((x, 9), 13, 11, boxstyle="round,pad=0.4", fc="white", ec=INK, lw=0.9))
    ax.text(x + 6.5, 14.5, t, ha="center", va="center")
for x0, x1 in [(14.5, 16.5), (30.5, 32.5), (46.5, 50.5)]:
    ax.add_patch(FancyArrowPatch((x0, 14.5), (x1, 14.5), arrowstyle="-|>", mutation_scale=9, color=INK, lw=0.9))
ax.add_patch(FancyBboxPatch((72, 19), 12, 7, boxstyle="round,pad=0.4", fc="white", ec=INK, lw=0.9))
ax.text(78, 22.5, "auto-pass\n(p ≤ α)", ha="center", va="center")
ax.add_patch(FancyBboxPatch((72, 3), 12, 7, boxstyle="round,pad=0.4", fc="white", ec=INK, lw=0.9))
ax.text(78, 6.5, "manual\nverification", ha="center", va="center")
ax.add_patch(FancyBboxPatch((89, 3), 10, 7, boxstyle="round,pad=0.4", fc="white", ec=INK, lw=0.9))
ax.text(94, 6.5, "repair /\nscrap", ha="center", va="center")
ax.add_patch(FancyArrowPatch((64.5, 16), (71.5, 22.5), arrowstyle="-|>", mutation_scale=9, color=INK, lw=0.9))
ax.add_patch(FancyArrowPatch((64.5, 13), (71.5, 6.5), arrowstyle="-|>", mutation_scale=9, color=INK, lw=0.9))
ax.add_patch(FancyArrowPatch((84.5, 6.5), (88.5, 6.5), arrowstyle="-|>", mutation_scale=9, color=INK, lw=0.9))
ax.text(86, 23.5, "slip if a true\ndefect is passed", color=DEF, va="center", fontsize=7.5)
ax.text(57.5, 4.0, "threshold set on held-out\ndefects of past boards", color=MUTED, ha="center", fontsize=7)


def board(ax, x, y, calls, fold_of_call, title=None):
    ax.add_patch(Rectangle((x, y), 8, 8, fc="#f6f6f6", ec=MUTED, lw=0.7))
    for (cx, cy, is_def), f in zip(calls, fold_of_call):
        ax.scatter(x + cx, y + cy, s=34, marker="o", fc=VAL if f else TRAIN, ec=DEF if is_def else INK,
                   lw=1.6 if is_def else 0.6, zorder=3)
    if title:
        ax.text(x + 4, y + 9, title, ha="center", fontsize=7, color=MUTED)


calls_a = [(2, 6, True), (5.5, 6.5, True), (3, 2.5, False), (6, 2, False)]
calls_b = [(2.5, 5.5, False), (6, 5, False), (4, 2, False)]
calls_c = [(2, 2, True), (5, 6, False)]
# (b) 행 단위 / (c) 보드 단위 / (d) 날짜 단위
PANELS = [("(b) Row-wise folds", [[1, 0, 0, 1], [0, 1, 0], [1, 0]],
           "Calls of one board on both sides:\na held-out defect is scored by a\nmodel that saw its board-mates."),
          ("(c) Board-wise folds", [[1, 1, 1, 1], [0, 0, 0], [0, 0]],
           "Each board is held out whole, but\nboards of the same day stay on\nthe training side."),
          ("(d) Day-blocked folds", [[1, 1, 1, 1], [1, 1, 1], [0, 0]],
           "All boards of a day are held out\ntogether, as new days arrive\nin deployment.")]
for k, (title, folds, note) in enumerate(PANELS):
    ax = fig.add_axes([0.01 + 0.33 * k, 0.04, 0.32, 0.50])
    ax.set_xlim(0, 48); ax.set_ylim(-11, 22); ax.axis("off")
    ax.text(0, 21.5, title, fontsize=8.5, weight="bold", va="top")
    if k < 2:
        ax.add_patch(Rectangle((1, 1), 44, 14, fc="none", ec=INK, lw=0.9, ls="--"))
        ax.text(45, 15.6, "panel", ha="right", fontsize=7, color=MUTED)
    else:
        ax.add_patch(Rectangle((1, 1), 29, 14, fc="none", ec=INK, lw=0.9, ls="--"))
        ax.text(30, 15.6, "day 1", ha="right", fontsize=7, color=MUTED)
        ax.add_patch(Rectangle((31.5, 1), 14.5, 14, fc="none", ec=INK, lw=0.9, ls="--"))
        ax.text(46, 15.6, "day 2", ha="right", fontsize=7, color=MUTED)
    for j, (cl, fo) in enumerate(zip([calls_a, calls_b, calls_c], folds)):
        board(ax, 3 + 14.5 * j, 4, cl, fo, title=f"board {j + 1}")
    ax.text(1, -1.5, note, fontsize=7, va="top", color=INK)
# 범례
lax = fig.add_axes([0.02, 0.535, 0.96, 0.05]); lax.axis("off")
h = [plt.scatter([], [], s=34, fc=TRAIN, ec=INK, lw=0.6), plt.scatter([], [], s=34, fc=VAL, ec=INK, lw=0.6),
     plt.scatter([], [], s=34, fc="white", ec=DEF, lw=1.6)]
lax.legend(h, ["AOI call in training part of fold", "AOI call in held-out part of fold", "true defect (red ring)"],
           loc="center", ncol=3, frameon=False, fontsize=7.5)
fig.savefig(FIG / "fig1_schematic.png", dpi=300)
fig.savefig(FIG / "fig1_schematic.pdf")
print("written")
