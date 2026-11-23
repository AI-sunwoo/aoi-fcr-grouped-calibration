"""IEEE TCPMT판 그림 1~4 (2026-11-23). 인쇄 크기 그대로 그린다(한 단 3.5 in, 두 단 7.16 in), 글자 7~8 pt, 미국식 철자.

Fig. 1: 개념도(데이터 없음), 한 단 폭.
Fig. 2: 시간순 다섯 구간의 실제 놓침(51번 fig2_slices) — 52번 Fig. 2와 같은 값.
Fig. 3: 보정 곡선과 묶음 폭(54번 grouping_sensitivity) — 55번과 같은 값.
Fig. 4: 놓침–물량 곡선과 보정 점(교차적합 점수) — 52번 Fig. 3(옛 번호)과 같은 값.

사용: python -I 58_figs_ieee.py <cf 폴더> <analysis_v3b.json> <analysis_v4.json> <그림 폴더> [7월 설정 보드 단위 폴더] [7월 설정 날짜 단위 폴더]
      (뒤 두 인자를 주면 그림 4의 PHME 시간순 BoardSet은 7월 패널로 고른 설정의 결과를 쓴다 — 모의 심사 5번)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
RES, A3, A4, FIG = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), \
    json.loads(Path(sys.argv[3]).read_text()), Path(sys.argv[4])
FIG.mkdir(parents=True, exist_ok=True)
JULY_B = Path(sys.argv[5]) if len(sys.argv) > 5 else None
JULY_D = Path(sys.argv[6]) if len(sys.argv) > 6 else None
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.titlesize": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                     "pdf.fonttype": 42})
INK, MUTED, TRAIN, VAL, DEF = "#222222", "#666666", "#c9c9c9", "#1f77b4", "#d62728"


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 1 (single column)
fig = plt.figure(figsize=(3.5, 2.75))
ax = fig.add_axes([0, 0.55, 1, 0.45])
ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
ax.text(1, 39, "(a) FCR after AOI", fontsize=8, weight="bold", va="top")
for x, t_ in [(1, "SPI"), (17, "Reflow"), (33, "AOI\n(board)"), (49, "FCR\np-value")]:
    ax.add_patch(FancyBboxPatch((x, 12), 12.5, 13, boxstyle="round,pad=0.6", fc="white", ec=INK, lw=0.8))
    ax.text(x + 6.25, 18.5, t_, ha="center", va="center", fontsize=7)
for x0, x1 in [(14.3, 16.4), (30.3, 32.4), (46.3, 48.4)]:
    ax.add_patch(FancyArrowPatch((x0, 18.5), (x1, 18.5), arrowstyle="-|>", mutation_scale=7, color=INK, lw=0.8))
ax.add_patch(FancyBboxPatch((75, 22), 23, 12, boxstyle="round,pad=0.6", fc="white", ec=INK, lw=0.8))
ax.text(86.5, 28, "auto-pass\n(p ≤ α)", ha="center", va="center", fontsize=7)
ax.add_patch(FancyBboxPatch((75, 3), 23, 12, boxstyle="round,pad=0.6", fc="white", ec=INK, lw=0.8))
ax.text(86.5, 9, "manual\ncheck", ha="center", va="center", fontsize=7)
ax.add_patch(FancyArrowPatch((63, 21), (74.2, 27), arrowstyle="-|>", mutation_scale=7, color=INK, lw=0.8))
ax.add_patch(FancyArrowPatch((63, 16), (74.2, 10), arrowstyle="-|>", mutation_scale=7, color=INK, lw=0.8))
ax.text(99, 38, "slip = defect auto-passed", color=DEF, ha="right", va="center", fontsize=6.5)
ax.text(33, 4.5, "threshold from held-out defects of past boards", color=MUTED, ha="center", va="center", fontsize=6.5)


def board(ax, x, y, calls, fold_of_call):
    ax.add_patch(Rectangle((x, y), 9, 9, fc="#f4f4f4", ec=MUTED, lw=0.6))
    for (cx, cy, is_def), f in zip(calls, fold_of_call):
        ax.scatter(x + cx, y + cy, s=16, marker="o", fc=VAL if f else TRAIN, ec=DEF if is_def else INK,
                   lw=1.3 if is_def else 0.5, zorder=3)


calls = [[(2.5, 6.5, True), (6.5, 6.5, True), (2.5, 2.5, False), (6.5, 2.5, False)],
         [(2.5, 6, False), (6.5, 6.5, False), (4.5, 2.5, False)],
         [(3, 3, True), (6.5, 6.5, False)]]
PANELS = [("(b) Row-wise", [[1, 0, 0, 1], [0, 1, 0], [1, 0]], None),
          ("(c) Board-wise", [[1, 1, 1, 1], [0, 0, 0], [0, 0]], None),
          ("(d) Day-blocked", [[1, 1, 1, 1], [1, 1, 1], [0, 0]], "day")]
for k, (title, folds, kind) in enumerate(PANELS):
    ax = fig.add_axes([0.005 + 0.333 * k, 0.13, 0.325, 0.40])
    ax.set_xlim(0, 40); ax.set_ylim(0, 26); ax.axis("off")
    ax.text(0.5, 25.5, title, fontsize=7.5, weight="bold", va="top")
    if kind is None:
        ax.add_patch(Rectangle((0.5, 1), 38.5, 16, fc="none", ec=INK, lw=0.7, ls="--"))
        ax.text(39, 18, "panel", ha="right", fontsize=6.5, color=MUTED)
    else:
        ax.add_patch(Rectangle((0.5, 1), 25, 16, fc="none", ec=INK, lw=0.7, ls="--"))
        ax.text(25.5, 18, "day 1", ha="right", fontsize=6.5, color=MUTED)
        ax.add_patch(Rectangle((26.5, 1), 12.5, 16, fc="none", ec=INK, lw=0.7, ls="--"))
        ax.text(39, 18, "day 2", ha="right", fontsize=6.5, color=MUTED)
    for j in range(3):
        board(ax, 2 + 12.5 * j, 4, calls[j], folds[j])
lax = fig.add_axes([0, 0.0, 1, 0.12]); lax.axis("off")
h = [plt.scatter([], [], s=16, fc=TRAIN, ec=INK, lw=0.5), plt.scatter([], [], s=16, fc=VAL, ec=INK, lw=0.5),
     plt.scatter([], [], s=16, fc="white", ec=DEF, lw=1.3)]
lax.legend(h, ["training part", "held-out part", "true defect"], loc="center", ncol=3, frameon=False, fontsize=7,
           handletextpad=0.3, columnspacing=1.2)
save(fig, "fig1_schematic_ieee")

# ---------------------------------------------------------------- Fig. 2
f2 = A3["fig2_slices"]
LEVEL = [("row", "#d62728", "row-wise folds", -0.12), ("board", "#1f77b4", "board-wise folds", 0.0),
         ("day", "#2ca02c", "day-blocked folds", 0.12)]
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.0))
for ax, a in zip(axes, [0.01, 0.02, 0.05]):
    for level, col, lab, dx in LEVEL:
        s = np.array([100 * f2[f"{i}|{level}|{a}"]["slip"] for i in range(1, 6)])
        ci = np.array([[100 * x for x in f2[f"{i}|{level}|{a}"]["ci"]] for i in range(1, 6)])
        ax.errorbar(np.arange(1, 6) + dx, s, yerr=[s - ci[:, 0], ci[:, 1] - s], marker="o", ms=2.8, capsize=1.5,
                    color=col, label=lab, lw=1.1, elinewidth=0.8)
    ax.axhline(a * 100, color="k", lw=0.7, ls=":")
    ax.set_title(f"Target slip {a:.0%}")
    ax.set_xticks(np.arange(1, 6))
    ax.set_xticklabels([f"{i}\n({f2[f'{i}|board|{a}']['defects']})" for i in range(1, 6)])
    ax.grid(alpha=0.3, lw=0.5)
axes[1].set_xlabel("Chronological evaluation slice (defects)")
axes[0].set_ylabel("Realized slip (%)")
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.8)
save(fig, "fig2_slices_ieee")

# ---------------------------------------------------------------- Fig. 3
R = A4["grouping_sensitivity"]
GRID = [round(0.005 * k, 3) for k in range(1, 21)]
ORDER = ["row", "ts", "w1m", "w10m", "w1h", "w8h", "day", "day_p2h"]
NAMES = {"row": "row", "ts": "timestamp", "w1m": "1 min", "w10m": "10 min", "w1h": "1 h", "w8h": "8 h", "day": "day",
         "day_p2h": "day +2 h"}
COL = {"row": "#d62728", "ts": "#1f77b4", "w1h": "#9467bd", "day": "#2ca02c"}
LBL = {"row": "row", "ts": "timestamp", "w1h": "1-h window", "day": "calendar day"}
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.05))
for ax, kind, title in [(axes[0], "randboard", "(a) Random boards"), (axes[1], "randday", "(b) Random days")]:
    for g in ["row", "ts", "w1h", "day"]:
        e = R[kind][g]
        ax.plot([100 * a for a in GRID], [100 * a * e[f"ratio@{a}"] for a in GRID], color=COL[g], lw=1.2,
                label=f"{LBL[g]} folds")
    ax.plot([0, 10], [0, 10], color="k", lw=0.7, ls=":")
    ax.set_xlim(0, 10.2); ax.set_ylim(0, 32); ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel("Target slip α (%)")
    ax.set_title(title)
    ax.grid(alpha=0.3, lw=0.5)
axes[0].set_ylabel("Mean realized slip (%)")
axes[0].legend(frameon=False, loc="upper left", handlelength=1.4)
ax = axes[2]
x = np.arange(len(ORDER))
for kind, col, lab, dx in [("randboard", "#1f77b4", "random boards", -0.12), ("randday", "#2ca02c", "random days", 0.0),
                           ("chrono", "#000000", "chronological", 0.12)]:
    m = np.array([R[kind][g]["ratio@0.05"] for g in ORDER])
    ci = np.array([R[kind][g]["ratio_ci@0.05"] for g in ORDER])
    ax.errorbar(x + dx, m, yerr=[m - ci[:, 0], ci[:, 1] - m], marker="o", ms=2.8, capsize=1.5, color=col, lw=1.1,
                elinewidth=0.8, label=lab)
ax.axhline(1, color="k", lw=0.7, ls=":")
ax.set_xticks(x)
ax.set_xticklabels([NAMES[g] for g in ORDER], fontsize=6.8, rotation=40, ha="right", rotation_mode="anchor")
ax.set_ylabel("Realized / target slip at 5%")
ax.set_title("(c) Width of calibration groups")
ax.set_yscale("log")
ax.set_yticks([0.5, 1, 2, 4]); ax.set_yticklabels(["0.5", "1", "2", "4"])
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.grid(alpha=0.3, lw=0.5)
ax.legend(frameon=False, loc="upper right", handlelength=1.4)
fig.tight_layout(w_pad=0.8)
save(fig, "fig3_calibration_width_ieee")

# ---------------------------------------------------------------- Fig. 4
from common import slip_and_volume, threshold_for_slip  # noqa: E402
from fcr_eval import passed  # noqa: E402
LABEL = {"noctx": "GBDT, no context", "ctx": "GBDT + context", "two_stage": "Two-stage neighbor",
         "boardset_ctx": "BoardSet + context"}
COLOR = {"noctx": "#7f7f7f", "ctx": "#1f77b4", "two_stage": "#ff7f0e", "boardset_ctx": "#2ca02c"}
pm = dict(np.load(RES / "phme_meta.npz"))
sm = dict(np.load(RES / "siemens_meta.npz"))
h = int(sm["half"])
M = {"noctx": "lgb_noctx", "ctx": "lgb_ctx", "two_stage": "two_stage", "boardset_ctx": "boardset_ctx"}
cases = [("phme", "official", pm["y"][pm["official_dev"]], pm["y"][pm["official_eval"]], "PHME, official test", M),
         ("phme", "temporal", pm["y"][pm["temporal_dev"]], pm["y"][pm["temporal_eval"]], "PHME, July → September", M),
         ("siemens", "eval", sm["y"][:h], sm["y"][h:], "Siemens, second half", dict(M, noctx="lgb_base"))]
slips = np.linspace(0, 0.10, 41)
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2), sharey=True)
for ax, (ds, split, ydev, yev, title, models) in zip(axes, cases):
    for key, m in models.items():
        july = JULY_B is not None and ds == "phme" and split == "temporal" and key == "boardset_ctx"
        files = sorted(JULY_B.glob("phme_temporal_boardsetj_ctx_seed*.npz")) if july else sorted(RES.glob(f"{ds}_{split}_{m}_seed*.npz"))
        dsl = [dict(np.load(f)) for f in files]
        s = np.mean([d["ev"].mean(0) for d in dsl], axis=0)
        v = [slip_and_volume(yev, s, threshold_for_slip(yev, s, a))[1] * 100 for a in slips]
        ax.plot(slips * 100, v, color=COLOR[key], label=LABEL[key], lw=1.1)
        for level, fill in [("board", "white"), ("day", COLOR[key])]:
            day_dir = JULY_D if (july and JULY_D is not None) else RES
            dl = dsl if level == "board" else [dict(np.load(f)) for f in sorted(day_dir.glob(f"{ds}_{split}_{m}_day_seed*.npz"))]
            if not dl:
                continue
            for a, mk in [(0.02, "o"), (0.05, "s")]:
                pts = []
                for d in dl:
                    ps = passed("pooled", d["oof"], d["fold"], ydev, d["ev"], a)
                    pts.append([(ps & (yev == 1)).sum() / yev.sum() * 100, (ps & (yev == 0)).sum() / (yev == 0).sum() * 100])
                ax.scatter(*np.mean(pts, axis=0), marker=mk, s=16, facecolor=fill, edgecolor=COLOR[key], lw=1.0, zorder=4)
    ax.set_title(title)
    ax.set_xlabel("Slip (%)")
    ax.set_xlim(0, 10.5)
    ax.grid(alpha=0.3, lw=0.5)
axes[0].set_ylabel("Volume reduction (%)")
h_, l_ = axes[2].get_legend_handles_labels()
for mk, lab in [("o", "target 2%"), ("s", "target 5%")]:
    h_.append(plt.scatter([], [], marker=mk, s=16, facecolor="white", edgecolor="k", lw=1.0))
    l_.append(f"calibrated, {lab}")
h_.append(plt.scatter([], [], marker="o", s=16, facecolor="k", edgecolor="k"))
l_.append("filled: day-blocked")
fig.legend(h_, l_, loc="lower center", ncol=4, frameon=False, handletextpad=0.4, columnspacing=1.0)
fig.tight_layout(rect=(0, 0.19, 1, 1), w_pad=0.6)
save(fig, "fig4_slip_volume_ieee")
print("written", FIG)
