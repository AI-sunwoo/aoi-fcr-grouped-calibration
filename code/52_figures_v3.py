"""논문 그림 3판 (2026-11-23, 2판 심사 의견 반영).

Fig. 2: Siemens 평가 절반 다섯 구간의 실제 놓침 — 행 / 보드 / 날짜 단위 보정, GBDT(맥락 없음),
        기대 놓침 규칙, 목표 1%·2%·5%. 막대 = 구간 안 날짜 묶음 부트스트랩 95% 구간(51번 결과).
Fig. 3: 놓침-물량 곡선(다섯 겹 확률 평균의 씨앗 평균, 평가 정답으로 문턱값을 훑음 — 순위 품질)과
        보정 규칙의 실제 점: 보드 단위(빈 표시)·날짜 단위(채운 표시), 목표 2%(원)·5%(네모), 씨앗 평균.

사용: python -I 52_figures_v3.py <cf 폴더> <analysis_v3.json> <그림 폴더>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import slip_and_volume, threshold_for_slip  # noqa: E402
from fcr_eval import passed  # noqa: E402

RES, A3, FIG = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3])
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans"})

# ---------------- Fig. 2
f2 = A3["fig2_slices"]
LEVEL = [("row", "#d62728", "row-wise folds", -0.12), ("board", "#1f77b4", "board-wise folds", 0.0),
         ("day", "#2ca02c", "day-blocked folds", 0.12)]
fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), sharey=False)
for ax, a in zip(axes, [0.01, 0.02, 0.05]):
    for level, col, lab, dx in LEVEL:
        s = np.array([100 * f2[f"{i}|{level}|{a}"]["slip"] for i in range(1, 6)])
        ci = np.array([[100 * x for x in f2[f"{i}|{level}|{a}"]["ci"]] for i in range(1, 6)])
        ax.errorbar(np.arange(1, 6) + dx, s, yerr=[s - ci[:, 0], ci[:, 1] - s], marker="o", ms=3.5, capsize=2,
                    color=col, label=lab, lw=1.5)
    ax.axhline(a * 100, color="k", lw=0.8, ls=":")
    ax.set_title(f"Target slip {a:.0%}")
    ax.set_xticks(np.arange(1, 6))
    ax.set_xticklabels([f"{i}\n({f2[f'{i}|board|{a}']['defects']})" for i in range(1, 6)])
    ax.set_xlabel("Chronological evaluation slice (defects)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Realised slip rate (%)")
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, fontsize=7.5, loc="lower center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(FIG / "fig2_slices_grouping.png", dpi=300)
fig.savefig(FIG / "fig2_slices_grouping.pdf")

# ---------------- Fig. 3
LABEL = {"noctx": "GBDT, no context", "ctx": "GBDT + context", "two_stage": "Two-stage neighbour",
         "boardset_ctx": "BoardSet + context"}
COLOR = {"noctx": "#7f7f7f", "ctx": "#1f77b4", "two_stage": "#ff7f0e", "boardset_ctx": "#2ca02c"}
pm = dict(np.load(RES / "phme_meta.npz"))
sm = dict(np.load(RES / "siemens_meta.npz"))
h = int(sm["half"])
cases = [("phme", "official", pm["y"][pm["official_dev"]], pm["y"][pm["official_eval"]], "PHME Task 2, official test",
          {"noctx": "lgb_noctx", "ctx": "lgb_ctx", "two_stage": "two_stage", "boardset_ctx": "boardset_ctx"}),
         ("phme", "temporal", pm["y"][pm["temporal_dev"]], pm["y"][pm["temporal_eval"]], "PHME Task 2, July → September",
          {"noctx": "lgb_noctx", "ctx": "lgb_ctx", "two_stage": "two_stage", "boardset_ctx": "boardset_ctx"}),
         ("siemens", "eval", sm["y"][:h], sm["y"][h:], "Siemens AOI, second half",
          {"noctx": "lgb_base", "ctx": "lgb_ctx", "two_stage": "two_stage", "boardset_ctx": "boardset_ctx"})]
slips = np.linspace(0, 0.10, 41)
fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), sharey=True)
for ax, (ds, split, ydev, yev, title, models) in zip(axes, cases):
    for key, m in models.items():
        files = sorted(RES.glob(f"{ds}_{split}_{m}_seed*.npz"))
        dsl = [dict(np.load(f)) for f in files]
        s = np.mean([d["ev"].mean(0) for d in dsl], axis=0)
        v = [slip_and_volume(yev, s, threshold_for_slip(yev, s, a))[1] * 100 for a in slips]
        ax.plot(slips * 100, v, color=COLOR[key], label=LABEL[key], lw=1.5)
        for level, fill in [("board", "white"), ("day", COLOR[key])]:
            fl = files if level == "board" else sorted(RES.glob(f"{ds}_{split}_{m}_day_seed*.npz"))
            dl = dsl if level == "board" else [dict(np.load(f)) for f in fl]
            if not dl:
                continue
            for a, mk in [(0.02, "o"), (0.05, "s")]:
                pts = []
                for d in dl:
                    ps = passed("pooled", d["oof"], d["fold"], ydev, d["ev"], a)
                    pts.append([(ps & (yev == 1)).sum() / yev.sum() * 100, (ps & (yev == 0)).sum() / (yev == 0).sum() * 100])
                ax.scatter(*np.mean(pts, axis=0), marker=mk, s=26, facecolor=fill, edgecolor=COLOR[key], lw=1.3, zorder=4)
    ax.set_title(title)
    ax.set_xlabel("Slip rate (%)")
    ax.set_xlim(0, 10.5)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Inspection-volume reduction (%)")
h_, l_ = axes[2].get_legend_handles_labels()
for mk, lab in [("o", "target 2%"), ("s", "target 5%")]:
    h_.append(plt.scatter([], [], marker=mk, s=26, facecolor="white", edgecolor="k"))
    l_.append(f"calibrated, {lab}, board-wise folds")
h_.append(plt.scatter([], [], marker="o", s=26, facecolor="k", edgecolor="k"))
l_.append("filled: day-blocked folds")
fig.legend(h_, l_, fontsize=7.5, loc="lower center", ncol=4, frameon=False)
fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.savefig(FIG / "fig4_slip_volume.png", dpi=300)
fig.savefig(FIG / "fig4_slip_volume.pdf")
print("written", FIG)
