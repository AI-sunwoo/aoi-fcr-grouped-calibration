"""논문 그림 (교차적합 결과, 2026-11-23).

Fig. 1: Siemens 시간순 다섯 구간의 실제 놓침 — 행 단위 CV 보정 vs 묶음 단위 CV 보정(목표 1%, 2%)
Fig. 2: 놓침-물량 곡선(평가 정답으로 문턱값을 훑음, 씨앗 평균 p값) — PHME official / PHME temporal / Siemens

사용: python -I 43_figures_crossfit.py <결과 폴더> <그림 폴더>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import slip_and_volume, threshold_for_slip  # noqa: E402

RES, FIG = Path(sys.argv[1]), Path(sys.argv[2])
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans"})
LABEL = {"lgb_noctx": "GBDT, no context", "lgb_base": "GBDT, no context", "lgb_ctx": "GBDT + context",
         "two_stage": "Two-stage neighbour", "boardset_ctx": "BoardSet", "boardset_noctx": "BoardSet, no hand-made context"}
COLOR = {"lgb_noctx": "#7f7f7f", "lgb_base": "#7f7f7f", "lgb_ctx": "#1f77b4", "two_stage": "#ff7f0e",
         "boardset_ctx": "#2ca02c", "boardset_noctx": "#98df8a"}


def pvalues(oof, fold, y_dev, ev):
    cal = [np.sort(oof[(fold == k) & (y_dev == 1)]) for k in range(ev.shape[0])]
    return np.mean([(1 + np.searchsorted(c, ev[k], side="right")) / (len(c) + 1) for k, c in enumerate(cal)], axis=0)


def seed_avg_p(ds, split, model, y_dev, leaky=False):
    ps = []
    for f in sorted(RES.glob(f"{ds}_{split}_{model}_seed*.npz")):
        d = dict(np.load(f))
        if leaky:
            ps.append(pvalues(d["oof_l"], d["fold_l"], y_dev, d["ev_l"]))
        else:
            ps.append(pvalues(d["oof"], d["fold"], y_dev, d["ev"]))
    return ps


# Fig. 1
sm = dict(np.load(RES / "siemens_meta.npz"))
h = int(sm["half"])
yd, ye = sm["y"][:h], sm["y"][h:]
slices = np.array_split(np.arange(len(ye)), 5)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
for ax, alpha in zip(axes, [0.01, 0.02]):
    for model, ls in [("lgb_base", "-"), ("lgb_ctx", "--")]:
        for leaky, col, lab in [(True, "#d62728", "row-wise CV"), (False, "#1f77b4", "group-wise CV")]:
            ps = seed_avg_p("siemens", "eval", model, yd, leaky)
            if not ps:
                continue
            vals = np.array([[((p[s] <= alpha) & (ye[s] == 1)).sum() / max(ye[s].sum(), 1) * 100 for s in slices]
                             for p in ps])
            ax.errorbar(np.arange(1, 6), vals.mean(0), yerr=vals.std(0), ls=ls, marker="o", ms=3, capsize=2,
                        color=col, label=f"{LABEL[model]}, {lab}")
    ax.axhline(alpha * 100, color="k", lw=0.8, ls=":")
    ax.set_title(f"Target slip {alpha:.0%}")
    ax.set_xlabel("Chronological evaluation slice")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Realised slip rate (%)")
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, fontsize=7, loc="lower center", ncol=2, frameon=False)
fig.tight_layout(rect=(0, 0.17, 1, 1))
fig.savefig(FIG / "fig1_rowwise_vs_groupwise_slip.png", dpi=300)
fig.savefig(FIG / "fig1_rowwise_vs_groupwise_slip.pdf")

# Fig. 2
slips = np.linspace(0, 0.2, 41)
fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), sharey=True)
pm = dict(np.load(RES / "phme_meta.npz"))
cases = [("phme", "official", pm["y"][pm["official_dev"]], pm["y"][pm["official_eval"]],
          "PHME Task 2, official test", ["lgb_noctx", "lgb_ctx", "two_stage", "boardset_ctx"]),
         ("phme", "temporal", pm["y"][pm["temporal_dev"]], pm["y"][pm["temporal_eval"]],
          "PHME Task 2, July → September", ["lgb_noctx", "lgb_ctx", "two_stage", "boardset_ctx"]),
         ("siemens", "eval", yd, ye, "Siemens AOI, second half", ["lgb_base", "lgb_ctx", "two_stage", "boardset_ctx"])]
for ax, (ds, split, ydev, yev, title, models) in zip(axes, cases):
    for m in models:
        ps = seed_avg_p(ds, split, m, ydev)
        if not ps:
            continue
        p = np.mean(ps, axis=0)
        v = [slip_and_volume(yev, p, threshold_for_slip(yev, p, a))[1] * 100 for a in slips]
        ax.plot(slips * 100, v, color=COLOR[m], label=LABEL[m])
    ax.set_title(title)
    ax.set_xlabel("Slip rate (%)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Inspection-volume reduction (%)")
axes[2].legend(fontsize=6.5, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "fig2_slip_volume.png", dpi=300)
fig.savefig(FIG / "fig2_slip_volume.pdf")
print("written", FIG)
