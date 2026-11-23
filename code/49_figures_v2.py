"""논문 그림 2판 (2026-11-23). 보정 규칙 = 합친 순위 교차 conformal(48번과 같음).

Fig. 2: Siemens 평가 기간 다섯 구간의 실제 놓침 — 행 / 보드 / 날짜 단위 교차검증 보정, GBDT(맥락 없음), 목표 1%·2%
        x축 눈금에 구간별 불량 수
Fig. 3: 놓침-물량 곡선(0~10%, 다섯 겹 확률 평균의 씨앗 평균, 평가 정답으로 문턱값을 훑음) + 보정 규칙의 실제 점(α=2%, 5%)

사용: python -I 49_figures_v2.py <cf 결과 폴더> <그림 폴더>
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


def passed_pooled(d, yd, a, keys=("oof", "fold", "ev")):
    oof, fold, ev = d[keys[0]], d[keys[1]], d[keys[2]]
    R, n = np.zeros(ev.shape[1]), 0
    for k in range(ev.shape[0]):
        c = np.sort(oof[(fold == k) & (yd == 1)])
        R += np.searchsorted(c, ev[k], side="right")
        n += len(c)
    return (1 + R) / (n + 1) <= a


# ---------------- Fig. 2
sm = dict(np.load(RES / "siemens_meta.npz"))
h = int(sm["half"])
yd, ye = sm["y"][:h], sm["y"][h:]
slices = np.array_split(np.arange(len(ye)), 5)
LEVEL = [("row", "#d62728", "row-wise folds", "lgb_base_seed*", ("oof_l", "fold_l", "ev_l")),
         ("board", "#1f77b4", "board-wise folds", "lgb_base_seed*", ("oof", "fold", "ev")),
         ("day", "#2ca02c", "day-blocked folds", "lgb_base_day_seed*", ("oof", "fold", "ev"))]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
for ax, alpha in zip(axes, [0.01, 0.02]):
    for level, col, lab, pat, keys in LEVEL:
        files = sorted(RES.glob(f"siemens_eval_{pat}.npz"))
        vals = []
        for f in files:
            ps = passed_pooled(dict(np.load(f)), yd, alpha, keys)
            vals.append([((ps[s]) & (ye[s] == 1)).sum() / ye[s].sum() * 100 for s in slices])
        vals = np.array(vals)
        ax.errorbar(np.arange(1, 6), vals.mean(0), yerr=vals.std(0, ddof=1), marker="o", ms=3.5, capsize=2,
                    color=col, label=lab, lw=1.6)
    ax.axhline(alpha * 100, color="k", lw=0.8, ls=":")
    ax.set_title(f"Target slip {alpha:.0%}")
    ax.set_xticks(np.arange(1, 6))
    ax.set_xticklabels([f"{i + 1}\n({int(ye[s].sum())})" for i, s in enumerate(slices)])
    ax.set_xlabel("Chronological evaluation slice (defects)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Realised slip rate (%)")
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, fontsize=7.5, loc="lower center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(FIG / "fig2_slices_grouping.png", dpi=300)
fig.savefig(FIG / "fig2_slices_grouping.pdf")

# ---------------- Fig. 3
LABEL = {"lgb_noctx": "GBDT, no context", "lgb_base": "GBDT, no context", "lgb_ctx": "GBDT + context",
         "two_stage": "Two-stage neighbour", "boardset_ctx": "BoardSet + context"}
COLOR = {"lgb_noctx": "#7f7f7f", "lgb_base": "#7f7f7f", "lgb_ctx": "#1f77b4", "two_stage": "#ff7f0e",
         "boardset_ctx": "#2ca02c"}
slips = np.linspace(0, 0.10, 41)
pm = dict(np.load(RES / "phme_meta.npz"))
cases = [("phme", "official", pm["y"][pm["official_dev"]], pm["y"][pm["official_eval"]], "PHME Task 2, official test",
          ["lgb_noctx", "lgb_ctx", "two_stage", "boardset_ctx"]),
         ("phme", "temporal", pm["y"][pm["temporal_dev"]], pm["y"][pm["temporal_eval"]], "PHME Task 2, July → September",
          ["lgb_noctx", "lgb_ctx", "two_stage", "boardset_ctx"]),
         ("siemens", "eval", yd, ye, "Siemens AOI, second half", ["lgb_base", "lgb_ctx", "two_stage", "boardset_ctx"])]
fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True)
for ax, (ds, split, ydev, yev, title, models) in zip(axes, cases):
    for m in models:
        files = sorted(RES.glob(f"{ds}_{split}_{m}_seed*.npz"))
        if not files:
            continue
        ds_ = [dict(np.load(f)) for f in files]
        s = np.mean([d["ev"].mean(0) for d in ds_], axis=0)
        v = [slip_and_volume(yev, s, threshold_for_slip(yev, s, a))[1] * 100 for a in slips]
        ax.plot(slips * 100, v, color=COLOR[m], label=LABEL[m], lw=1.6)
        for a, mk in [(0.02, "o"), (0.05, "s")]:
            pts = []
            for d in ds_:
                ps = passed_pooled(d, ydev, a)
                pts.append([(ps & (yev == 1)).sum() / yev.sum() * 100, (ps & (yev == 0)).sum() / (yev == 0).sum() * 100])
            pts = np.mean(pts, axis=0)
            ax.scatter(*pts, marker=mk, s=28, facecolor="white", edgecolor=COLOR[m], lw=1.4, zorder=4)
    ax.set_title(title)
    ax.set_xlabel("Slip rate (%)")
    ax.set_xlim(0, 10.5)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Inspection-volume reduction (%)")
h_, l_ = axes[2].get_legend_handles_labels()
h_ += [plt.scatter([], [], marker="o", s=28, facecolor="white", edgecolor="k"),
       plt.scatter([], [], marker="s", s=28, facecolor="white", edgecolor="k")]
l_ += ["calibrated, target 2%", "calibrated, target 5%"]
fig.legend(h_, l_, fontsize=7.5, loc="lower center", ncol=6, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(FIG / "fig3_slip_volume.png", dpi=300)
fig.savefig(FIG / "fig3_slip_volume.pdf")
print("written", FIG)
