"""논문 그림 (2026-11-23). 저장된 점수와 분석 결과만 쓴다.

Fig. 2: 놓침-물량 곡선(평가 기간, 정답 기준 문턱값을 훑음) — PHME official / PHME temporal / Siemens
Fig. 3: Siemens 시간순 다섯 구간의 놓침: 행 단위 OOF 문턱값 vs 묶음 단위 OOF 문턱값 (lgb_base, 목표 1%)

사용: python -I 41_figures.py <결과 폴더(점수)> <그림 폴더>
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
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
LABEL = {"lgb_noctx": "LightGBM (no context)", "lgb_base": "LightGBM (no context)", "lgb_ctx": "LightGBM + context",
         "two_stage": "Two-stage neighbour", "boardset_ctx": "BoardSet", "boardset_noctx": "BoardSet (no hand-made context)"}
COLOR = {"lgb_noctx": "#888888", "lgb_base": "#888888", "lgb_ctx": "#1f77b4", "two_stage": "#ff7f0e",
         "boardset_ctx": "#2ca02c", "boardset_noctx": "#98df8a"}


def curve(y, s, slips):
    return [slip_and_volume(y, s, threshold_for_slip(y, s, a))[1] for a in slips]


def seed_mean_scores(pattern):
    runs = [dict(np.load(f)) for f in sorted(RES.glob(pattern))]
    if not runs:
        return None
    return np.mean([r["eval"] for r in runs], axis=0), runs


slips = np.linspace(0.0, 0.2, 41)
fig, axes = plt.subplots(1, 3, figsize=(10, 3.1), sharey=True)
panels = []
pm = dict(np.load(RES / "phme_meta.npz"))
for split, ax in zip(["official", "temporal"], axes[:2]):
    ev = pm[f"{split}_eval"]
    for m in ["lgb_noctx", "lgb_ctx", "two_stage", "boardset_ctx"]:
        r = seed_mean_scores(f"phme_{split}_{m}_seed*.npz")
        if r is not None:
            ax.plot(slips * 100, np.array(curve(pm["y"][ev], r[0], slips)) * 100, label=LABEL[m], color=COLOR[m])
    ax.set_title(f"PHME 2022 Task 2 ({'official test' if split == 'official' else 'July → September'})")
sm = dict(np.load(RES / "siemens_meta.npz"))
half = int(sm["half"])
for m in ["lgb_base", "lgb_ctx", "two_stage", "boardset_ctx"]:
    r = seed_mean_scores(f"siemens_{m}_seed*.npz")
    if r is not None:
        axes[2].plot(slips * 100, np.array(curve(sm["y"][half:], r[0], slips)) * 100, label=LABEL[m], color=COLOR[m])
axes[2].set_title("Siemens AOI (second half)")
for ax in axes:
    ax.set_xlabel("Slip rate (%)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Inspection-volume reduction (%)")
axes[2].legend(loc="lower right", fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "fig2_slip_volume.png", dpi=300)
fig.savefig(FIG / "fig2_slip_volume.pdf")

# Fig. 3
r = [dict(np.load(f)) for f in sorted(RES.glob("siemens_lgb_base_seed*.npz"))]
if r:
    y = sm["y"]
    ydev, yev = y[:half], y[half:]
    sl = np.array_split(np.arange(len(yev)), 5)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for key, lab, col in [("oof_leaky", "Threshold from row-wise CV", "#d62728"),
                          ("oof_grouped", "Threshold from group-wise CV", "#1f77b4")]:
        vals = []
        for run in r:
            t = threshold_for_slip(ydev, run[key], 0.01)
            vals.append([slip_and_volume(yev[i], run["eval"][i], t)[0] * 100 for i in sl])
        vals = np.array(vals)
        ax.errorbar(np.arange(1, 6), vals.mean(0), yerr=vals.std(0), marker="o", capsize=3, label=lab, color=col)
    ax.axhline(1.0, ls="--", color="k", lw=0.8)
    ax.set_xlabel("Chronological evaluation slice")
    ax.set_ylabel("Slip rate (%)")
    ax.set_title("Siemens, LightGBM, target slip 1%")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_leaky_vs_grouped_slip.png", dpi=300)
    fig.savefig(FIG / "fig3_leaky_vs_grouped_slip.pdf")
print("figures written to", FIG)
