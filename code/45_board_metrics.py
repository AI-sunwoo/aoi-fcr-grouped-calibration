"""보드 단위 운영 지표와 작업량 환산 (2026-11-23).

작업자는 호출(call)이 아니라 보드를 다룬다. 보정 규칙(p̄ ≤ α, 묶음 단위 교차적합)을 쓸 때
  - touch-free boards : 호출이 모두 자동 통과되어 확인대에 갈 필요가 없는 보드의 비율
  - board escape      : 불량이 있는 보드 가운데 불량 호출이 '모두' 자동 통과된 보드의 비율(보드가 통째로 빠져나감)
  - partial escape    : 불량이 있는 보드 가운데 불량 호출이 '하나라도' 자동 통과된 보드의 비율
  - per 10,000 calls  : 줄어든 확인 건수와 통과된 불량 건수
PHME 보드 = 패널+FigureID, Siemens 보드 = 타임스탬프 묶음.

보정 두 가지:
  call  : 호출 단위(본문 4.3절) — 보류된 불량 호출의 점수로 p값
  board : 보드 단위 — 보류된 '불량 보드'마다 그 보드 불량 호출 점수의 최댓값을 보정 표본으로 쓴다.
          보드가 통째로 빠져나가는 것은 그 최댓값까지 통과될 때뿐이므로, 목표가 보드 유출률이 된다.

사용: python -I 45_board_metrics.py <결과 폴더> <출력 json>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RES, OUT = Path(sys.argv[1]), Path(sys.argv[2])
ALPHAS = [0.01, 0.02, 0.05, 0.10]


def pv(s, c):
    return (1 + np.searchsorted(c, s, side="right")) / (len(c) + 1)


def p_ev(d, yd, board_dev=None):
    K = d["ev"].shape[0]
    if board_dev is None:
        cal = [np.sort(d["oof"][(d["fold"] == k) & (yd == 1)]) for k in range(K)]
    else:
        cal = []
        for k in range(K):
            m = (d["fold"] == k) & (yd == 1)
            cal.append(np.sort(pd.Series(d["oof"][m]).groupby(board_dev[m]).max().values))
    return np.mean([pv(d["ev"][k], c) for k, c in enumerate(cal)], axis=0)


def board_metrics(ye, board, passed):
    df = pd.DataFrame({"b": board, "y": ye, "p": passed})
    g = df.groupby("b")
    all_pass = g["p"].all()
    has_def = g["y"].any()
    d = df[df.y == 1].groupby("b")["p"]
    def_all_pass = d.all()
    def_any_pass = d.any()
    n = len(ye)
    return {"touch_free_boards": float(all_pass.mean()),
            "touch_free_clean_boards": float(all_pass[~has_def].mean()),
            "board_escape": float(def_all_pass.mean()),
            "partial_escape": float(def_any_pass.mean()),
            "n_boards": int(len(all_pass)), "n_defective_boards": int(has_def.sum()),
            "checks_saved_per_10k": float(((passed) & (ye == 0)).sum() / n * 1e4),
            "defects_passed_per_10k": float(((passed) & (ye == 1)).sum() / n * 1e4),
            "call_slip": float(((passed) & (ye == 1)).sum() / max(ye.sum(), 1))}


report = {}
for ds in ["phme", "siemens"]:
    meta = dict(np.load(RES / f"{ds}_meta.npz"))
    y = meta["y"]
    if ds == "phme":
        splits = {s: (meta[f"{s}_dev"], meta[f"{s}_eval"]) for s in ["official", "temporal"]}
        board = meta["board"]
        models = ["lgb_noctx", "lgb_ctx", "two_stage", "boardset_ctx"]
    else:
        h = int(meta["half"])
        splits = {"eval": (np.arange(h), np.arange(h, len(y)))}
        board = meta["groups"]
        models = ["lgb_base", "lgb_ctx", "two_stage", "boardset_noctx", "boardset_ctx"]
    report[ds] = {}
    for split, (dev, ev) in splits.items():
        yd, ye = y[dev], y[ev]
        R = {"calls_per_board_eval": float(len(ev) / len(np.unique(board[ev]))),
             "defects_per_10k": float(ye.sum() / len(ye) * 1e4)}
        for m in models:
            files = sorted(RES.glob(f"{ds}_{split}_{m}_seed*.npz"))
            if not files:
                continue
            for cal in ["call", "board"]:
                rows = {a: [] for a in ALPHAS}
                for f in files:
                    p = p_ev(dict(np.load(f)), yd, None if cal == "call" else board[dev])
                    for a in ALPHAS:
                        rows[a].append(board_metrics(ye, board[ev], p <= a))
                key = m if cal == "call" else f"{m}|boardcal"
                R[key] = {str(a): {k: float(np.mean([r[k] for r in rows[a]])) for k in rows[a][0]} for a in ALPHAS}
        R["n_defective_boards_dev"] = int(len(np.unique(board[dev][yd == 1])))
        report[ds][split] = R
        print("=====", ds, split, "calls/board %.1f" % R["calls_per_board_eval"], "defects/10k %.1f" % R["defects_per_10k"])
        print("defective boards in dev", R["n_defective_boards_dev"])
        for m in [x for x in R if isinstance(R[x], dict)]:
            print(m, " | ".join(
                f"a{a}: tf {100*R[m][str(a)]['touch_free_boards']:.1f}% esc {100*R[m][str(a)]['board_escape']:.1f}% "
                f"part {100*R[m][str(a)]['partial_escape']:.1f}% slip {100*R[m][str(a)]['call_slip']:.1f}% "
                f"saved/10k {R[m][str(a)]['checks_saved_per_10k']:.0f} passdef/10k {R[m][str(a)]['defects_passed_per_10k']:.1f}"
                for a in ALPHAS))
Path(OUT).write_text(json.dumps(report, indent=1), encoding="utf-8")
