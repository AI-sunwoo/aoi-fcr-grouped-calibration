"""묶음 단위별 보정 비교: 행 / 보드(타임스탬프·패널) / 날짜 (2026-11-23).

26번(행·보드)과 27번(날짜) 출력에서 GBDT의 실제 놓침·확인 물량을 목표 α별로 나란히 본다.
Siemens는 평가 기간 다섯 구간도 함께 본다.

사용: python -I 46_grouping_levels.py <결과 폴더> <출력 json>
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

RES, OUT = Path(sys.argv[1]), Path(sys.argv[2])
ALPHAS = [0.01, 0.02, 0.05, 0.10]


def pv(s, c):
    return (1 + np.searchsorted(c, s, side="right")) / (len(c) + 1)


def p_ev(oof, fold, ev, yd):
    cal = [np.sort(oof[(fold == k) & (yd == 1)]) for k in range(ev.shape[0])]
    return np.mean([pv(ev[k], c) for k, c in enumerate(cal)], axis=0)


def sv(ye, p, a):
    ps = p <= a
    return float((ps & (ye == 1)).sum() / max(ye.sum(), 1)), float((ps & (ye == 0)).sum() / max((ye == 0).sum(), 1))


report = {}
for ds in ["siemens", "phme"]:
    meta = dict(np.load(RES / f"{ds}_meta.npz"))
    y = meta["y"]
    if ds == "phme":
        splits = {s: (meta[f"{s}_dev"], meta[f"{s}_eval"]) for s in ["official", "temporal"]}
        models = ["lgb_noctx", "lgb_ctx"]
    else:
        h = int(meta["half"])
        splits = {"eval": (np.arange(h), np.arange(h, len(y)))}
        models = ["lgb_base", "lgb_ctx"]
    report[ds] = {}
    for split, (dev, ev) in splits.items():
        yd, ye = y[dev], y[ev]
        R = {}
        for m in models:
            R[m] = {}
            for level in ["row", "board", "day"]:
                if level == "day":
                    files = sorted(RES.glob(f"{ds}_{split}_{m}_day_seed*.npz"))
                else:
                    files = sorted(RES.glob(f"{ds}_{split}_{m}_seed*.npz"))
                if not files:
                    continue
                rows, slices, devap = [], [], []
                for f in files:
                    d = dict(np.load(f))
                    k = ("oof_l", "fold_l", "ev_l") if level == "row" else ("oof", "fold", "ev")
                    p = p_ev(d[k[0]], d[k[1]], d[k[2]], yd)
                    devap.append(average_precision_score(yd, d[k[0]]))
                    rows.append({a: sv(ye, p, a) for a in ALPHAS})
                    if ds == "siemens":
                        slices.append([[sv(ye[s], p[s], a)[0] for a in [0.01, 0.02, 0.05]]
                                       for s in np.array_split(np.arange(len(ev)), 5)])
                R[m][level] = {"n_seeds": len(files), "dev_ap": float(np.mean(devap)),
                               **{f"slip@{a}": float(np.mean([r[a][0] for r in rows])) for a in ALPHAS},
                               **{f"vol@{a}": float(np.mean([r[a][1] for r in rows])) for a in ALPHAS}}
                if slices:
                    R[m][level]["slice_slip@1,2,5"] = np.mean(slices, axis=0).round(4).tolist()
        report[ds][split] = R
        print("=====", ds, split)
        for m, v in R.items():
            for level, r in v.items():
                print(f"{m:10s} {level:6s} n={r['n_seeds']} devAP {r['dev_ap']:.3f} "
                      + " ".join(f"a{a}: {100*r[f'slip@{a}']:.1f}/{100*r[f'vol@{a}']:.1f}" for a in ALPHAS)
                      + (f" slices@1%: {[round(100*x[0], 1) for x in r['slice_slip@1,2,5']]}"
                         if "slice_slip@1,2,5" in r else ""))
Path(OUT).write_text(json.dumps(report, indent=1), encoding="utf-8")
