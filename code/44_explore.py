"""교차적합 결과 추가 점검 (2026-11-23).

1) F1 문턱값 방식 비교: (a) p값 평균 + 개발 p값으로 고른 문턱값, (b) 겹 모델 확률 평균 + OOF 확률로 고른 문턱값
2) 씨앗 앙상블(씨앗 평균)의 AP·F1
3) 겹 단위 앙상블: 두 모델의 겹별 p값을 평균한 점수로 다시 conformal 보정 (two_stage + boardset_ctx)
4) 겹 배정이 모델끼리 같은지 확인

사용: python -I 44_explore.py <결과 폴더> <출력 json>
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import slip_and_volume, threshold_for_f1, threshold_for_slip  # noqa: E402

RES, OUT = Path(sys.argv[1]), Path(sys.argv[2])
ALPHAS = [0.01, 0.02, 0.05, 0.10]


def pv(s, c):
    return (1 + np.searchsorted(c, s, side="right")) / (len(c) + 1)


def fold_p(d, y_dev):
    """겹별 p값: dev(OOF, 자기 겹 보정) 와 eval(K x n)."""
    K = d["ev"].shape[0]
    cal = [np.sort(d["oof"][(d["fold"] == k) & (y_dev == 1)]) for k in range(K)]
    p_dev = np.zeros(len(d["oof"]))
    for k in range(K):
        m = d["fold"] == k
        p_dev[m] = pv(d["oof"][m], cal[k])
    p_ev = np.stack([pv(d["ev"][k], cal[k]) for k in range(K)])
    return p_dev, p_ev


def metrics(y_dev, y_ev, s_dev, s_ev):
    t = threshold_for_f1(y_dev, s_dev)
    r = {"ap": average_precision_score(y_ev, s_ev), "f1": f1_score(y_ev, (s_ev >= t).astype(int), zero_division=0)}
    for a in ALPHAS:
        r[f"ovol@{a}"] = slip_and_volume(y_ev, s_ev, threshold_for_slip(y_ev, s_ev, a))[1]
    return r


def calib(y_dev, fold, oof_score, ev_scores):
    """겹 단위 점수(oof_score, ev_scores K x n)를 conformal p값으로."""
    d = {"oof": oof_score, "fold": fold, "ev": ev_scores}
    p_dev, p_ev = fold_p(d, y_dev)
    return p_dev, p_ev.mean(0)


def realized(y_ev, p_ev):
    return {f"slip@{a}": float(((p_ev <= a) & (y_ev == 1)).sum() / y_ev.sum()) for a in ALPHAS} | \
           {f"vol@{a}": float(((p_ev <= a) & (y_ev == 0)).sum() / (y_ev == 0).sum()) for a in ALPHAS}


report = {}
for ds in ["phme", "siemens"]:
    meta = dict(np.load(RES / f"{ds}_meta.npz"))
    y = meta["y"]
    if ds == "phme":
        splits = {s: (meta[f"{s}_dev"], meta[f"{s}_eval"]) for s in ["official", "temporal"]}
        models = ["lgb_noctx", "lgb_ctx", "two_stage", "boardset_noctx", "boardset_ctx"]
    else:
        h = int(meta["half"])
        splits = {"eval": (np.arange(h), np.arange(h, len(y)))}
        models = ["lgb_base", "lgb_ctx", "two_stage", "boardset_noctx", "boardset_ctx"]
    report[ds] = {}
    for split, (dev, ev) in splits.items():
        yd, ye = y[dev], y[ev]
        R = {}
        D = {m: [dict(np.load(f)) for f in sorted(RES.glob(f"{ds}_{split}_{m}_seed*.npz"))] for m in models}
        n_seeds = min(len(v) for v in D.values())
        same = all(np.array_equal(D[models[0]][s]["fold"], D[m][s]["fold"]) for m in models for s in range(n_seeds))
        R["folds_identical"] = bool(same)
        for m in models:
            a_rows, b_rows = [], []
            pe_all, raw_all, pd_all, rawd_all = [], [], [], []
            for d in D[m]:
                p_dev, p_ev = fold_p(d, yd)
                a_rows.append(metrics(yd, ye, p_dev, p_ev.mean(0)))
                b_rows.append(metrics(yd, ye, d["oof"], d["ev"].mean(0)))
                pe_all.append(p_ev.mean(0)); pd_all.append(p_dev)
                raw_all.append(d["ev"].mean(0)); rawd_all.append(d["oof"])
            R[m] = {"pval_thr": {k: [float(np.mean([r[k] for r in a_rows])), float(np.std([r[k] for r in a_rows]))]
                                 for k in a_rows[0]},
                    "raw_thr": {k: [float(np.mean([r[k] for r in b_rows])), float(np.std([r[k] for r in b_rows]))]
                                for k in b_rows[0]},
                    "seed_ens_pval": metrics(yd, ye, np.mean(pd_all, 0), np.mean(pe_all, 0)),
                    "seed_ens_raw": metrics(yd, ye, np.mean(rawd_all, 0), np.mean(raw_all, 0))}
        # 겹 단위 앙상블
        for combo in [("two_stage", "boardset_ctx"), ("lgb_ctx", "boardset_ctx"), ("two_stage", "boardset_noctx")]:
            rows, rl = [], []
            pe_all = []
            for s in range(n_seeds):
                parts = [fold_p(D[m][s], yd) for m in combo]
                fold = D[combo[0]][s]["fold"]
                oof_e = np.mean([p[0] for p in parts], 0)
                ev_e = np.mean([p[1] for p in parts], 0)
                p_dev, p_ev = calib(yd, fold, oof_e, ev_e)
                rows.append(metrics(yd, ye, p_dev, p_ev))
                rl.append(realized(ye, p_ev))
                pe_all.append(p_ev)
            key = "+".join(combo)
            R[key] = {"pval_thr": {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))]
                                   for k in rows[0]},
                      "realized": {k: float(np.mean([r[k] for r in rl])) for k in rl[0]},
                      "seed_ens_ap": float(average_precision_score(ye, np.mean(pe_all, 0)))}
        report[ds][split] = R
        print("=====", ds, split, "folds identical:", same)
        for m, v in R.items():
            if m == "folds_identical":
                continue
            pt = v["pval_thr"]
            line = f"{m:28s} AP {pt['ap'][0]:.3f} F1(p) {pt['f1'][0]:.3f}±{pt['f1'][1]:.3f}"
            if "raw_thr" in v:
                rt = v["raw_thr"]
                line += (f" F1(raw) {rt['f1'][0]:.3f}±{rt['f1'][1]:.3f} AP(raw) {rt['ap'][0]:.3f}"
                         f" | seedens AP {v['seed_ens_pval']['ap']:.3f} F1(p) {v['seed_ens_pval']['f1']:.3f}"
                         f" F1(raw) {v['seed_ens_raw']['f1']:.3f}")
            else:
                line += f" | seedens AP {v['seed_ens_ap']:.3f} " + \
                        " ".join(f"a{a}:{v['realized'][f'slip@{a}']:.3f}/{v['realized'][f'vol@{a}']:.3f}"
                                 for a in ALPHAS)
            line += " ovol " + "/".join(f"{pt[f'ovol@{a}'][0]:.3f}" for a in [0.02, 0.05, 0.1])
            print(line)
Path(OUT).write_text(json.dumps(report, indent=1), encoding="utf-8")
