"""교차적합 결과 분석 (2026-11-23). 26번(LightGBM 계열)과 22번(BoardSet) 출력을 함께 읽는다.

보정: 겹 k 모델의 OOF 불량 점수 C_k로 불량 기준 conformal p값 p_k(s) = (1 + #{c in C_k : c <= s}) / (|C_k| + 1).
평가 항목의 p = 다섯 겹 p_k의 평균. 자동 통과 = p <= alpha (불량 하나가 통과할 확률이 대략 alpha 이하).
F1은 겹 모델 확률의 평균 점수에 개발 기간 OOF 확률로 고른 문턱값을 쓴다(p값은 상단이 1로 몰려 F1 문턱값이 흔들림).
'정답 기준(oracle)'은 평가 정답으로 고른 문턱값으로 순위 능력만 본다.

사용: python -I 42_analyze_crossfit.py <결과 폴더> <출력 폴더>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import slip_and_volume, threshold_for_f1, threshold_for_slip  # noqa: E402

RES, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
ALPHAS = [0.01, 0.02, 0.05, 0.10]
rng = np.random.default_rng(2026)


def pvalues(oof, fold, y_dev, ev):
    K = ev.shape[0]
    cal = [np.sort(oof[(fold == k) & (y_dev == 1)]) for k in range(K)]

    def pv(s, c):
        return (1 + np.searchsorted(c, s, side="right")) / (len(c) + 1)
    p_dev = np.zeros(len(oof))
    for k in range(K):
        m = fold == k
        p_dev[m] = pv(oof[m], cal[k])
    p_ev = np.mean([pv(ev[k], cal[k]) for k in range(K)], axis=0)
    return p_dev, p_ev


def evaluate(y_dev, y_ev, p_dev, p_ev, raw_dev, raw_ev):
    r = {"eval_ap": average_precision_score(y_ev, p_ev)}
    t = threshold_for_f1(y_dev, raw_dev)
    r["eval_f1"] = f1_score(y_ev, (raw_ev >= t).astype(int), zero_division=0)
    for a in ALPHAS:
        passed = p_ev <= a
        r[f"slip@{a}"] = float((passed & (y_ev == 1)).sum() / max(y_ev.sum(), 1))
        r[f"vol@{a}"] = float((passed & (y_ev == 0)).sum() / max((y_ev == 0).sum(), 1))
        r[f"oracle_vol@{a}"] = slip_and_volume(y_ev, p_ev, threshold_for_slip(y_ev, p_ev, a))[1]
    return r


def boot(y, a, b, groups, n=1000, kind="ap"):
    ug = np.unique(groups)
    idx = pd.Series(np.arange(len(y))).groupby(groups).apply(np.array).to_dict()
    out = []
    for _ in range(n):
        ii = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        if y[ii].sum() == 0:
            continue
        if kind == "ap":
            out.append(average_precision_score(y[ii], a[ii]) - average_precision_score(y[ii], b[ii]))
        else:
            fa = slip_and_volume(y[ii], a[ii], threshold_for_slip(y[ii], a[ii], 0.05))[1]
            fb = slip_and_volume(y[ii], b[ii], threshold_for_slip(y[ii], b[ii], 0.05))[1]
            out.append(fa - fb)
    d = np.array(out)
    return [float(d.mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


def summarize(rows):
    df = pd.DataFrame(rows)
    return {c: [float(df[c].mean()), float(df[c].std(ddof=0))] for c in df.columns}


MODELS = ["lgb_noctx", "lgb_base", "lgb_ctx", "two_stage", "boardset_noctx", "boardset_ctx"]
report = {}
for ds in ["phme", "siemens"]:
    mf = RES / f"{ds}_meta.npz"
    if not mf.exists():
        continue
    meta = dict(np.load(mf))
    y = meta["y"]
    if ds == "phme":
        splits = {s: (meta[f"{s}_dev"], meta[f"{s}_eval"]) for s in ["official", "temporal"]}
        grp = meta["panel"]
    else:
        h = int(meta["half"])
        splits = {"eval": (np.arange(h), np.arange(h, len(y)))}
        grp = meta["groups"]
    report[ds] = {}
    for split, (dev, ev) in splits.items():
        yd, ye = y[dev], y[ev]
        res, ens = {}, {}
        for m in MODELS:
            files = sorted(RES.glob(f"{ds}_{split}_{m}_seed*.npz"))
            if not files:
                continue
            rows, rows_l, pevs, slices, raws, rawd = [], [], [], [], [], []
            for f in files:
                d = dict(np.load(f))
                p_dev, p_ev = pvalues(d["oof"], d["fold"], yd, d["ev"])
                r = evaluate(yd, ye, p_dev, p_ev, d["oof"], d["ev"].mean(0))
                r["dev_ap"] = average_precision_score(yd, d["oof"])
                rows.append(r)
                pevs.append(p_ev)
                raws.append(d["ev"].mean(0))
                rawd.append(d["oof"])
                if "oof_l" in d:
                    pl_dev, pl_ev = pvalues(d["oof_l"], d["fold_l"], yd, d["ev_l"])
                    rl = evaluate(yd, ye, pl_dev, pl_ev, d["oof_l"], d["ev_l"].mean(0))
                    rl["dev_ap"] = average_precision_score(yd, d["oof_l"])
                    rows_l.append(rl)
                if ds == "siemens":
                    sl_rows = []
                    for sl in np.array_split(np.arange(len(ev)), 5):
                        e = {"n_bad": int(ye[sl].sum())}
                        for a in [0.01, 0.02, 0.05]:
                            ps = p_ev[sl] <= a
                            e[f"slip@{a}"] = float((ps & (ye[sl] == 1)).sum() / max(ye[sl].sum(), 1))
                            e[f"vol@{a}"] = float((ps & (ye[sl] == 0)).sum() / max((ye[sl] == 0).sum(), 1))
                            if "oof_l" in d:
                                psl = pl_ev[sl] <= a
                                e[f"leaky_slip@{a}"] = float((psl & (ye[sl] == 1)).sum() / max(ye[sl].sum(), 1))
                                e[f"leaky_vol@{a}"] = float((psl & (ye[sl] == 0)).sum() / max((ye[sl] == 0).sum(), 1))
                        sl_rows.append(e)
                    slices.append(sl_rows)
            res[m] = {"seeds": len(files), "grouped": summarize(rows)}
            if rows_l:
                res[m]["rowwise"] = summarize(rows_l)
            if slices:
                res[m]["slices"] = [summarize([s[i] for s in slices]) for i in range(5)]
            pe = np.mean(pevs, axis=0)
            ens[m] = pe
            res[m]["seed_avg_eval_ap"] = float(average_precision_score(ye, pe))
            t_ens = threshold_for_f1(yd, np.mean(rawd, axis=0))
            res[m]["seed_avg_eval_f1"] = float(f1_score(ye, (np.mean(raws, axis=0) >= t_ens).astype(int)))
            res[m]["seed_avg_oracle_vol"] = {str(a): slip_and_volume(ye, pe, threshold_for_slip(ye, pe, a))[1]
                                             for a in ALPHAS}
        ref = "lgb_ctx"
        bs = {}
        for m in ens:
            if m != ref and ref in ens:
                bs[f"{m} - {ref}"] = {"ap": boot(ye, ens[m], ens[ref], grp[ev], 500 if ds == "siemens" else 1000),
                                      "oracle_vol@0.05": boot(ye, ens[m], ens[ref], grp[ev],
                                                              500 if ds == "siemens" else 1000, kind="vol")}
        report[ds][split] = {"models": res, "bootstrap": bs,
                             "n_dev": [int(len(dev)), int(yd.sum())], "n_eval": [int(len(ev)), int(ye.sum())]}

(OUT / "analysis_crossfit.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


def f(v):
    return f"{v[0]:.3f}±{v[1]:.3f}"


for ds, sp in report.items():
    for split, r in sp.items():
        print(f"===== {ds} {split}  dev {r['n_dev']} eval {r['n_eval']}")
        for m, v in r["models"].items():
            g = v["grouped"]
            print(f"{m:15s} n={v['seeds']} devAP {f(g['dev_ap'])} evalAP {f(g['eval_ap'])} F1 {f(g['eval_f1'])} "
                  + " ".join(f"a{a}: {g[f'slip@{a}'][0]:.3f}/{g[f'vol@{a}'][0]:.3f}({g[f'oracle_vol@{a}'][0]:.3f})"
                             for a in ALPHAS))
            if "rowwise" in v:
                l = v["rowwise"]
                print(f"{'  row-wise CV':15s}      devAP {f(l['dev_ap'])} "
                      + " ".join(f"a{a}: {l[f'slip@{a}'][0]:.3f}/{l[f'vol@{a}'][0]:.3f}" for a in ALPHAS))
        for k, b in r["bootstrap"].items():
            print("  boot", k, {kk: [round(x, 4) for x in vv] for kk, vv in b.items()})
