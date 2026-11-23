"""저장된 점수로 모든 지표를 계산한다 (2026-11-23).

문턱값은 언제나 개발 기간 OOF(묶음 단위)에서 정한다. '누출' 행은 행 단위 OOF에서 정한 문턱값이다.
'최선값(oracle)'은 평가 기간 정답을 보고 고른 문턱값으로, 순위 능력만 보기 위한 참고값이다.

사용: python -I 40_analyze.py <결과 폴더(점수 npz)> <출력 폴더>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import slip_and_volume, threshold_for_f1, threshold_for_slip  # noqa: E402

RES = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
TARGETS = [0.01, 0.02, 0.05, 0.10]
rng = np.random.default_rng(12345)


def metrics(y_dev, oof, y_ev, sc, oof_leaky=None):
    r = {"eval_pr_auc": average_precision_score(y_ev, sc), "dev_pr_auc": average_precision_score(y_dev, oof)}
    t_f1 = threshold_for_f1(y_dev, oof)
    r["eval_f1"] = f1_score(y_ev, (sc >= t_f1).astype(int), zero_division=0)
    for t in TARGETS:
        th = threshold_for_slip(y_dev, oof, t)
        s, v = slip_and_volume(y_ev, sc, th)
        r[f"slip@{t}"], r[f"vol@{t}"] = s, v
        r[f"oracle_vol@{t}"] = slip_and_volume(y_ev, sc, threshold_for_slip(y_ev, sc, t))[1]
        if oof_leaky is not None:
            th_l = threshold_for_slip(y_dev, oof_leaky, t)
            r[f"leaky_slip@{t}"], r[f"leaky_vol@{t}"] = slip_and_volume(y_ev, sc, th_l)
            r[f"leaky_dev_pr_auc"] = average_precision_score(y_dev, oof_leaky)
    return r


def load(pattern):
    files = sorted(RES.glob(pattern))
    return [dict(np.load(f)) for f in files]


def summarize(rows):
    df = pd.DataFrame(rows)
    return {c: [float(df[c].mean()), float(df[c].std(ddof=0))] for c in df.columns}


def group_bootstrap_diff(y, a, b, groups, n_boot=1000, metric="ap"):
    """묶음 단위 붓스트랩으로 (a - b) 지표 차이의 95% 구간."""
    ug = np.unique(groups)
    idx_by_g = pd.Series(np.arange(len(y))).groupby(groups).apply(np.array).to_dict()
    diffs = []
    for _ in range(n_boot):
        gs = rng.choice(ug, size=len(ug), replace=True)
        ii = np.concatenate([idx_by_g[g] for g in gs])
        if y[ii].sum() == 0:
            continue
        if metric == "ap":
            diffs.append(average_precision_score(y[ii], a[ii]) - average_precision_score(y[ii], b[ii]))
        else:  # 정답 기준 놓침 5%에서의 물량 감소 차이
            va = slip_and_volume(y[ii], a[ii], threshold_for_slip(y[ii], a[ii], 0.05))[1]
            vb = slip_and_volume(y[ii], b[ii], threshold_for_slip(y[ii], b[ii], 0.05))[1]
            diffs.append(va - vb)
    d = np.array(diffs)
    return [float(d.mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


report = {}

# ---------------- PHME ----------------
meta_f = RES / "phme_meta.npz"
if meta_f.exists():
    meta = dict(np.load(meta_f))
    y = meta["y"]
    report["phme"] = {}
    for split in ["official", "temporal"]:
        dev, ev = meta[f"{split}_dev"], meta[f"{split}_eval"]
        res, ens = {}, {}
        for model in ["lgb_noctx", "lgb_ctx", "two_stage", "boardset_noctx", "boardset_ctx"]:
            runs = load(f"phme_{split}_{model}_seed*.npz")
            if not runs:
                continue
            rows = [metrics(y[dev], r["oof_grouped"], y[ev], r["eval"], r.get("oof_leaky")) for r in runs]
            res[model] = {"seeds": len(runs), "per_seed": summarize(rows)}
            e_oof = np.mean([r["oof_grouped"] for r in runs], axis=0)
            e_ev = np.mean([r["eval"] for r in runs], axis=0)
            res[model]["seed_ensemble"] = metrics(y[dev], e_oof, y[ev], e_ev)
            ens[model] = e_ev
        # 붓스트랩: 각 모델 대 lgb_ctx, 그리고 lgb_ctx 대 lgb_noctx (씨앗 평균 점수, 패널 단위)
        boots = {}
        pairs = [(m, "lgb_ctx") for m in ens if m != "lgb_ctx"] + [("lgb_ctx", "lgb_noctx")]
        for a, b in pairs:
            if a in ens and b in ens:
                boots[f"{a}-{b}"] = {"ap": group_bootstrap_diff(y[ev], ens[a], ens[b], meta["panel"][ev]),
                                     "oracle_vol@0.05": group_bootstrap_diff(y[ev], ens[a], ens[b],
                                                                            meta["panel"][ev], metric="vol")}
        report["phme"][split] = {"models": res, "bootstrap_vs": boots,
                                 "n_dev": [int(len(dev)), int(y[dev].sum())], "n_eval": [int(len(ev)), int(y[ev].sum())]}

# ---------------- Siemens ----------------
meta_f = RES / "siemens_meta.npz"
if meta_f.exists():
    meta = dict(np.load(meta_f))
    y, half = meta["y"], int(meta["half"])
    n = len(y)
    dev, ev = np.arange(half), np.arange(half, n)
    slices = np.array_split(np.arange(n - half), 5)
    rep, ens = {}, {}
    for model in ["lgb_base", "lgb_ctx", "two_stage", "boardset_noctx", "boardset_ctx"]:
        runs = load(f"siemens_{model}_seed*.npz")
        if not runs:
            continue
        rows, slice_rows = [], [[] for _ in slices]
        for r in runs:
            rows.append(metrics(y[dev], r["oof_grouped"], y[ev], r["eval"], r.get("oof_leaky")))
            for i, sl in enumerate(slices):
                yy, ss = y[ev][sl], r["eval"][sl]
                m = {"pr_auc": average_precision_score(yy, ss)}
                for t in TARGETS:
                    m[f"slip@{t}"], m[f"vol@{t}"] = slip_and_volume(yy, ss, threshold_for_slip(y[dev], r["oof_grouped"], t))
                    m[f"oracle_vol@{t}"] = slip_and_volume(yy, ss, threshold_for_slip(yy, ss, t))[1]
                    if "oof_leaky" in r:
                        m[f"leaky_slip@{t}"], m[f"leaky_vol@{t}"] = slip_and_volume(
                            yy, ss, threshold_for_slip(y[dev], r["oof_leaky"], t))
                slice_rows[i].append(m)
        rep[model] = {"seeds": len(runs), "per_seed": summarize(rows),
                      "slices": [summarize(s) | {"n": int(len(sl)), "n_bad": int(y[ev][sl].sum())}
                                 for s, sl in zip(slice_rows, slices)]}
        e_oof = np.mean([r["oof_grouped"] for r in runs], axis=0)
        e_ev = np.mean([r["eval"] for r in runs], axis=0)
        rep[model]["seed_ensemble"] = metrics(y[dev], e_oof, y[ev], e_ev)
        ens[model] = e_ev
    boots = {}
    pairs = [(m, "lgb_ctx") for m in ens if m != "lgb_ctx"] + [("lgb_ctx", "lgb_base")]
    for a, b in pairs:
        if a in ens and b in ens:
            boots[f"{a}-{b}"] = {"ap": group_bootstrap_diff(y[ev], ens[a], ens[b], meta["groups"][ev], 500),
                                 "oracle_vol@0.05": group_bootstrap_diff(y[ev], ens[a], ens[b], meta["groups"][ev],
                                                                        500, metric="vol")}
    report["siemens"] = {"models": rep, "bootstrap_vs": boots,
                         "n_dev": [int(len(dev)), int(y[dev].sum())], "n_eval": [int(len(ev)), int(y[ev].sum())]}

(OUT / "analysis.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


# 짧은 표 출력
def fmt(ms):
    return f"{ms[0]:.3f}±{ms[1]:.3f}"


for ds in report:
    parts = report[ds].items() if ds == "phme" else [("eval", report[ds])]
    for split, r in parts:
        print(f"===== {ds} {split}")
        for m, v in r["models"].items():
            p = v["per_seed"]
            print(f"{m:15s} n={v['seeds']} devAP {fmt(p['dev_pr_auc'])} evalAP {fmt(p['eval_pr_auc'])} "
                  f"F1 {fmt(p['eval_f1'])} | slip/vol@1% {p['slip@0.01'][0]:.3f}/{p['vol@0.01'][0]:.3f} "
                  f"@2% {p['slip@0.02'][0]:.3f}/{p['vol@0.02'][0]:.3f} @5% {p['slip@0.05'][0]:.3f}/{p['vol@0.05'][0]:.3f} "
                  f"| oracle vol@1/2/5% {p['oracle_vol@0.01'][0]:.3f}/{p['oracle_vol@0.02'][0]:.3f}/{p['oracle_vol@0.05'][0]:.3f}")
            if "leaky_slip@0.01" in p:
                print(f"{'':15s} leaky devAP {fmt(p['leaky_dev_pr_auc'])} leaky slip/vol@1% "
                      f"{p['leaky_slip@0.01'][0]:.3f}/{p['leaky_vol@0.01'][0]:.3f} @5% "
                      f"{p['leaky_slip@0.05'][0]:.3f}/{p['leaky_vol@0.05'][0]:.3f}")
        for k, b in r["bootstrap_vs"].items():
            print("  boot", k, {kk: [round(x, 4) for x in vv] for kk, vv in b.items()})
