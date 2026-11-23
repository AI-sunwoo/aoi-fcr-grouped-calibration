"""분석 2판 (2026-11-23, 심사 의견 반영). 42·44·45·46번을 대체한다.

바뀐 점
  - 보정 규칙 세 가지
      pooled : Vovk(2015) 교차 conformal. 겹마다 보류 불량 점수 C_k, 평가 호출 z의 합친 순위
               R(z) = Σ_k #{c ∈ C_k : c ≤ f_k(z)}, p = (1 + R) / (n + 1), n = Σ|C_k|. p ≤ α이면 자동 통과.  (본문 규칙)
      mean   : 겹별 p값의 평균(1판 규칙, aggregated conformal) — 비교용
      pac    : 높은 확신 규칙. 놓침이 α를 넘을 확률을 δ = 0.05 이하로: R ≤ r*,
               r* = max{r : P(Bin(n, α) ≤ r) ≤ δ} (Tong 외 2018의 NP umbrella와 같은 꼴). r*가 없으면 아무것도 통과 안 함.
  - 순위 지표(AP, 정답 기준 물량)는 다섯 겹 모델 확률의 평균(원점수)으로 잰다.
  - 모델 간 차이: 씨앗 평균 원점수로 구한 '관측 차이'와 묶음 부트스트랩 구간(백분위·basic 둘 다).
    묶음: PHME 패널, Siemens 날짜(시간 의존을 고려해 날짜 단위로 크게 묶음).
  - 보드 지표: 보드 유출(불량 호출이 모두 통과된 불량 보드의 비율), 손 안 대는 보드 비율, 1만 호출당 절감·통과 불량.

사용: python -I 48_analysis_v2.py <cf 결과 폴더> <controls 결과 폴더> <출력 json>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.metrics import average_precision_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import slip_and_volume, threshold_for_f1, threshold_for_slip  # noqa: E402

RES, CTRL, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ALPHAS = [0.01, 0.02, 0.05, 0.10]
DELTA = 0.05
rng = np.random.default_rng(2026)


# ---------------------------------------------------------------- 보정
def pooled_rank(oof, fold, yd, ev):
    """평가 호출마다 R(z)와 n을 돌려준다."""
    K = ev.shape[0]
    R = np.zeros(ev.shape[1])
    n = 0
    for k in range(K):
        c = np.sort(oof[(fold == k) & (yd == 1)])
        R += np.searchsorted(c, ev[k], side="right")
        n += len(c)
    return R, n


def mean_p(oof, fold, yd, ev):
    K = ev.shape[0]
    ps = []
    for k in range(K):
        c = np.sort(oof[(fold == k) & (yd == 1)])
        ps.append((1 + np.searchsorted(c, ev[k], side="right")) / (len(c) + 1))
    return np.mean(ps, axis=0)


def pac_r(n, a, delta=DELTA):
    r = np.arange(n + 1)
    ok = r[binom.cdf(r, n, a) <= delta]
    return int(ok.max()) if len(ok) else -1


def passed(rule, d, yd, a, keys=("oof", "fold", "ev")):
    oof, fold, ev = d[keys[0]], d[keys[1]], d[keys[2]]
    if rule == "mean":
        return mean_p(oof, fold, yd, ev) <= a
    R, n = pooled_rank(oof, fold, yd, ev)
    if rule == "pooled":
        return (1 + R) / (n + 1) <= a
    return R <= pac_r(n, a)


def sv(ye, ps):
    return (float((ps & (ye == 1)).sum() / max(ye.sum(), 1)),
            float((ps & (ye == 0)).sum() / max((ye == 0).sum(), 1)))


def oracle(ye, s, a):
    return slip_and_volume(ye, s, threshold_for_slip(ye, s, a))[1]


# ---------------------------------------------------------------- 부트스트랩
def boot(ye, a, b, blocks, fn, n=1000):
    ub = np.unique(blocks)
    idx = pd.Series(np.arange(len(ye))).groupby(blocks).apply(np.array).to_dict()
    obs = fn(ye, a) - fn(ye, b)
    out = []
    for _ in range(n):
        ii = np.concatenate([idx[g] for g in rng.choice(ub, len(ub))])
        if ye[ii].sum() < 5:
            continue
        out.append(fn(ye[ii], a[ii]) - fn(ye[ii], b[ii]))
    d = np.array(out)
    lo, hi = np.quantile(d, [0.025, 0.975])
    return {"obs": float(obs), "pct": [float(lo), float(hi)], "basic": [float(2 * obs - hi), float(2 * obs - lo)],
            "boot_mean": float(d.mean())}


def ap_fn(y, s):
    return average_precision_score(y, s)


def ov_fn(a):
    return lambda y, s: oracle(y, s, a)


# ---------------------------------------------------------------- 보드 지표
def board_stats(ye, board, ps):
    df = pd.DataFrame({"b": board, "y": ye, "p": ps})
    g = df.groupby("b")
    has_def = g["y"].any()
    dd = df[df.y == 1].groupby("b")["p"]
    n = len(ye)
    return {"touch_free": float(g["p"].all().mean()), "board_escape": float(dd.all().mean()),
            "partial_escape": float(dd.any().mean()), "n_def_boards": int(has_def.sum()),
            "saved_per_10k": float((ps & (ye == 0)).sum() / n * 1e4),
            "passed_def_per_10k": float((ps & (ye == 1)).sum() / n * 1e4)}


def msd(v):
    v = np.asarray(v, float)
    return [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0]


def fold_ap(yd, oof, fold):
    """겹 안 AP의 평균. 겹마다 모델이 달라 점수 눈금이 다르면, 겹을 모두 합친 OOF AP는 순위 품질을 낮게 잰다
    (날짜 단위 겹에서 특히 그렇다). 그래서 개발 추정치는 겹 안에서 재고 평균한다."""
    aps = [average_precision_score(yd[fold == k], oof[fold == k]) for k in np.unique(fold)
           if 0 < yd[fold == k].sum() < (fold == k).sum()]
    return float(np.mean(aps))


# ---------------------------------------------------------------- 본 분석
report = {}
MODELS = {"phme": ["lgb_noctx", "lgb_ctx", "two_stage", "boardset_single", "boardset_noctx", "boardset_ctx",
                   "lgbt_noctx", "lgbt_ctx", "two_stage_t",
                   "lgbj_noctx", "lgbj_ctx", "two_stage_j", "boardsetj_ctx", "boardsetj_noctx"],  # j: 7월만으로 고른 설정(35번, 시간순 분할만)
          "siemens": ["lgb_base", "lgb_ctx", "two_stage", "boardset_single", "boardset_noctx", "boardset_ctx",
                      "lgbt_base", "lgbt_ctx", "two_stage_t", "lgb_agg"]}  # lgb_agg: 묶음 전체 집계 특징(34번)
for ds in ["phme", "siemens"]:
    meta = dict(np.load(RES / f"{ds}_meta.npz"))
    y = meta["y"]
    if ds == "phme":
        splits = {s: (meta[f"{s}_dev"], meta[f"{s}_eval"]) for s in ["official", "temporal"]}
        blocks_all, board_all = meta["panel"], meta["board"]
    else:
        h = int(meta["half"])
        splits = {"eval": (np.arange(h), np.arange(h, len(y)))}
        blocks_all = (meta["ts"] // 86_400_000_000_000).astype(np.int64)  # UTC 날짜
        board_all = meta["groups"]
    report[ds] = {}
    for split, (dev, ev) in splits.items():
        yd, ye = y[dev], y[ev]
        R = {"n_dev": [int(len(dev)), int(yd.sum())], "n_eval": [int(len(ev)), int(ye.sum())],
             "models": {}, "leakage": {}, "boot": {}}
        ens = {}
        for m in MODELS[ds]:
            files = sorted(RES.glob(f"{ds}_{split}_{m}_seed*.npz"))
            if not files:
                continue
            rows, raws, rawd = [], [], []
            for f in files:
                d = dict(np.load(f))
                raw_ev, raw_dev = d["ev"].mean(0), d["oof"]
                r = {"dev_ap": average_precision_score(yd, raw_dev), "dev_ap_fold": fold_ap(yd, raw_dev, d["fold"]),
                     "eval_ap": average_precision_score(ye, raw_ev)}
                t = threshold_for_f1(yd, raw_dev)
                r["eval_f1"] = f1_score(ye, (raw_ev >= t).astype(int), zero_division=0)
                for a in ALPHAS:
                    r[f"oracle@{a}"] = oracle(ye, raw_ev, a)
                    for rule in ["pooled", "mean", "pac"]:
                        s_, v_ = sv(ye, passed(rule, d, yd, a))
                        r[f"{rule}_slip@{a}"], r[f"{rule}_vol@{a}"] = s_, v_
                    bs = board_stats(ye, board_all[ev], passed("pooled", d, yd, a))
                    for k, v in bs.items():
                        r[f"board_{k}@{a}"] = v
                if ds == "siemens":
                    for i, sl in enumerate(np.array_split(np.arange(len(ev)), 5)):
                        for a in [0.01, 0.02, 0.05]:
                            ps = passed("pooled", d, yd, a)
                            r[f"slice{i + 1}_slip@{a}"] = sv(ye[sl], ps[sl])[0]
                rows.append(r)
                raws.append(raw_ev)
                rawd.append(raw_dev)
            df = pd.DataFrame(rows)
            ens[m] = np.mean(raws, axis=0)
            t_ens = threshold_for_f1(yd, np.mean(rawd, axis=0))
            R["models"][m] = {"n_seeds": len(files), **{c: msd(df[c]) for c in df.columns},
                              "ens_eval_ap": float(average_precision_score(ye, ens[m])),
                              "ens_eval_f1": float(f1_score(ye, (ens[m] >= t_ens).astype(int))),
                              **{f"ens_oracle@{a}": oracle(ye, ens[m], a) for a in ALPHAS}}
        # 누출: 행 / 보드 / 날짜
        for m in (["lgb_noctx", "lgb_ctx"] if ds == "phme" else ["lgb_base", "lgb_ctx"]) + ["two_stage", "boardset_ctx", "boardset_noctx", "lgb_agg"]:
            lv = {}
            for level in ["row", "board", "day"]:
                pat = f"{ds}_{split}_{m}_day_seed*.npz" if level == "day" else f"{ds}_{split}_{m}_seed*.npz"
                files = sorted(RES.glob(pat))
                keys = ("oof_l", "fold_l", "ev_l") if level == "row" else ("oof", "fold", "ev")
                if not files or keys[0] not in np.load(files[0]).files:
                    continue
                rows = []
                for f in files:
                    d = dict(np.load(f))
                    r = {"dev_ap": average_precision_score(yd, d[keys[0]]), "dev_ap_fold": fold_ap(yd, d[keys[0]], d[keys[1]]),
                         "eval_ap": average_precision_score(ye, d[keys[2]].mean(0))}
                    for a in ALPHAS:
                        for rule in ["pooled", "pac"]:
                            s_, v_ = sv(ye, passed(rule, d, yd, a, keys))
                            r[f"{rule}_slip@{a}"], r[f"{rule}_vol@{a}"] = s_, v_
                        bs = board_stats(ye, board_all[ev], passed("pooled", d, yd, a, keys))
                        r[f"board_escape@{a}"] = bs["board_escape"]
                    if ds == "siemens":
                        for i, sl in enumerate(np.array_split(np.arange(len(ev)), 5)):
                            for a in [0.01, 0.02, 0.05]:
                                r[f"slice{i + 1}_slip@{a}"] = sv(ye[sl], passed("pooled", d, yd, a, keys)[sl])[0]
                                r[f"slice{i + 1}_pac_slip@{a}"] = sv(ye[sl], passed("pac", d, yd, a, keys)[sl])[0]
                    rows.append(r)
                df = pd.DataFrame(rows)
                lv[level] = {"n_seeds": len(files), **{c: msd(df[c]) for c in df.columns}}
            R["leakage"][m] = lv
        # 부트스트랩: 기준 = 맥락 포함 GBDT
        ref = "lgb_ctx"
        nb = 2000  # 3판 심사 의견: 모든 비교 2,000회
        for m in ens:
            if m == ref:
                continue
            R["boot"][f"{m} - {ref}"] = {"ap": boot(ye, ens[m], ens[ref], blocks_all[ev], ap_fn, nb),
                                         "oracle@0.02": boot(ye, ens[m], ens[ref], blocks_all[ev], ov_fn(0.02), nb),
                                         "oracle@0.05": boot(ye, ens[m], ens[ref], blocks_all[ev], ov_fn(0.05), nb)}
        if "two_stage" in ens and "boardset_ctx" in ens:
            R["boot"]["two_stage - boardset_ctx"] = {
                "ap": boot(ye, ens["two_stage"], ens["boardset_ctx"], blocks_all[ev], ap_fn, nb),
                "oracle@0.02": boot(ye, ens["two_stage"], ens["boardset_ctx"], blocks_all[ev], ov_fn(0.02), nb),
                "oracle@0.05": boot(ye, ens["two_stage"], ens["boardset_ctx"], blocks_all[ev], ov_fn(0.05), nb)}
        report[ds][split] = R
        print("=====", ds, split, R["n_dev"], R["n_eval"])
        for m, v in R["models"].items():
            print(f"{m:16s} n={v['n_seeds']} devAP {v['dev_ap'][0]:.3f} evAP {v['eval_ap'][0]:.3f}±{v['eval_ap'][1]:.3f} "
                  f"F1 {v['eval_f1'][0]:.3f}±{v['eval_f1'][1]:.3f} ensAP {v['ens_eval_ap']:.3f} ensF1 {v['ens_eval_f1']:.3f} "
                  f"orc {'/'.join(f'{100*v[f'ens_oracle@{a}']:.1f}' for a in ALPHAS)} | "
                  + " ".join(f"a{a}: P {100*v[f'pooled_slip@{a}'][0]:.1f}/{100*v[f'pooled_vol@{a}'][0]:.1f} "
                             f"M {100*v[f'mean_slip@{a}'][0]:.1f}/{100*v[f'mean_vol@{a}'][0]:.1f} "
                             f"C {100*v[f'pac_slip@{a}'][0]:.1f}/{100*v[f'pac_vol@{a}'][0]:.1f} "
                             f"esc {100*v[f'board_board_escape@{a}'][0]:.1f}" for a in ALPHAS))
        for m, lv in R["leakage"].items():
            for level, v in lv.items():
                print(f"  leak {m:12s} {level:5s} n={v['n_seeds']} devAP {v['dev_ap'][0]:.3f} foldAP {v['dev_ap_fold'][0]:.3f} "
                      f"evAP {v['eval_ap'][0]:.3f} "
                      + " ".join(f"a{a}: {100*v[f'pooled_slip@{a}'][0]:.1f}/{100*v[f'pooled_vol@{a}'][0]:.1f}"
                                 f" pac {100*v[f'pac_slip@{a}'][0]:.1f}/{100*v[f'pac_vol@{a}'][0]:.1f}" for a in ALPHAS)
                      + (" sl@1% " + str([round(100 * v[f'slice{i}_slip@0.01'][0], 1) for i in range(1, 6)])
                         if ds == "siemens" else ""))
        for k, b in R["boot"].items():
            print("  boot", k, {kk: [round(vv["obs"], 3)] + [round(x, 3) for x in vv["pct"]] for kk, vv in b.items()})

# ---------------------------------------------------------------- 대조 실험(Siemens)
cm = CTRL / "siemens_controls_meta.npz"
if cm.exists():
    meta = dict(np.load(cm))
    y = meta["y"]
    report["controls"] = {}
    for split in ["randboard", "randday"]:
        dev, ev = meta[f"{split}_dev"], meta[f"{split}_eval"]
        yd, ye = y[dev], y[ev]
        out = {}
        for m in ["lgb_base", "lgb_ctx"]:
            files = sorted(CTRL.glob(f"siemens_{split}_{m}_seed*.npz"))
            lv = {}
            for level, keys in [("row", ("oof_l", "fold_l", "ev_l")), ("board", ("oof", "fold", "ev")),
                                ("day", ("oof_d", "fold_d", "ev_d"))]:
                rows = []
                for f in files:
                    d = dict(np.load(f))
                    r = {"dev_ap": average_precision_score(yd, d[keys[0]]), "dev_ap_fold": fold_ap(yd, d[keys[0]], d[keys[1]]),
                         "eval_ap": average_precision_score(ye, d[keys[2]].mean(0))}
                    for a in ALPHAS:
                        s_, v_ = sv(ye, passed("pooled", d, yd, a, keys))
                        r[f"slip@{a}"], r[f"vol@{a}"] = s_, v_
                        s_, v_ = sv(ye, passed("pac", d, yd, a, keys))
                        r[f"pac_slip@{a}"], r[f"pac_vol@{a}"] = s_, v_
                    rows.append(r)
                if rows:
                    df = pd.DataFrame(rows)
                    lv[level] = {c: msd(df[c]) for c in df.columns}
            out[m] = lv
        report["controls"][split] = {"n_dev": [int(len(dev)), int(yd.sum())], "n_eval": [int(len(ev)), int(ye.sum())],
                                     "models": out}
        print("===== control", split, report["controls"][split]["n_dev"], report["controls"][split]["n_eval"])
        for m, lv in out.items():
            for level, v in lv.items():
                print(f"  {m:9s} {level:5s} devAP {v['dev_ap'][0]:.3f} foldAP {v['dev_ap_fold'][0]:.3f} evAP {v['eval_ap'][0]:.3f} "
                      + " ".join(f"a{a}: {100*v[f'slip@{a}'][0]:.1f}/{100*v[f'vol@{a}'][0]:.1f}" for a in ALPHAS))
pf = CTRL / "siemens_pfab_protocol.json"
if pf.exists():
    report["pfab_protocol"] = json.loads(pf.read_text())
    for k, v in report["pfab_protocol"].items():
        print("pfab", k, "holdout slip %.3f" % v["holdout_slip"], "slices", [round(x, 3) for x in v["slice_slip"]],
              "2nd half slip %.3f vol %.3f" % (v["second_half_slip"], v["second_half_vol"]))
Path(OUT).write_text(json.dumps(report, indent=1), encoding="utf-8")
