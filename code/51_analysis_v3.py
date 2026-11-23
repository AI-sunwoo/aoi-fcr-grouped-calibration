"""분석 3판 (2026-11-23, 2판 심사 의견 반영). 48번(2판 분석)에 더하는 것만 계산한다.

(a) 실제 놓침·물량의 날짜 묶음 부트스트랩 구간(Siemens 시간순 평가 절반, 씨앗 평균 판정)과
    묶음 단계 사이의 짝 차이(행−보드, 보드−날짜), 그림 2용 구간별 구간.
(b) 같은 물량에서의 보드 유출(정답 기준 문턱값을 물량으로 맞춤) — 모델 비교, 부트스트랩.
(c) 날짜 단위 산포를 '불량 보드 수'로 다시 잰다(보드 안 군집의 영향을 뺀다).
(d) 이동 원점(32번), 대조 나눔 반복(33번), 수리 확인 라벨(36번) 요약.

사용: python -I 51_analysis_v3.py <cf 폴더> <v3 폴더> <출력 json>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import slip_and_volume, threshold_for_slip  # noqa: E402
from fcr_eval import ALPHAS, passed  # noqa: E402

RES, V3, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rng = np.random.default_rng(2027)
NB = 2000
DAY_NS = 86_400_000_000_000
report = {}


def q(v):
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


# =============================================================== (a) 실제 놓침의 부트스트랩
meta = dict(np.load(RES / "siemens_meta.npz"))
y, h, grp = meta["y"], int(meta["half"]), meta["groups"]
yd, ye = y[:h], y[h:]
dayE = (meta["ts"][h:] // DAY_NS).astype(np.int64)
_, dinv = np.unique(dayE, return_inverse=True)
nD = dinv.max() + 1
Dd = np.bincount(dinv, weights=(ye == 1).astype(float), minlength=nD)
Fd = np.bincount(dinv, weights=(ye == 0).astype(float), minlength=nD)
W = rng.multinomial(nD, np.full(nD, 1 / nD), size=NB).astype(float)
slices = np.array_split(np.arange(len(ye)), 5)

CONF = {"lgb_base": ["row", "board", "day"], "lgb_ctx": ["row", "board", "day"], "two_stage": ["board", "day"],
        "boardset_ctx": ["board", "day"], "boardset_noctx": ["board", "day"], "lgb_agg": ["board", "day"]}


def files_for(m, level):
    pat = f"siemens_eval_{m}_day_seed*.npz" if level == "day" else f"siemens_eval_{m}_seed*.npz"
    keys = ("oof_l", "fold_l", "ev_l") if level == "row" else ("oof", "fold", "ev")
    return sorted(RES.glob(pat)), keys


per_day = {}   # (m, level, rule, a) -> (A_def_passed_per_day, B_false_passed_per_day, per-slice dicts)
real = {}
for m, levels in CONF.items():
    for level in levels:
        fl, keys = files_for(m, level)
        if not fl:
            continue
        ds = [dict(np.load(f)) for f in fl]
        for rule in ["pooled", "pac"]:
            for a in ALPHAS:
                A = np.zeros(nD)
                B = np.zeros(nD)
                sl = np.zeros((5, 2))
                for d in ds:
                    ps = passed(rule, d[keys[0]], d[keys[1]], yd, d[keys[2]], a)
                    A += np.bincount(dinv, weights=(ps & (ye == 1)).astype(float), minlength=nD)
                    B += np.bincount(dinv, weights=(ps & (ye == 0)).astype(float), minlength=nD)
                    for i, s in enumerate(slices):
                        sl[i] += [(ps[s] & (ye[s] == 1)).sum(), ye[s].sum()]
                A /= len(ds)
                B /= len(ds)
                per_day[(m, level, rule, a)] = (A, B)
                slip_b, vol_b = (W @ A) / (W @ Dd), (W @ B) / (W @ Fd)
                real[f"{m}|{level}|{rule}|{a}"] = {
                    "n_seeds": len(ds), "slip": float(A.sum() / Dd.sum()), "slip_ci": q(slip_b),
                    "vol": float(B.sum() / Fd.sum()), "vol_ci": q(vol_b),
                    "p_slip_gt_target": float((slip_b > a).mean()),
                    # 11-23 4판 심사 지적: sl[:, 1]도 씨앗마다 더해지므로 len(ds)로 한 번 더 나누면 1/3이 된다
                    "slice_slip": (sl[:, 0] / sl[:, 1]).tolist()}
report["realised"] = real

# 묶음 단계 사이 짝 차이
pairs = {}
for m, a_, b_ in [("lgb_base", "row", "board"), ("lgb_base", "board", "day"), ("lgb_ctx", "row", "board"),
                  ("lgb_ctx", "board", "day"), ("two_stage", "board", "day"), ("boardset_ctx", "board", "day")]:
    for rule in ["pooled", "pac"]:
        for a in ALPHAS:
            if (m, a_, rule, a) not in per_day or (m, b_, rule, a) not in per_day:
                continue
            A1, B1 = per_day[(m, a_, rule, a)]
            A2, B2 = per_day[(m, b_, rule, a)]
            ds_ = (W @ A1 - W @ A2) / (W @ Dd)
            dv_ = (W @ B1 - W @ B2) / (W @ Fd)
            pairs[f"{m}|{a_}-{b_}|{rule}|{a}"] = {"d_slip": float((A1.sum() - A2.sum()) / Dd.sum()), "d_slip_ci": q(ds_),
                                                 "d_vol": float((B1.sum() - B2.sum()) / Fd.sum()), "d_vol_ci": q(dv_)}
report["grouping_pairs"] = pairs

# 그림 2용: 구간별 실제 놓침의 날짜 부트스트랩 구간 (GBDT 맥락 없음, 기대 놓침 규칙)
fig2 = {}
for i, s in enumerate(slices):
    di = dinv[s]
    u, inv = np.unique(di, return_inverse=True)
    Ws = rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=NB).astype(float)
    Dsl = np.bincount(inv, weights=(ye[s] == 1).astype(float), minlength=len(u))
    for level in ["row", "board", "day"]:
        fl, keys = files_for("lgb_base", level)
        ds = [dict(np.load(f)) for f in fl]
        for a in [0.01, 0.02, 0.05]:
            A = np.zeros(len(u))
            for d in ds:
                ps = passed("pooled", d[keys[0]], d[keys[1]], yd, d[keys[2]], a)[s]
                A += np.bincount(inv, weights=(ps & (ye[s] == 1)).astype(float), minlength=len(u))
            A /= len(ds)
            fig2[f"{i + 1}|{level}|{a}"] = {"slip": float(A.sum() / Dsl.sum()), "ci": q((Ws @ A) / np.maximum(Ws @ Dsl, 1)),
                                            "defects": int(Dsl.sum())}
report["fig2_slices"] = fig2
print("(a) done", flush=True)


# =============================================================== (b) 같은 물량에서의 보드 유출
def matched_volume(ye_, board_, block_, scores, vols, nb=NB):  # 11-23 4판: 500 → NB(2,000), 본문 5.6절과 맞춤
    """scores: {모델: 점수}. 물량 v가 되도록 거짓 호출 점수의 분위로 문턱값을 정한다(정답 기준, 비교용)."""
    bu, binv = np.unique(block_, return_inverse=True)
    nbk = len(bu)
    Wb = rng.multinomial(nbk, np.full(nbk, 1 / nbk), size=nb).astype(float)
    out = {}
    pos = ye_ == 1
    dfb = pd.DataFrame({"b": board_[pos], "blk": binv[pos]})
    for m, s in scores.items():
        neg_s, neg_blk = s[~pos], binv[~pos]
        o = np.argsort(neg_s)
        neg_sorted, neg_blk_sorted = neg_s[o], neg_blk[o]
        dfb["s"] = s[pos]
        bmax = dfb.groupby("b").agg(mx=("s", "max"), blk=("blk", "first"))
        res = {}
        for v in vols:
            t = neg_sorted[int(np.floor(v * len(neg_sorted))) - 1]
            esc = (bmax["mx"].values <= t)
            slip = (s[pos] <= t)
            boot_esc, boot_slip = [], []
            for w in Wb:
                cw = np.cumsum(w[neg_blk_sorted])
                k = np.searchsorted(cw, v * cw[-1])
                tb = neg_sorted[min(k, len(neg_sorted) - 1)]
                wb = w[bmax["blk"].values]
                boot_esc.append((wb * (bmax["mx"].values <= tb)).sum() / wb.sum())
                wd = w[binv[pos]]
                boot_slip.append((wd * (s[pos] <= tb)).sum() / wd.sum())
            res[str(v)] = {"slip": float(slip.mean()), "esc": float(esc.mean()), "esc_n": int(esc.sum()),
                           "def_boards": int(len(esc)), "slip_boot": np.array(boot_slip), "esc_boot": np.array(boot_esc)}
        out[m] = res
    return out


def finish_mv(mv, ref):
    rep = {}
    for m, res in mv.items():
        rep[m] = {}
        for v, r in res.items():
            e = {"slip": r["slip"], "esc": r["esc"], "esc_n": r["esc_n"], "def_boards": r["def_boards"],
                 "esc_ci": q(r["esc_boot"]), "slip_ci": q(r["slip_boot"])}
            if m != ref:
                e["d_esc_vs_ref"] = float(r["esc"] - mv[ref][v]["esc"])
                e["d_esc_vs_ref_ci"] = q(r["esc_boot"] - mv[ref][v]["esc_boot"])
                e["d_slip_vs_ref"] = float(r["slip"] - mv[ref][v]["slip"])
                e["d_slip_vs_ref_ci"] = q(r["slip_boot"] - mv[ref][v]["slip_boot"])
            rep[m][v] = e
    return rep


def ens(pattern):
    fl = sorted(RES.glob(pattern))
    return np.mean([np.load(f)["ev"].mean(0) for f in fl], axis=0) if fl else None


VOLS = [0.2, 0.3, 0.4, 0.5, 0.6]
sc = {m: ens(f"siemens_eval_{m}_seed*.npz") for m in ["lgb_base", "lgb_ctx", "two_stage", "boardset_noctx",
                                                       "boardset_ctx", "lgb_agg"]}
sc = {k: v for k, v in sc.items() if v is not None}
report["matched_volume_siemens"] = finish_mv(matched_volume(ye, grp[h:], dayE, sc, VOLS), "lgb_base")
pm = dict(np.load(RES / "phme_meta.npz"))
report["matched_volume_phme"] = {}
for split in ["official", "temporal"]:
    ev = pm[f"{split}_eval"]
    sc = {m: ens(f"phme_{split}_{m}_seed*.npz") for m in ["lgb_noctx", "lgb_ctx", "two_stage", "boardset_ctx"]}
    report["matched_volume_phme"][split] = finish_mv(
        matched_volume(pm["y"][ev], pm["board"][ev], pm["panel"][ev], sc, [0.1, 0.2, 0.3, 0.4]), "lgb_noctx")
print("(b) done", flush=True)

# =============================================================== (c) 날짜 단위 산포(불량 보드 수)
dall = (meta["ts"] // DAY_NS).astype(np.int64)
b = pd.DataFrame({"g": grp, "d": dall, "y": y}).groupby("g").agg(d=("d", "first"), y=("y", "max"))
gd = b.groupby("d")["y"].agg(["sum", "size"])
p = gd["sum"].sum() / gd["size"].sum()
x2 = (((gd["sum"] - gd["size"] * p) ** 2) / (gd["size"] * p * (1 - p))).sum()
report["day_dispersion_boards"] = {"days": int(len(gd)), "boards": int(gd["size"].sum()), "def_board_share": float(p),
                                   "chi2_df": float(x2 / (len(gd) - 1)),
                                   "daily_share_q": [float(x) for x in (gd["sum"] / gd["size"]).quantile([0, .25, .5, .75, 1])]}
print("(c)", report["day_dispersion_boards"], flush=True)

# =============================================================== (d) 요약
def summ(rows):
    rows = np.asarray(rows, float)
    return {"mean": float(rows.mean()), "min": float(rows.min()), "max": float(rows.max()), "n": int(len(rows))}


ro = {}
for f in sorted(V3.glob("rolling_origin_*.json")):
    for k, r in json.loads(f.read_text()).items():
        ro[k] = r
report["rolling_origin"] = ro
cr = {}
for kind in ["randboard", "randday"]:
    f = V3 / f"controls_repeat_{kind}.json"
    if not f.exists():
        continue
    d = json.loads(f.read_text())
    cr[kind] = {}
    for level in ["row", "board", "day"]:
        rows = [r for k, r in d.items() if k.endswith(f"|{level}")]
        if not rows:
            continue
        cr[kind][level] = {"n_partitions": len(rows),
                           **{f"{rule}_slip@{a}": summ([r[f"{rule}_slip@{a}"] for r in rows]) for rule in ["pooled", "pac"]
                              for a in ALPHAS},
                           **{f"pooled_vol@{a}": summ([r[f"pooled_vol@{a}"] for r in rows]) for a in ALPHAS},
                           **{f"exceed@{a}": int(sum(r[f"pooled_slip@{a}"] > a for r in rows)) for a in ALPHAS},
                           "dev_ap_fold": summ([r["dev_ap_fold"] for r in rows]), "eval_ap": summ([r["eval_ap"] for r in rows])}
report["controls_repeat"] = cr


def boot_blocks(ye_, a, b_, blocks, fn, nb=NB):  # 11-23 4판: 1,000 → NB(2,000)
    idx = pd.Series(np.arange(len(ye_))).groupby(blocks).apply(np.array).to_dict()
    ub = np.array(list(idx.keys()))
    out = []
    for _ in range(nb):
        ii = np.concatenate([idx[g] for g in rng.choice(ub, len(ub))])
        if ye_[ii].sum() >= 5:
            out.append(fn(ye_[ii], a[ii]) - fn(ye_[ii], b_[ii]))
    return {"obs": float(fn(ye_, a) - fn(ye_, b_)), "ci": q(np.array(out))}


vm = V3 / "phme_verified_meta.npz"
if vm.exists():
    vmeta = dict(np.load(vm))
    yv, dv, evv = vmeta["y"], vmeta["dev"], vmeta["ev"]
    ydv, yev = yv[dv], yv[evv]
    ver = {"n_dev": [int(len(dv)), int(ydv.sum())], "n_eval": [int(len(evv)), int(yev.sum())], "models": {}, "boot": {}}
    E = {}
    for m in ["lgb_noctx", "lgb_ctx", "two_stage", "boardset_ctx"]:
        fl = sorted(V3.glob(f"phme_verified_{m}_seed*.npz"))
        if not fl:
            continue
        ds = [dict(np.load(f)) for f in fl]
        E[m] = np.mean([d["ev"].mean(0) for d in ds], axis=0)
        r = {"n_seeds": len(ds), "eval_ap": [float(np.mean([average_precision_score(yev, d["ev"].mean(0)) for d in ds])),
                                             float(np.std([average_precision_score(yev, d["ev"].mean(0)) for d in ds], ddof=1))],
             "dev_ap": float(np.mean([average_precision_score(ydv, d["oof"]) for d in ds])),
             "ens_ap": float(average_precision_score(yev, E[m]))}
        for a in [0.02, 0.05, 0.10]:
            r[f"ens_oracle@{a}"] = float(slip_and_volume(yev, E[m], threshold_for_slip(yev, E[m], a))[1])
            sv = [passed("pooled", d["oof"], d["fold"], ydv, d["ev"], a) for d in ds]
            r[f"pooled_slip@{a}"] = float(np.mean([(p_ & (yev == 1)).sum() / yev.sum() for p_ in sv]))
            r[f"pooled_vol@{a}"] = float(np.mean([(p_ & (yev == 0)).sum() / (yev == 0).sum() for p_ in sv]))
        ver["models"][m] = r
    blocks = vmeta["panel"][evv]
    apf = average_precision_score

    def orc(a):
        return lambda yy, ss: slip_and_volume(yy, ss, threshold_for_slip(yy, ss, a))[1]
    for a_, b_ in [("lgb_ctx", "lgb_noctx"), ("two_stage", "lgb_ctx"), ("boardset_ctx", "lgb_ctx"), ("two_stage", "lgb_noctx")]:
        if a_ in E and b_ in E:
            ver["boot"][f"{a_} - {b_}"] = {"ap": boot_blocks(yev, E[a_], E[b_], blocks, apf),
                                           "oracle@0.05": boot_blocks(yev, E[a_], E[b_], blocks, orc(0.05)),
                                           "oracle@0.1": boot_blocks(yev, E[a_], E[b_], blocks, orc(0.10))}
    report["verified"] = ver
    print("verified", json.dumps(ver, indent=0)[:1500], flush=True)

Path(OUT).write_text(json.dumps(report, indent=1), encoding="utf-8")
print("written", OUT)
