"""Siemens: 누출과 시간 변화를 가르는 대조 실험 (2026-11-23, 심사 의견 반영).

part = controls
  시간 분리가 없는 나눔에서 보정 묶음 단위(행/보드/날짜)가 실제 놓침을 어떻게 바꾸는지 본다.
  - randboard : 전체 기간의 타임스탬프 묶음을 무작위 50:50 (같은 날의 다른 보드가 양쪽에 있음)
  - randday   : 전체 기간의 날짜를 무작위 50:50 (같은 날은 한쪽에만)
  각 나눔에서 개발 쪽 5겹 교차적합(행/보드/날짜 묶음) → 26번과 같은 형식으로 저장.
part = pfab
  Pfab & Rothering(2025)의 절차를 그대로 따른다: 앞 절반 안에서 80:20 무작위 나눔, 80%에서 5겹 교차검증,
  겹마다 놓침 1% 문턱값을 구해 평균, 80% 전체로 다시 학습, 20%와 뒤 절반 다섯 구간에서 놓침을 잰다.
  나눔 방식: row(층화 무작위 — 원 논문) / board(타임스탬프 묶음 단위). 모델: LightGBM, 랜덤 포레스트.

사용: python -I 28_siemens_controls.py <tables 폴더> <결과 폴더> <controls|pfab> <씨앗 개수>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold,
                                     StratifiedShuffleSplit)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import slip_and_volume, threshold_for_slip  # noqa: E402
from models import crossfit, lgb_fit_predict  # noqa: E402

TAB, OUT, PART, N_SEEDS = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
OUT.mkdir(parents=True, exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


df = pd.read_parquet(TAB / "siemens_rows.parquet")
y = df["class"].values.astype(int)
grp = df["group"].values
day = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d").factorize()[0]
CAT = ["inspection_type", "meta_feat1", "meta_feat2", "meta_feat3", "meta_feat4"]
insp = [c for c in df.columns if c.startswith("inspection_feat")]
ctx = [c for c in df.columns if c.startswith("grp_")]
Xb = df[CAT + insp].copy()
for c in CAT:
    Xb[c] = Xb[c].astype("category")
Xc = pd.concat([Xb, df[ctx]], axis=1)

if PART == "controls":
    rng = np.random.default_rng(2026)
    ug, ud = np.unique(grp), np.unique(day)
    eval_g = rng.choice(ug, len(ug) // 2, replace=False)
    eval_d = rng.choice(ud, len(ud) // 2, replace=False)
    splits = {"randboard": np.isin(grp, eval_g), "randday": np.isin(day, eval_d)}
    meta = {"y": y, "groups": grp, "day": day}
    for name, is_ev in splits.items():
        meta[f"{name}_dev"] = np.flatnonzero(~is_ev)
        meta[f"{name}_eval"] = np.flatnonzero(is_ev)
        log(name, "dev rows", (~is_ev).sum(), "defects", y[~is_ev].sum(), "eval rows", is_ev.sum(),
            "defects", y[is_ev].sum())
    np.savez_compressed(OUT / "siemens_controls_meta.npz", **meta)
    for name in splits:
        dev, ev = meta[f"{name}_dev"], meta[f"{name}_eval"]
        for seed in range(N_SEEDS):
            for mname, X in [("lgb_base", Xb), ("lgb_ctx", Xc)]:
                g = crossfit(X.iloc[dev], y[dev], grp[dev], seed, X.iloc[ev], grouped=True)
                r = crossfit(X.iloc[dev], y[dev], grp[dev], seed, X.iloc[ev], grouped=False)
                d = crossfit(X.iloc[dev], y[dev], day[dev], seed, X.iloc[ev], grouped=True)
                np.savez_compressed(OUT / f"siemens_{name}_{mname}_seed{seed}.npz",
                                    oof=g[0], fold=g[1], ev=g[2], oof_l=r[0], fold_l=r[1], ev_l=r[2],
                                    oof_d=d[0], fold_d=d[1], ev_d=d[2])
                log(name, mname, seed)
    (OUT / "siemens_controls_done").write_text("done")

elif PART == "pfab":
    half = len(df) // 2
    model_idx = np.arange(half)
    slices = np.array_split(np.arange(half, len(df)), 5)

    def rf_fit_predict(Xtr, ytr, Xte_list, seed):
        Xtr = Xtr.copy()
        for c in CAT:
            Xtr[c] = Xtr[c].cat.codes
        m = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", min_samples_leaf=20,
                                   n_jobs=16, random_state=seed)
        m.fit(Xtr.fillna(-1).values, ytr)
        outs = []
        for Xte in Xte_list:
            Xte = Xte.copy()
            for c in CAT:
                Xte[c] = Xte[c].cat.codes
            outs.append(m.predict_proba(Xte.fillna(-1).values)[:, 1])
        return outs

    def lgb_fp(Xtr, ytr, Xte_list, seed):
        return lgb_fit_predict(Xtr, ytr, Xte_list, seed)

    res = {}
    for split in ["row", "board"]:
        for mname, fp in [("lgb", lgb_fp), ("rf", rf_fit_predict)]:
            rows = []
            for seed in range(N_SEEDS):
                if split == "row":
                    a, b = next(StratifiedShuffleSplit(1, test_size=0.2, random_state=seed)
                                .split(model_idx, y[model_idx]))
                else:
                    a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed)
                                .split(model_idx, y[model_idx], grp[model_idx]))
                hp, te = model_idx[a], model_idx[b]
                Xh, yh = Xb.iloc[hp], y[hp]
                if split == "row":
                    fl = StratifiedKFold(5, shuffle=True, random_state=seed).split(Xh, yh)
                else:
                    fl = StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(Xh, yh, grp[hp])
                ts = []
                for tr, va in fl:
                    p = fp(Xh.iloc[tr], yh[tr], [Xh.iloc[va]], seed)[0]
                    ts.append(threshold_for_slip(yh[va], p, 0.01))
                t = float(np.mean(ts))
                p_te, p_ev = fp(Xh, yh, [Xb.iloc[te], Xb.iloc[half:]], seed)
                r = {"holdout": slip_and_volume(y[te], p_te, t), "slices": []}
                off = 0
                for sl in slices:
                    r["slices"].append(slip_and_volume(y[sl], p_ev[off:off + len(sl)], t))
                    off += len(sl)
                r["second_half"] = slip_and_volume(y[half:], p_ev, t)
                rows.append(r)
                log(split, mname, seed, "holdout slip %.3f vol %.3f" % r["holdout"],
                    "slices slip", [round(s[0], 3) for s in r["slices"]])
            res[f"{split}_{mname}"] = {
                "holdout_slip": float(np.mean([r["holdout"][0] for r in rows])),
                "holdout_vol": float(np.mean([r["holdout"][1] for r in rows])),
                "slice_slip": np.mean([[s[0] for s in r["slices"]] for r in rows], axis=0).tolist(),
                "slice_vol": np.mean([[s[1] for s in r["slices"]] for r in rows], axis=0).tolist(),
                "second_half_slip": float(np.mean([r["second_half"][0] for r in rows])),
                "second_half_vol": float(np.mean([r["second_half"][1] for r in rows])),
                "n_seeds": N_SEEDS}
    (OUT / "siemens_pfab_protocol.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(json.dumps(res, indent=1))
