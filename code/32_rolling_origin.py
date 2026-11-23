"""Siemens 이동 원점(rolling-origin) 평가 (2026-11-23, 3차 검토 반영).

한 번의 절반 나눔 대신 여러 시점에서 "그때까지의 모든 날짜로 개발 → 다음 16일 평가"를 되풀이한다.
원점(개발에 쓰는 날짜 수) = 36, 52, 68, 84일. 평가 창 = 다음 16일(마지막은 끝까지).
모델: lgb_base(맥락 없음) / lgb_ctx(맥락) — 행 / 보드 / 날짜 단위 교차적합,
      two_stage — 보드 / 날짜 단위(날짜일 때 안쪽 겹은 보드 단위, 31번과 같음).
씨앗 0. 지표만 json으로 쓴다(fcr_eval.summarise).

사용: python -I 32_rolling_origin.py <tables 폴더> <결과 폴더> <모델들(쉼표)> <스레드 수>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fcr_eval import summarise  # noqa: E402
from models import LGB_PARAMS, crossfit, folds, lgb_fit_predict, neighbour_features, oof_scores  # noqa: E402

TAB, OUT, MODELS, NJ = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split(","), int(sys.argv[4])
OUT.mkdir(parents=True, exist_ok=True)
PARAMS = dict(LGB_PARAMS, n_jobs=NJ)
ORIGINS, WIN = [36, 52, 68, 84], 16
GAP = int(__import__("os").environ.get("ROLL_GAP", "0"))   # 11-23 모의 심사 2번: 개발 끝과 평가 시작 사이 간격(일)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def two_stage_crossfit(X_dev, y_dev, outer, inner, lv_dev, X_ev, lv_ev, seed, params):
    def stack(X, s1, levels):
        nb = neighbour_features(s1, levels)
        nb.index = X.index
        return pd.concat([X, nb], axis=1)
    fl = folds(y_dev, outer, seed)
    oof2, fold_id, ev2 = np.zeros(len(y_dev)), np.full(len(y_dev), -1), np.zeros((len(fl), len(X_ev)))
    for k, (tr, va) in enumerate(fl):
        Xtr, ytr = X_dev.iloc[tr], y_dev[tr]
        s1_tr = oof_scores(Xtr, ytr, inner[tr], seed, grouped=True, params=params)
        s1_va, s1_ev = lgb_fit_predict(Xtr, ytr, [X_dev.iloc[va], X_ev], seed, params)
        p_va, p_ev = lgb_fit_predict(stack(Xtr, s1_tr, {n: v[tr] for n, v in lv_dev.items()}), ytr,
                                     [stack(X_dev.iloc[va], s1_va, {n: v[va] for n, v in lv_dev.items()}),
                                      stack(X_ev, s1_ev, lv_ev)], seed, params)
        oof2[va], fold_id[va], ev2[k] = p_va, k, p_ev
    return oof2, fold_id, ev2


df = pd.read_parquet(TAB / "siemens_rows.parquet")
y = df["class"].values.astype(int)
grp = df["group"].values
dstr = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d").values
days = np.sort(np.unique(dstr))
day = np.searchsorted(days, dstr)
CAT = ["inspection_type", "meta_feat1", "meta_feat2", "meta_feat3", "meta_feat4"]
insp = [c for c in df.columns if c.startswith("inspection_feat")]
ctx = [c for c in df.columns if c.startswith("grp_")]
Xb = df[CAT + insp].copy()
for c in CAT:
    Xb[c] = Xb[c].astype("category")
Xc = pd.concat([Xb, df[ctx]], axis=1)
out_file = OUT / (f"rolling_origin_{'_'.join(MODELS)}" + (f"_gap{GAP}" if GAP else "") + ".json")
res = json.loads(out_file.read_text()) if out_file.exists() else {}
for o in ORIGINS:
    dev = np.flatnonzero(day < o)
    start = o + GAP
    end = len(days) if o == ORIGINS[-1] else min(start + WIN, len(days))
    ev = np.flatnonzero((day >= start) & (day < end))
    yd, ye = y[dev], y[ev]
    info = {"dev_days": int(o), "gap_days": GAP, "eval_days": int(end - start), "n_dev": [int(len(dev)), int(yd.sum())],
            "n_eval": [int(len(ev)), int(ye.sum())], "eval_def_boards": int(pd.Series(grp[ev][ye == 1]).nunique())}
    log("origin", o, info)
    for m in MODELS:
        for level in (["row", "board", "day"] if m.startswith("lgb") else ["board", "day"]):
            key = f"{o}|{m}|{level}"
            if key in res:
                continue
            t0 = time.time()
            if m.startswith("lgb"):
                X = Xb if m == "lgb_base" else Xc
                g = {"row": grp, "board": grp, "day": day}[level]
                oof, fold, evs = crossfit(X.iloc[dev], yd, g[dev], 0, X.iloc[ev], grouped=(level != "row"), params=PARAMS)
            else:
                outer = grp if level == "board" else day
                oof, fold, evs = two_stage_crossfit(Xc.iloc[dev], yd, outer[dev], grp[dev], {"grp": grp[dev]},
                                                    Xc.iloc[ev], {"grp": grp[ev]}, 0, PARAMS)
            r = summarise(yd, ye, grp[ev], oof, fold, evs)
            res[key] = {**info, **r, "sec": round(time.time() - t0, 1)}
            out_file.write_text(json.dumps(res, indent=1))
            log(key, "slip@1/2/5%", [round(100 * r[f"pooled_slip@{a}"], 1) for a in [0.01, 0.02, 0.05]],
                "vol@5%", round(100 * r["pooled_vol@0.05"], 1), "sec", res[key]["sec"])
log("done")
