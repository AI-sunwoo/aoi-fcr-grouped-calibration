"""PHME 2022 과제 2(작업자가 AOI 판정을 확인하는지) 기준 모델 (2026-11-23).

- 단위: AOI가 잡은 부품(PanelID, FigureID, ComponentID). 라벨: 그 부품의 AOI 행 중 하나라도 작업자 Bad면 1.
- 특징: AOI 판정 종류별 개수, 같은 보드·패널의 맥락(대회 1위가 쓴 것과 같은 종류),
  부품의 SPI 핀 값 요약, AOI가 잡은 핀의 SPI 값, 부품 이름.
- 모델: LightGBM, 씨앗 5개.
- 문턱값: 훈련 자료의 패널 단위 5겹 교차검증 예측(OOF)으로만 정한다. 시험 정답은 보지 않는다.
  F1 문턱값(대회 점수와 비교용)과 놓침 1% 문턱값(현장 기준) 두 가지.
- 분할 두 가지
  official: 대회의 훈련/시험(같은 날짜들에서 패널을 무작위로 나눈 것)
  temporal: 훈련+시험을 합쳐 7월(2019-07-25~29)로 학습, 9월(2019-09-11~25)로 시험

사용: python -I 01_baseline_phme.py <데이터 폴더> <결과 폴더>
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import evaluate, threshold_for_f1, threshold_for_slip, volume_at_slips  # noqa: E402

DATA = Path(sys.argv[1]) / "phme"
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
VARIANT = sys.argv[3] if len(sys.argv) > 3 else "full"
CTX_COLS = ["board_rows", "board_comps", "board_labels", "panel_rows", "panel_boards", "samecomp_boards"]
SLIP = 0.01
SEEDS = [0, 1, 2, 3, 4]
IDS = {"PanelID": str, "FigureID": str, "ComponentID": str, "PinNumber": str, "PadID": str}
K = ["PanelID", "FigureID", "ComponentID"]
NUM = ["Volume(%)", "Height(um)", "Area(%)", "OffsetX(%)", "OffsetY(%)", "SizeX", "SizeY", "Shape(um)",
       "PosX(mm)", "PosY(mm)"]


def log(*a):
    print(*a, flush=True)


def read_aoi(name):
    a = pd.read_csv(DATA / f"AOI_{name}.csv.zip", dtype=IDS)
    a["PinNumber"] = a["PinNumber"].where(a["PinNumber"].notna() & (a["PinNumber"] != "NA"), None)
    return a


def read_spi(prefix, keys):
    parts = sorted(DATA.glob(f"SPI_{prefix}_*.csv.zip"))
    out = []
    for p in parts:
        s = pd.read_csv(p, low_memory=False, dtype=IDS)
        s["FigureID"] = s["FigureID"].str.replace(r"\.0$", "", regex=True)
        s = s.dropna(subset=K)
        out.append(s)
    s = pd.concat(out, ignore_index=True)
    s["Date"] = pd.to_datetime(s["Date"], errors="coerce")
    # 일부 측정 열이 글자로 읽힌다(대회 참가 논문에도 같은 지적). 숫자로 바꾸고 못 바꾸는 값은 비운다.
    for c in NUM:
        s[c] = pd.to_numeric(s[c], errors="coerce")
    return s


def build(aoi, spi):
    """AOI가 잡은 부품마다 특징 한 행."""
    a = aoi.copy()
    a["bad"] = (a["OperatorLabel"] == "Bad").astype(int)
    comp = a.groupby(K).agg(y=("bad", "max"), n_rows=("AOILabel", "size"),
                            n_pinless=("PinNumber", lambda v: int(v.isna().sum())),
                            machine_B=("MachineID", lambda v: int((v == "B").any()))).reset_index()
    lab = pd.crosstab([a[c] for c in K], a["AOILabel"]).add_prefix("aoi_").reset_index()
    comp = comp.merge(lab, on=K, how="left")
    # 맥락: 같은 보드, 같은 패널, 같은 패널의 다른 보드에서 같은 부품
    board = a.groupby(["PanelID", "FigureID"]).agg(board_rows=("AOILabel", "size"),
                                                   board_comps=("ComponentID", "nunique"),
                                                   board_labels=("AOILabel", "nunique")).reset_index()
    panel = a.groupby("PanelID").agg(panel_rows=("AOILabel", "size"),
                                     panel_boards=("FigureID", "nunique")).reset_index()
    samepos = a.groupby(["PanelID", "ComponentID"]).agg(samecomp_boards=("FigureID", "nunique")).reset_index()
    comp = comp.merge(board, on=["PanelID", "FigureID"]).merge(panel, on="PanelID").merge(
        samepos, on=["PanelID", "ComponentID"])
    # SPI: 이 부품의 모든 핀 요약
    s = spi.merge(comp[K], on=K, how="inner")
    g = s.groupby(K)
    agg = g[NUM].agg(["mean", "min", "max", "std"])
    agg.columns = [f"spi_{c}_{f}" for c, f in agg.columns]
    agg["spi_pins"] = g.size()
    agg["spi_not_good"] = g["Result"].agg(lambda v: int((v != "GOOD").sum()))
    agg["date"] = g["Date"].min()
    comp = comp.merge(agg.reset_index(), on=K, how="left")
    # SPI: AOI가 잡은 핀의 값
    ap = a.dropna(subset=["PinNumber"])[K + ["PinNumber"]].drop_duplicates()
    sp = s.merge(ap, on=K + ["PinNumber"], how="inner")
    fp = sp.groupby(K)[["Volume(%)", "Height(um)", "Area(%)", "OffsetX(%)", "OffsetY(%)", "Shape(um)"]].agg(
        ["mean", "max", "min"])
    fp.columns = [f"flagpin_{c}_{f}" for c, f in fp.columns]
    comp = comp.merge(fp.reset_index(), on=K, how="left")
    comp["comp_type"] = comp["ComponentID"].str.extract(r"^([A-Za-z]+)")[0]
    return comp


def features(df):
    drop = {"PanelID", "FigureID", "y", "date"}
    if VARIANT == "no_ctx":
        drop |= set(CTX_COLS)
    if VARIANT == "aoi_only":
        drop |= {c for c in df.columns if c.startswith(("spi_", "flagpin_"))}
    X = df[[c for c in df.columns if c not in drop]].copy()
    for c in ["ComponentID", "comp_type"]:
        X[c] = X[c].astype("category")
    return X


def fit_predict(Xtr, ytr, Xte, seed):
    pos = max(ytr.sum(), 1)
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           scale_pos_weight=(len(ytr) - pos) / pos, random_state=seed, verbose=-1,
                           deterministic=True, force_row_wise=True, n_jobs=16)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def run_split(name, tr, te):
    Xtr, ytr, Xte, yte = features(tr), tr["y"].values, features(te), te["y"].values
    # 시험에 없는 AOI 판정 열은 0으로 채워 훈련과 열을 같게
    for c in Xtr.columns:
        if c not in Xte.columns:
            Xte[c] = 0
    Xte = Xte[Xtr.columns]
    # 범주 열의 범주를 훈련·시험에서 같게
    for c in ["ComponentID", "comp_type"]:
        cats = pd.api.types.union_categoricals([Xtr[c], Xte[c]]).categories
        Xtr[c] = Xtr[c].cat.set_categories(cats)
        Xte[c] = Xte[c].cat.set_categories(cats)
    res = []
    for seed in SEEDS:
        oof = np.zeros(len(tr))
        for f_tr, f_va in GroupKFold(n_splits=5).split(Xtr, ytr, groups=tr["PanelID"]):
            oof[f_va] = fit_predict(Xtr.iloc[f_tr], ytr[f_tr], Xtr.iloc[f_va], seed)
        t_f1 = threshold_for_f1(ytr, oof)
        t_slip = threshold_for_slip(ytr, oof, SLIP)
        score = fit_predict(Xtr, ytr, Xte, seed)
        r = {"seed": seed, "t_f1": t_f1, "t_slip": t_slip,
             "cv": evaluate(ytr, oof, t_f1, t_slip), "test": evaluate(yte, score, t_f1, t_slip)}
        # 참고: 시험 정답을 보고 고른 문턱값(현장에서는 불가능, 비교용)
        t_or = threshold_for_slip(yte, score, SLIP)
        r["test_oracle_slip_threshold"] = evaluate(yte, score, t_f1, t_or)
        r["volume_at_slips"] = volume_at_slips(ytr, oof, yte, score)
        np.savez_compressed(OUT / f"scores_{VARIANT}_{name}_seed{seed}.npz", oof=oof, y_cal=ytr, test=score, y_test=yte)
        res.append(r)
        log(name, seed, json.dumps(r["test"]))
    return res


def summary(res, part):
    keys = ["pr_auc", "f1_bad", "slip", "volume_reduction"]
    return {k: (float(np.mean([r[part][k] for r in res])), float(np.std([r[part][k] for r in res])))
            for k in keys}


aoi_tr, aoi_te = read_aoi("training"), read_aoi("testing")
spi_tr, spi_te = read_spi("training", None), read_spi("testing", None)
tr = build(aoi_tr, spi_tr)
te = build(aoi_te, spi_te)
log("components train/test", len(tr), len(te), "bad", tr.y.sum(), te.y.sum())
out = {}
out["official"] = run_split("official", tr, te)
allc = pd.concat([tr, te], ignore_index=True)
jul = allc[allc["date"] < "2019-08-01"].reset_index(drop=True)
sep = allc[allc["date"] >= "2019-08-01"].reset_index(drop=True)
log("temporal jul/sep", len(jul), len(sep), "bad", jul.y.sum(), sep.y.sum())
out["temporal"] = run_split("temporal", jul, sep)
summ = {s: {"cv": summary(out[s], "cv"), "test": summary(out[s], "test"),
            "test_oracle_slip_threshold": summary(out[s], "test_oracle_slip_threshold")} for s in out}
for sp in ["official", "temporal"]:
    summ[sp]["volume_at_slips"] = {t: {k: float(np.mean([r["volume_at_slips"][t][k] for r in out[sp]]))
                                       for k in out[sp][0]["volume_at_slips"][t]}
                                   for t in out[sp][0]["volume_at_slips"]}
out["summary"] = summ
out["variant"] = VARIANT
out["sizes"] = {"official_train": [int(len(tr)), int(tr.y.sum())], "official_test": [int(len(te)), int(te.y.sum())],
                "temporal_train_jul": [int(len(jul)), int(jul.y.sum())],
                "temporal_test_sep": [int(len(sep)), int(sep.y.sum())]}
(OUT / f"baseline_phme_{VARIANT}.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
log(json.dumps(summ, indent=1))
log(json.dumps(out["sizes"]))
