"""BoardSet 실험 (2026-11-23). 20·21번과 같은 기간 나눔·같은 교차검증 묶음으로 점수를 저장한다.

사용: python -I 22_boardset.py <phme|siemens> <tables 폴더> <결과 폴더> <씨앗 개수> <변형들(쉼표)>
  phme 변형   : ctx(사람이 만든 맥락 개수 포함), noctx(맥락은 집합 층이 스스로 배워야 함)
  siemens 변형: ctx, noctx
  single     : noctx와 같되 집합 크기 1(각 항목이 자기 집합) — 주의(attention)가 다른 호출을 못 보게 해서
               BoardSet의 이득이 집합 층에서 오는지 가린다(2026-11-23 추가). 교차검증 묶음은 그대로 보드·패널.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fcr_eval import siemens_half  # noqa: E402
from boardset import fit_predict, prep_numeric  # noqa: E402
from models import folds  # noqa: E402
from phme_data import CTX_COLS, feature_columns  # noqa: E402

DS, TAB, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
N_SEEDS = int(sys.argv[4])
VARIANTS = sys.argv[5].split(",")
import json  # noqa: E402
CFG = json.loads(Path(sys.argv[6]).read_text()) if len(sys.argv) > 6 else {}
OUT.mkdir(parents=True, exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cat_codes(df_cat: pd.DataFrame, fit_rows, rows_list):
    """훈련 행에서 본 값에 1..K, 처음 보는 값은 0."""
    cards, outs = [], [np.zeros((len(r), df_cat.shape[1]), np.int64) for r in rows_list]
    for j, c in enumerate(df_cat.columns):
        vals = pd.Index(pd.unique(df_cat[c].iloc[fit_rows].astype(str)))
        cards.append(len(vals))
        for o, r in zip(outs, rows_list):
            idx = vals.get_indexer(df_cat[c].iloc[r].astype(str))
            o[:, j] = idx + 1  # -1(모름) → 0
    return cards, outs


def run(name, num: np.ndarray, catdf: pd.DataFrame, y, sets, cv_groups, dev, ev, seed):
    """교차적합: 개발 기간 5겹 각 모델의 OOF 점수와 평가 기간 점수(겹별)를 저장한다."""
    fl = folds(y[dev], cv_groups[dev], seed)
    oof = np.zeros(len(dev), np.float32)
    fold_id = np.full(len(dev), -1)
    evk = np.zeros((len(fl), len(ev)), np.float32)
    for k, (tr, va) in enumerate(fl):
        rtr, rva = dev[tr], dev[va]
        XN_tr, XN_va, XN_ev = prep_numeric(num[rtr], [num[rva], num[ev]])
        cards, (XC_tr, XC_va, XC_ev) = cat_codes(catdf, rtr, [rtr, rva, ev])
        p_va, p_ev = fit_predict(XN_tr, XC_tr, y[rtr], sets[rtr], [XN_va, XN_ev], [XC_va, XC_ev],
                                 [sets[rva], sets[ev]], cards, seed,
                                 val_groups=None if sets is cv_groups else cv_groups[rtr], **CFG)
        oof[va], fold_id[va], evk[k] = p_va, k, p_ev
    np.savez_compressed(OUT / f"{name}_seed{seed}.npz", oof=oof, fold=fold_id, ev=evk)
    log(name, seed, "done")


if DS == "phme":
    tab = pd.read_parquet(TAB / "phme_components.parquet")
    tab = tab[tab["date"].notna()].reset_index(drop=True)
    y = tab["y"].values.astype(int)
    panel = tab["PanelID"].factorize()[0]
    splits = {"official": (np.where(tab["split"] == "train")[0], np.where(tab["split"] == "test")[0]),
              "temporal": (np.where(tab["date"] < "2019-08-01")[0], np.where(tab["date"] >= "2019-08-01")[0])}
    catdf = tab[["ComponentID", "comp_type", "FigureID"]]
    for v in VARIANTS:
        cols = [c for c in feature_columns(tab, use_ctx=(v == "ctx")) if c not in ("ComponentID", "comp_type")]
        num = tab[cols].astype(float).values
        sets = np.arange(len(tab)) if v == "single" else panel
        for split, (dev, ev) in splits.items():
            for seed in range(N_SEEDS):
                run(f"phme_{split}_boardset_{v}", num, catdf, y, sets, panel, dev, ev, seed)
else:
    df = pd.read_parquet(TAB / "siemens_rows.parquet")
    y = df["class"].values.astype(int)
    grp = df["group"].values
    n = len(df)
    half = siemens_half(df)
    dev, ev = np.arange(half), np.arange(half, n)
    catdf = df[["inspection_type", "meta_feat1", "meta_feat2", "meta_feat3", "meta_feat4"]]
    insp = [c for c in df.columns if c.startswith("inspection_feat")]
    ctx = [c for c in df.columns if c.startswith("grp_")]
    for v in VARIANTS:
        num = df[insp + (ctx if v == "ctx" else [])].astype(float).values
        sets = np.arange(n) if v == "single" else grp
        for seed in range(N_SEEDS):
            run(f"siemens_eval_boardset_{v}", num, catdf, y, sets, grp, dev, ev, seed)
