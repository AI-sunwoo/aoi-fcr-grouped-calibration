"""보드당 추론 시간 측정 (2026-11-23, 편집자 관점 점검 ③ 반영). Siemens 날짜 경계 분할.

배치 상황: 한 보드의 AOI 호출이 모두 나온 뒤 FCR이 그 보드를 한 번에 판정한다. 보드 하나의 판정 시간 =
  GBDT(맥락 없음/맥락): 다섯 겹 모델 예측의 합
  2단계: 다섯 1단계 예측 + 같은 보드 이웃 특징 계산 + 다섯 2단계 예측
  BoardSet: 다섯 겹 × 초기화 3개 = 15번의 순전파(같은 구조: 폭 128, 2층, 헤드 4). 추론 시간은 가중치 값과
            무관하므로 무작위 초기화 망으로 잰다.
GBDT 모델은 개발 절반으로 실제로 학습한다(겹 나눔은 무작위 5등분; 시간 측정용). 평가 절반의 보드 1,000개(무작위)와
가장 큰 보드를 CPU 1스레드로, BoardSet은 GPU로도 잰다.

사용: python -I 62_latency.py <tables 폴더> <출력 json>
"""
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from boardset import BoardSetNet, prep_numeric  # noqa: E402
from fcr_eval import siemens_half  # noqa: E402
from models import LGB_PARAMS, neighbour_features  # noqa: E402

TAB, OUT = Path(sys.argv[1]), Path(sys.argv[2])
os.environ["SIEMENS_SPLIT"] = "day"
torch.set_num_threads(1)
df = pd.read_parquet(TAB / "siemens_rows.parquet")
y = df["class"].values.astype(int)
grp = df["group"].values
h = siemens_half(df)
dev, ev = np.arange(h), np.arange(h, len(df))
CAT = ["inspection_type", "meta_feat1", "meta_feat2", "meta_feat3", "meta_feat4"]
insp = [c for c in df.columns if c.startswith("inspection_feat")]
ctx = [c for c in df.columns if c.startswith("grp_")]
Xb = df[CAT + insp].copy()
for c in CAT:
    Xb[c] = Xb[c].astype("category")
Xc = pd.concat([Xb, df[ctx]], axis=1)
rng = np.random.default_rng(0)
fold = rng.integers(0, 5, len(dev))


def fit(X, yy, seed):
    p = dict(LGB_PARAMS, n_jobs=int(os.environ.get("LAT_TRAIN_THREADS", "16")))   # 운영 서버에서는 줄여서 실행
    pos = max(int(yy.sum()), 1)
    m = lgb.LGBMClassifier(scale_pos_weight=(len(yy) - pos) / pos, random_state=seed, **p)
    m.fit(X, yy)
    return m


t0 = time.time()
M_base = [fit(Xb.iloc[dev[fold != k]], y[dev[fold != k]], k) for k in range(5)]
M_ctx = [fit(Xc.iloc[dev[fold != k]], y[dev[fold != k]], k) for k in range(5)]
s1_dev = np.mean([m.predict_proba(Xc.iloc[dev], num_threads=int(os.environ.get("LAT_TRAIN_THREADS", "16")))[:, 1] for m in M_ctx], axis=0)
nb = neighbour_features(s1_dev, {"grp": grp[dev]})
nb.index = Xc.index[dev]
X2_dev = pd.concat([Xc.iloc[dev], nb], axis=1)
M_two = [fit(X2_dev.iloc[np.flatnonzero(fold != k)], y[dev[fold != k]], k) for k in range(5)]
print("training for timing done", round(time.time() - t0), "s", flush=True)

# BoardSet inputs and nets
num_cols = insp + ctx
XN_dev, XN_ev = prep_numeric(df[num_cols].values[dev].astype(float), [df[num_cols].values[ev].astype(float)])
cards, XC = [], np.zeros((len(df), len(CAT)), np.int64)
for j, c in enumerate(CAT):
    codes, uniq = pd.factorize(df[c].astype(str))
    XC[:, j] = codes + 1
    cards.append(len(uniq))
XC_ev = XC[ev]
nets = []
for i in range(15):
    torch.manual_seed(i)
    net = BoardSetNet(XN_ev.shape[1], cards, d=128, heads=4, layers=2, dropout=0.1).eval()
    nets.append(net)

ev_groups = pd.Series(np.arange(len(ev))).groupby(grp[ev]).apply(np.array)
sizes = ev_groups.apply(len)
sample = list(rng.choice(ev_groups.index.values, 1000, replace=False)) + [sizes.idxmax()]


def t_gbdt(models, X):
    t = time.perf_counter()
    s = np.mean([m.predict_proba(X, num_threads=1)[:, 1] for m in models], axis=0)
    return (time.perf_counter() - t) * 1e3, s


def t_two(Xcb):
    t = time.perf_counter()
    s1 = np.mean([m.predict_proba(Xcb, num_threads=1)[:, 1] for m in M_ctx], axis=0)
    f = neighbour_features(s1, {"grp": np.zeros(len(s1), int)})
    f.index = Xcb.index
    X2 = pd.concat([Xcb, f], axis=1)
    _ = [m.predict_proba(X2, num_threads=1)[:, 1] for m in M_two]
    return (time.perf_counter() - t) * 1e3


@torch.no_grad()
def t_bs(rows, device):
    xn = torch.as_tensor(XN_ev[rows][None], device=device)
    xc = torch.as_tensor(XC_ev[rows][None], device=device)
    pad = torch.zeros((1, len(rows)), dtype=torch.bool, device=device)
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for net in nets:
        net(xn, xc, pad)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1e3


res = {k: [] for k in ["gbdt_base", "gbdt_ctx", "two_stage", "boardset_cpu", "boardset_gpu"]}
gpu = torch.cuda.is_available()
if gpu:
    for net in nets:
        net.to("cuda")
    t_bs(ev_groups[sample[0]], "cuda")
for b in sample:
    rows = ev_groups[b]
    idx = ev[rows]
    res["gbdt_base"].append(t_gbdt(M_base, Xb.iloc[idx])[0])
    res["gbdt_ctx"].append(t_gbdt(M_ctx, Xc.iloc[idx])[0])
    res["two_stage"].append(t_two(Xc.iloc[idx]))
    if gpu:
        res["boardset_gpu"].append(t_bs(rows, "cuda"))
if gpu:
    for net in nets:
        net.to("cpu")
for b in sample:
    res["boardset_cpu"].append(t_bs(ev_groups[b], "cpu"))
cpu = subprocess.run(["bash", "-c", "LC_ALL=C lscpu | grep 'Model name' | head -1"], capture_output=True, text=True).stdout.strip()
rep = {"cpu": cpu, "machine": platform.machine(), "gpu": torch.cuda.get_device_name(0) if gpu else None,
       "threads_cpu": 1, "boards_timed": len(sample) - 1, "largest_board_calls": int(sizes.max()),
       "median_board_calls": float(sizes.median()), "summary": {}}
for k, v in res.items():
    if not v:
        continue
    a = np.array(v[:-1])
    rep["summary"][k] = {"mean_ms": float(a.mean()), "median_ms": float(np.median(a)), "p95_ms": float(np.quantile(a, 0.95)),
                         "max_ms": float(a.max()), "largest_board_ms": float(v[-1])}
Path(OUT).write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
