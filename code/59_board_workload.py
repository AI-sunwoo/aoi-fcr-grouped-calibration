"""보드 단위 결과와 작업량 (2026-11-23, 모의 심사 4번 반영). Siemens 시간순(날짜 경계) 평가 절반, 기대 놓침 규칙.

보드 지표 세 가지(불량 보드 기준 두 가지 + 전체 보드 기준 한 가지):
  partial_escape : 불량 호출이 하나라도 자동 통과된 불량 보드의 비율 — 통과된 불량은 작업자에게 보이지 않으므로
                   불량을 품은 채 나갈 수 있는 보드의 상한
  full_escape    : 불량 호출이 모두 자동 통과된 불량 보드의 비율 — 수동 확인에서 그 보드의 불량이 하나도 보이지 않는 경우
  touch_free     : 모든 호출이 자동 통과돼 수동 확인 자리를 거치지 않는 보드의 비율(전체 보드 기준)
작업량: 1만 호출당 확인이 줄어든 가성 호출 수, 통과된 불량 호출 수. 시드별로 계산해 평균한다.

사용: python -I 59_board_workload.py <cf 폴더> <출력 json>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fcr_eval import passed  # noqa: E402

RES, OUT = Path(sys.argv[1]), Path(sys.argv[2])
meta = dict(np.load(RES / "siemens_meta.npz"))
y, h, grp = meta["y"], int(meta["half"]), meta["groups"]
yd, ye, gb = y[:h], y[h:], grp[h:]
MODELS = ["lgb_base", "lgb_ctx", "lgb_agg", "two_stage", "two_stage_dd", "boardset_ctx", "boardset_noctx"]


def stats(ps):
    df = pd.DataFrame({"b": gb, "y": ye, "p": ps})
    g = df.groupby("b")
    dd = df[df.y == 1].groupby("b")["p"]
    n = len(ye)
    return {"slip": float(ps[ye == 1].mean()), "vol": float(ps[ye == 0].mean()),
            "partial_escape": float(dd.any().mean()), "full_escape": float(dd.all().mean()),
            "touch_free": float(g["p"].all().mean()),
            "touch_free_def": float(df[df.b.isin(dd.groups.keys())].groupby("b")["p"].all().mean()),
            "saved_per_10k": float((ps & (ye == 0)).sum() / n * 1e4),
            "passed_def_per_10k": float((ps & (ye == 1)).sum() / n * 1e4)}


rep = {"n_eval": int(len(ye)), "n_def": int(ye.sum()), "n_boards": int(len(np.unique(gb))),
       "n_def_boards": int(pd.Series(gb[ye == 1]).nunique()), "per_10k_def": float(ye.mean() * 1e4)}
for m in MODELS:
    for level, pat in [("board", f"siemens_eval_{m}_seed*.npz"), ("day", f"siemens_eval_{m}_day_seed*.npz")]:
        fl = sorted(RES.glob(pat))
        if not fl:
            continue
        ds = [dict(np.load(f)) for f in fl]
        for a in [0.01, 0.02, 0.05, 0.10]:
            S = [stats(passed("pooled", d["oof"], d["fold"], yd, d["ev"], a)) for d in ds]
            rep[f"{m}|{level}|{a}"] = {"n_seeds": len(ds), **{k: float(np.mean([s[k] for s in S])) for k in S[0]}}
Path(OUT).write_text(json.dumps(rep, indent=1), encoding="utf-8")
print({k: v for k, v in rep.items() if not isinstance(v, dict)})
for k, v in rep.items():
    if isinstance(v, dict) and ("|0.02" in k or "|0.05" in k):
        print(f"{k:28s} slip {100*v['slip']:5.2f} vol {100*v['vol']:5.1f} partial {100*v['partial_escape']:5.1f} "
              f"full {100*v['full_escape']:5.1f} touchfree {100*v['touch_free']:5.1f} saved {v['saved_per_10k']:6.0f} "
              f"passed {v['passed_def_per_10k']:4.1f}")
