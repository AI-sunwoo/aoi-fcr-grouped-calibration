"""조율한 Siemens GBDT 결과를 공개 가능한 요약 JSON으로 모은다 (2026-11-23, 모의 심사 A-1 반영).

입력: 29_gbdt_tuned.py(SIEMENS_SPLIT=day)가 남긴 siemens_eval_{lgbt_base,lgbt_ctx,two_stage_t}_seed*.npz와
      gbdt_tuned_siemens.json (같은 결과 폴더).
출력: 모델별 평가 AP(씨앗별, 평균 ± 표준편차)와 고른 설정. 61_supp_tcpmt_db.py가 표 S14의 조율 Siemens 행을 이 파일에서 읽는다.
행 단위 점수는 담지 않는다.

사용: SIEMENS_SPLIT=day python -I 63_tuned_siemens_summary.py <tables 폴더> <결과 폴더> <출력 json>
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fcr_eval import siemens_half  # noqa: E402

TAB, RES, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
assert os.environ.get("SIEMENS_SPLIT") == "day", "run with SIEMENS_SPLIT=day"
df = pd.read_parquet(TAB / "siemens_rows.parquet", columns=["class", "timestamp"])
h = siemens_half(df)
y_ev = df["class"].values.astype(int)[h:]
rep = {"split": "day_boundary", "half": int(h), "config": json.loads((RES / "gbdt_tuned_siemens.json").read_text()), "models": {}}
for m in ["lgbt_base", "lgbt_ctx", "two_stage_t"]:
    files = sorted(RES.glob(f"siemens_eval_{m}_seed*.npz"))
    assert files, m
    aps = []
    for p in files:
        ev = np.load(p)["ev"]
        s = ev.mean(0) if ev.ndim == 2 else ev
        assert len(s) == len(y_ev), (p.name, len(s), len(y_ev))
        aps.append(float(average_precision_score(y_ev, s)))
    rep["models"][m] = {"eval_ap_per_seed": aps, "eval_ap": [float(np.mean(aps)), float(np.std(aps, ddof=1))]}
OUT.write_text(json.dumps(rep))
print(json.dumps(rep["models"], indent=1))
