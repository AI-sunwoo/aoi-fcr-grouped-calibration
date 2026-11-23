"""부품 단위 표를 한 번 만들어 저장한다 (2026-11-23).

출력
  <out>/phme_components.parquet : PHME 부품 표(훈련+시험, split 열)
  <out>/siemens_rows.parquet    : Siemens 행 표(시각순 정렬, 묶음 번호 group, 맥락 특징 포함)

사용: python -I 10_build_tables.py <데이터 폴더> <출력 폴더>
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phme_data import build_table  # noqa: E402

DATA = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

tab = build_table(DATA / "phme")
tab.to_parquet(OUT / "phme_components.parquet", index=False)
print("phme", tab.shape, tab.groupby("split")["y"].agg(["size", "sum"]).to_dict(), flush=True)

df = pd.read_csv(DATA / "siemens" / "dataset.csv", low_memory=False)
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
df["group"] = df["timestamp"].factorize()[0]
insp = [c for c in df.columns if c.startswith("inspection_feat")]
g = df.groupby("group")
df["grp_size"] = g["class"].transform("size")
for t in sorted(df["inspection_type"].unique()):
    df[f"grp_type{t}"] = (df["inspection_type"] == t).groupby(df["group"]).transform("sum")
for c in insp[:10]:
    df[f"grp_dev_{c}"] = df[c] - g[c].transform("mean")
df.to_parquet(OUT / "siemens_rows.parquet", index=False)
print("siemens", df.shape, int(df["class"].sum()), flush=True)
