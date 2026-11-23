"""원고 2판 3.4절·6.5절의 기술 통계 (2026-11-23). 날짜 단위 산포, Siemens 불량 보드 구성.

사용: python -I 50_dependence_stats.py <tables 폴더>
"""
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import chi2

TAB = Path(sys.argv[1])


def dispersion(day, y):
    g = pd.DataFrame({"day": day, "y": y}).groupby("day")["y"].agg(["sum", "size"])
    p = g["sum"].sum() / g["size"].sum()
    x2 = (((g["sum"] - g["size"] * p) ** 2) / (g["size"] * p * (1 - p))).sum()
    return g, p, x2 / (len(g) - 1), chi2.sf(x2, len(g) - 1)


df = pd.read_parquet(TAB / "siemens_rows.parquet")
day = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")
g, p, disp, pv = dispersion(day, df["class"].values)
r = g["sum"] / g["size"]
print("Siemens: days", len(g), "overall defect share %", round(100 * p, 3),
      "daily share % min/q25/median/q75/max", [round(100 * x, 2) for x in r.quantile([0, .25, .5, .75, 1])],
      "days without defect", int((g["sum"] == 0).sum()), "dispersion chi2/df", round(disp, 1), "p", pv,
      "median calls per day", int(g["size"].median()))
half = len(df) // 2
for name, part in [("dev", df.iloc[:half]), ("eval", df.iloc[half:])]:
    b = part.groupby("group")["class"].agg(["sum", "size"])
    d = b[b["sum"] > 0]
    print(f"Siemens {name}: calls", len(part), "days", day.iloc[part.index].nunique(), "boards", len(b),
          "defective boards", len(d), "share %", round(100 * len(d) / len(b), 2),
          "single-defect share of defective boards %", round(100 * (d["sum"] == 1).mean(), 1),
          "defects on multi-defect boards %", round(100 * d.loc[d["sum"] > 1, "sum"].sum() / d["sum"].sum(), 1))

t = pd.read_parquet(TAB / "phme_components.parquet")
t = t[t["date"].notna()]
g, p, disp, pv = dispersion(pd.to_datetime(t["date"]).dt.strftime("%Y-%m-%d"), t["y"].values)
print("PHME: days", len(g), "daily share %", [round(100 * x, 2) for x in (g["sum"] / g["size"]).tolist()],
      "components per day", g["size"].tolist(), "dispersion chi2/df", round(disp, 1), "p", pv)
