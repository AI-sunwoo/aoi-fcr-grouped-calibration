"""두 공개 데이터의 구조를 확인한다 (2026-11-23).

확인할 것
- PHME 2022: 훈련·시험의 날짜 범위(시험이 훈련 뒤 기간인지), 패널·보드·부품·핀 수,
  부품당 핀 수, AOI·작업자 라벨과 SPI의 연결, 날짜별 진짜 불량 수
- Siemens AOI: 기간, 라벨, 검사 유형, 같은 시각에 찍힌 행 묶음(보드 검사로 볼 수 있는지),
  범주 열의 종류 수, 날짜별 진짜 불량 비율

사용: python -I 00_profile.py <데이터 폴더> <결과 폴더>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
report = {}


def log(*a):
    print(*a, flush=True)


# ---------------- PHME 2022 ----------------
phme = DATA / "phme"
IDS = {"PanelID": str, "FigureID": str, "ComponentID": str, "PinNumber": str, "PadID": str}
# PanelID는 20자리라 숫자로 읽으면 float로 바뀌어 자릿수가 깨진다. 반드시 문자열로 읽는다.
aoi_tr = pd.read_csv(phme / "AOI_training.csv.zip", dtype=IDS)
aoi_te = pd.read_csv(phme / "AOI_testing.csv.zip", dtype=IDS)


def read_spi(prefix):
    parts = sorted(phme.glob(f"{prefix}_*.csv.zip"))
    df = pd.concat([pd.read_csv(p, low_memory=False, dtype=IDS) for p in parts], ignore_index=True)
    # SPI의 FigureID·PadID는 "1.0"처럼 적혀 있고 AOI는 "1"이다. 맞춰야 이어진다.
    for c in ["FigureID", "PadID"]:
        df[c] = df[c].str.replace(r"\.0$", "", regex=True)
    return df


spi_tr = read_spi("SPI_training")
spi_te = read_spi("SPI_testing")
log("SPI columns:", list(spi_tr.columns))
r = {}
for name, spi, aoi in [("train", spi_tr, aoi_tr), ("test", spi_te, aoi_te)]:
    d = {}
    d["spi_rows"] = int(len(spi))
    d["spi_rows_with_null_ids"] = int(spi[["PanelID", "FigureID", "ComponentID"]].isna().any(axis=1).sum())
    s = spi.dropna(subset=["PanelID", "FigureID", "ComponentID"]).copy()
    for c in ["PanelID", "FigureID", "ComponentID", "PinNumber"]:
        s[c] = s[c].astype(str)
    d["panels"] = int(s["PanelID"].nunique())
    d["boards"] = int(s[["PanelID", "FigureID"]].drop_duplicates().shape[0])
    d["components"] = int(s[["PanelID", "FigureID", "ComponentID"]].drop_duplicates().shape[0])
    d["component_types"] = int(s["ComponentID"].nunique())
    pins = s.groupby(["PanelID", "FigureID", "ComponentID"]).size()
    d["pins_per_component"] = {k: float(v) for k, v in pins.describe().items()}
    d["pins_per_component_counts_top"] = {str(k): int(v) for k, v in pins.value_counts().head(10).items()}
    # 날짜
    dates = pd.to_datetime(s["Date"], errors="coerce")
    d["date_min"] = str(dates.min())
    d["date_max"] = str(dates.max())
    d["n_dates"] = int(dates.dt.date.nunique())
    d["panels_per_date"] = {str(k): int(v) for k, v in
                            s.assign(_d=dates.dt.date).groupby("_d")["PanelID"].nunique().items()}
    d["spi_result_counts"] = {str(k): int(v) for k, v in s["Result"].value_counts().head(10).items()}
    # AOI와 SPI 연결
    a = aoi.copy()
    for c in ["PanelID", "FigureID", "ComponentID"]:
        a[c] = a[c].astype(str)
    comp_lab = a.groupby(["PanelID", "FigureID", "ComponentID"])["OperatorLabel"].apply(
        lambda v: "Bad" if (v == "Bad").any() else "Good").rename("op").reset_index()
    d["aoi_components"] = int(len(comp_lab))
    d["aoi_bad_components"] = int((comp_lab["op"] == "Bad").sum())
    keys = s[["PanelID", "FigureID", "ComponentID"]].drop_duplicates()
    m = comp_lab.merge(keys, how="left", on=["PanelID", "FigureID", "ComponentID"], indicator=True)
    d["aoi_components_found_in_spi"] = int((m["_merge"] == "both").sum())
    d["aoi_rows_without_pin"] = int(a["PinNumber"].isna().sum() + (a["PinNumber"].astype(str) == "NA").sum())
    # 날짜별 AOI 부품과 진짜 불량
    pdate = s.assign(_d=dates.dt.date).groupby("PanelID")["_d"].min().astype(str)
    comp_lab["date"] = comp_lab["PanelID"].map(pdate)
    byd = comp_lab.groupby("date").agg(flagged=("op", "size"), bad=("op", lambda v: int((v == "Bad").sum())))
    d["aoi_by_date"] = {k: {"flagged": int(v.flagged), "bad": int(v.bad)} for k, v in byd.iterrows()}
    # 부품당 AOI가 잡은 핀 수
    d["aoi_rows_per_flagged_component"] = {k: float(v) for k, v in
                                           a.groupby(["PanelID", "FigureID", "ComponentID"]).size().describe().items()}
    d["pos_columns_present"] = [c for c in s.columns if c.startswith("Pos")]
    bl = comp_lab.groupby(["PanelID", "FigureID"])["op"].agg(n="size", bad=lambda v: int((v == "Bad").sum()))
    d["boards_with_aoi_flag"] = int(len(bl))
    d["boards_with_bad"] = int((bl["bad"] > 0).sum())
    base = float((comp_lab["op"] == "Bad").mean())
    multi = bl[bl["n"] >= 2]
    # 같은 보드에 다른 진짜 불량이 있을 때 이 부품이 진짜 불량일 확률
    num = float((multi["bad"] * (multi["bad"] - 1)).sum())
    den = float((multi["bad"] * (multi["n"] - 1)).sum())
    d["p_bad_given_other_bad_same_board"] = num / den if den else None
    d["p_bad_base"] = base
    pl = comp_lab.groupby("PanelID")["op"].agg(n="size", bad=lambda v: int((v == "Bad").sum()))
    d["panels_with_bad"] = int((pl["bad"] > 0).sum())
    r[name] = d
    log(name, json.dumps({k: v for k, v in d.items() if k not in ("panels_per_date", "aoi_by_date")},
                         ensure_ascii=False)[:1500])
    log(name, "aoi_by_date", d["aoi_by_date"])
# 훈련과 시험이 겹치는 패널
tr_p = set(spi_tr["PanelID"].dropna().astype(str))
te_p = set(spi_te["PanelID"].dropna().astype(str))
r["panel_overlap_train_test"] = len(tr_p & te_p)
report["phme"] = r
del spi_tr, spi_te

# ---------------- Siemens AOI ----------------
sm = pd.read_csv(DATA / "siemens" / "dataset.csv", low_memory=False)
sm = sm.drop(columns=[c for c in sm.columns if c.startswith("Unnamed")])
sm["timestamp"] = pd.to_datetime(sm["timestamp"], utc=True)
d = {}
d["rows"] = int(len(sm))
d["columns"] = list(sm.columns[:8]) + [f"... {len(sm.columns)} cols"]
d["time_min"], d["time_max"] = str(sm["timestamp"].min()), str(sm["timestamp"].max())
d["n_days"] = int(sm["timestamp"].dt.date.nunique())
d["class_counts"] = {str(k): int(v) for k, v in sm["class"].value_counts().items()}
d["by_inspection_type"] = {str(k): {"rows": int(len(g)), "defect": int(g["class"].sum())}
                           for k, g in sm.groupby("inspection_type")}
for c in ["meta_feat1", "meta_feat2", "meta_feat3", "meta_feat4"]:
    d[f"{c}_nunique"] = int(sm[c].nunique())
# 같은 시각 묶음
grp = sm.groupby("timestamp")
size = grp.size()
d["timestamp_groups"] = int(len(size))
d["rows_per_timestamp"] = {k: float(v) for k, v in size.describe().items()}
d["rows_per_timestamp_counts_top"] = {str(k): int(v) for k, v in size.value_counts().head(10).items()}
gdef = grp["class"].sum()
gs = pd.DataFrame({"n": size, "bad": gdef})
gm = gs[gs["n"] >= 2]
d["p_defect_given_other_defect_same_timestamp"] = float((gm["bad"] * (gm["bad"] - 1)).sum() / (gm["bad"] * (gm["n"] - 1)).sum())
d["p_defect_base"] = float(sm["class"].mean())
d["groups_with_defect"] = int((gdef > 0).sum())
d["defect_groups_size_mean"] = float(size[gdef > 0].mean())
# 같은 시각 안에서 범주 열이 같은지(같은 보드라면 어떤 열은 같아야 한다)
for c in ["meta_feat1", "meta_feat2", "meta_feat3", "meta_feat4", "inspection_type"]:
    d[f"{c}_constant_within_timestamp_share"] = float((grp[c].nunique() == 1).mean())
# 날짜별 불량 비율
daily = sm.groupby(sm["timestamp"].dt.date)["class"].agg(["size", "sum"])
daily["rate"] = daily["sum"] / daily["size"]
d["daily_defect_rate"] = {k: float(v) for k, v in daily["rate"].describe().items()}
d["daily_rows"] = {k: float(v) for k, v in daily["size"].describe().items()}
# 시간 순서대로 5등분했을 때 구간별 불량 비율과 검사 유형 비중
sm = sm.sort_values("timestamp").reset_index(drop=True)
sm["slice"] = pd.qcut(np.arange(len(sm)), 10, labels=False)
sl = sm.groupby("slice").agg(start=("timestamp", "min"), rows=("class", "size"), defect=("class", "sum"))
sl["rate"] = sl["defect"] / sl["rows"]
d["deciles"] = [{"slice": int(i), "start": str(v.start), "rows": int(v.rows), "defect": int(v.defect),
                 "rate": round(float(v.rate), 5)} for i, v in sl.iterrows()]
mix = pd.crosstab(sm["slice"], sm["inspection_type"], normalize="index").round(3)
d["inspection_type_mix_by_decile"] = mix.to_dict(orient="index")
report["siemens"] = d
log("siemens", json.dumps(d, ensure_ascii=False, default=str)[:4000])

(OUT / "profile.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
log("saved", OUT / "profile.json")
