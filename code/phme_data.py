"""PHME 2022 데이터 읽기와 부품 단위 표 만들기 (01_baseline_phme.py의 함수를 모듈로 옮김).

부품 = AOI가 잡은 (PanelID, FigureID, ComponentID). 라벨 y = 그 부품의 AOI 행 중 하나라도 작업자 Bad.
특징 묶음
  aoi_*      : AOI 판정 종류별 개수, 행 수, 핀 없는 행 수, 기계 B 여부
  ctx        : 같은 보드·패널에서 AOI가 잡은 개수 (CTX_COLS)
  spi_*      : 이 부품 모든 핀의 SPI 값 요약
  flagpin_*  : AOI가 잡은 핀의 SPI 값
  ComponentID, comp_type : 범주
"""
from pathlib import Path

import pandas as pd

IDS = {"PanelID": str, "FigureID": str, "ComponentID": str, "PinNumber": str, "PadID": str}
K = ["PanelID", "FigureID", "ComponentID"]
NUM = ["Volume(%)", "Height(um)", "Area(%)", "OffsetX(%)", "OffsetY(%)", "SizeX", "SizeY", "Shape(um)",
       "PosX(mm)", "PosY(mm)"]
CTX_COLS = ["board_rows", "board_comps", "board_labels", "panel_rows", "panel_boards", "samecomp_boards"]
META = ["PanelID", "FigureID", "y", "date", "split"]


def read_aoi(data_dir: Path, name: str) -> pd.DataFrame:
    a = pd.read_csv(data_dir / f"AOI_{name}.csv.zip", dtype=IDS)
    a["PinNumber"] = a["PinNumber"].where(a["PinNumber"].notna() & (a["PinNumber"] != "NA"), None)
    return a


def read_spi(data_dir: Path, prefix: str) -> pd.DataFrame:
    out = []
    for p in sorted(data_dir.glob(f"SPI_{prefix}_*.csv.zip")):
        s = pd.read_csv(p, low_memory=False, dtype=IDS)
        s["FigureID"] = s["FigureID"].str.replace(r"\.0$", "", regex=True)
        s = s.dropna(subset=K)
        out.append(s)
    s = pd.concat(out, ignore_index=True)
    s["Date"] = pd.to_datetime(s["Date"], errors="coerce")
    for c in NUM:
        s[c] = pd.to_numeric(s[c], errors="coerce")
    return s


def build(aoi: pd.DataFrame, spi: pd.DataFrame) -> pd.DataFrame:
    a = aoi.copy()
    a["bad"] = (a["OperatorLabel"] == "Bad").astype(int)
    comp = a.groupby(K).agg(y=("bad", "max"), n_rows=("AOILabel", "size"),
                            n_pinless=("PinNumber", lambda v: int(v.isna().sum())),
                            machine_B=("MachineID", lambda v: int((v == "B").any()))).reset_index()
    lab = pd.crosstab([a[c] for c in K], a["AOILabel"]).add_prefix("aoi_").reset_index()
    comp = comp.merge(lab, on=K, how="left")
    board = a.groupby(["PanelID", "FigureID"]).agg(board_rows=("AOILabel", "size"),
                                                   board_comps=("ComponentID", "nunique"),
                                                   board_labels=("AOILabel", "nunique")).reset_index()
    panel = a.groupby("PanelID").agg(panel_rows=("AOILabel", "size"),
                                     panel_boards=("FigureID", "nunique")).reset_index()
    samepos = a.groupby(["PanelID", "ComponentID"]).agg(samecomp_boards=("FigureID", "nunique")).reset_index()
    comp = comp.merge(board, on=["PanelID", "FigureID"]).merge(panel, on="PanelID").merge(
        samepos, on=["PanelID", "ComponentID"])
    s = spi.merge(comp[K], on=K, how="inner")
    g = s.groupby(K)
    agg = g[NUM].agg(["mean", "min", "max", "std"])
    agg.columns = [f"spi_{c}_{f}" for c, f in agg.columns]
    agg["spi_pins"] = g.size()
    agg["spi_not_good"] = g["Result"].agg(lambda v: int((v != "GOOD").sum()))
    agg["date"] = g["Date"].min()
    comp = comp.merge(agg.reset_index(), on=K, how="left")
    ap = a.dropna(subset=["PinNumber"])[K + ["PinNumber"]].drop_duplicates()
    sp = s.merge(ap, on=K + ["PinNumber"], how="inner")
    fp = sp.groupby(K)[["Volume(%)", "Height(um)", "Area(%)", "OffsetX(%)", "OffsetY(%)", "Shape(um)"]].agg(
        ["mean", "max", "min"])
    fp.columns = [f"flagpin_{c}_{f}" for c, f in fp.columns]
    comp = comp.merge(fp.reset_index(), on=K, how="left")
    comp["comp_type"] = comp["ComponentID"].str.extract(r"^([A-Za-z]+)")[0]
    return comp


def build_table(data_dir: Path) -> pd.DataFrame:
    """훈련·시험을 합친 부품 표. split 열이 train/test. AOI 판정 열은 두 쪽에 모두 있게 맞춘다."""
    parts = []
    for name, spi_prefix, split in [("training", "training", "train"), ("testing", "testing", "test")]:
        t = build(read_aoi(data_dir, name), read_spi(data_dir, spi_prefix))
        t["split"] = split
        parts.append(t)
    tab = pd.concat(parts, ignore_index=True)
    for c in [c for c in tab.columns if c.startswith("aoi_")]:
        tab[c] = tab[c].fillna(0).astype(int)
    return tab


def feature_columns(tab: pd.DataFrame, use_ctx: bool = True, use_spi: bool = True) -> list:
    cols = [c for c in tab.columns if c not in META]
    if not use_ctx:
        cols = [c for c in cols if c not in CTX_COLS]
    if not use_spi:
        cols = [c for c in cols if not c.startswith(("spi_", "flagpin_"))]
    return cols
