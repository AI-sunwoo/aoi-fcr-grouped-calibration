# aoi-fcr-grouped-calibration

Code, configurations and aggregated results for the paper

> **Board- and Day-Level Dependence in Automated Optical Inspection False-Call Reduction: Grouped Calibration and Board-Context Models**
> Sunwoo Lee, Hyeonbin Kim, and Su Young Kim — submitted to *IEEE Transactions on Components, Packaging and Manufacturing Technology* (under review).

False-call reduction (FCR) auto-passes low-risk AOI calls while limiting the share of true defects that slip through. On two public SMT datasets (PHME 2022 / Bitron and Siemens AOI), true defects cluster within boards, panels and production days. The study varies only how the folds of cross-fitted conformal slip thresholds are grouped (call, board, time window, day) and measures realized slip and the share of false calls auto-passed, together with a two-stage neighbor model and a set-transformer model (BoardSet) that use the other calls of a board.

## What is (and is not) in this repository

| Included | Not included |
|---|---|
| All analysis code (`code/`) | Raw data (download from the sources below) |
| Model configurations selected on development data (`configs/`) | Derived tables (`tables/*.parquet`) and row-level model scores (`*.npz`) |
| Aggregated results behind every table and figure (`results/`) | The manuscript |

Row-level scores are not redistributed; they can be regenerated from the public data with the code.

## Data

| Dataset | Source | Licence / notice |
|---|---|---|
| PHME 2022 data challenge (Bitron) | https://github.com/PHME-Datachallenge/Data-Challenge-2022 | CC BY-NC-SA 4.0 |
| Siemens AOI false-call data, version 2 | https://doi.org/10.17632/99jzmh9658.2 | CC BY-NC 3.0 and the record's legal notice (non-commercial testing in connection with the data article) |

This is non-commercial research. Place the files as `data/phme/*.csv.zip` and `data/siemens/{dataset.csv,mapping.json}`. SHA-256 of every file used:

```
a1bfec7626dfce694fe3f8a0a3e087a533d83d15c3530786bfcaf7723fea9c11  phme/AOI_training.csv.zip
02352e3a44d2343bf34d9c5ca18e94644e54bbf6606520d4b380b1da863b2b87  phme/AOI_testing.csv.zip
317ef4bf7bd0ab4eacdb0c9adcb3f3eefc19c1fa7f63b9729b3233b8b6b8a87a  phme/SPI_training_0.csv.zip
f91135f23525e686ccb7609569dea3697e0039f5abca886845e9f99f0fd4b9a7  phme/SPI_training_1.csv.zip
f2e4caabc43c0249095b3d9b21ef39f1431f5352492d6f34f2405b02d63660e8  phme/SPI_training_2.csv.zip
c5cfd232a73b48021fbcb396d87444e413b7646ad8915b2938617d6e37da7e3b  phme/SPI_training_3.csv.zip
fce397628226f43b21fe4b391196b609dc5723586c9e171ac607be42e206003c  phme/SPI_testing_0.csv.zip
7d4f613f6d4923beece4bf20161fc9d3c3fc691acc32e9250413d318515d8956  phme/SPI_testing_1.csv.zip
53e8568743216d556856ed69b388f6750fbfa0b8c59ad31f970515ac9eb10e62  siemens/dataset.csv
3b20f440b6d9ed0baefa662e1a6f03688befbe0f28341a3b54655d3058c6e486  siemens/mapping.json
```

## Environment

Python 3.12 with the versions in `requirements.txt` (numpy 2.3.5, pandas 2.3.3, pyarrow, scikit-learn 1.9.1, LightGBM 4.7.0, SciPy 1.18.1, PyTorch 2.12.0, matplotlib). LightGBM runs with `deterministic=True, force_row_wise=True`. Results were produced on NVIDIA DGX Spark machines (GB10); BoardSet uses a GPU if available. Code comments are partly in Korean; the script index below summarizes each script.

## Split definitions

- **PHME official**: training release → test release. **PHME temporal**: inspection date < 2019-08-01 (July) → ≥ 2019-08-01 (September).
- **Siemens chronological (main analysis)**: rows in time order, cut at the calendar-day boundary closest to the row midpoint (row 226,622 of 440,274; development 64 days, 1,752 defects; evaluation 37 days, 2,870 defects; no shared day or timestamp group). Set `SIEMENS_SPLIT=day`. Without it the row midpoint is used (as in Pfab & Rothering, 2025), which shares one day and one timestamp group; it is used only to reproduce their protocol (Table S2).
- **Siemens controls**: 20 random-board and 20 random-day halves, partition *i* drawn with `numpy.random.default_rng(5000+i)`.
- **Rolling origins**: development on the first 36, 52, 68 or 84 days with data, evaluation on the next 16 days with data (last origin: to the end); `ROLL_GAP=7` shifts each evaluation window 7 recorded production days later.
- **Cross-fitting folds**: `StratifiedGroupKFold(5, shuffle=True, random_state=seed)`, grouped by nothing (row-wise), timestamp group / panel (board-wise) or calendar date (day-blocked); seeds 0–2 (Siemens), 0–4 (PHME).

## Reproduction

Run from the repository root with `P="python -I code"`, `T=tables`, `D=data`. The sequence below is complete but has not yet been run end to end in a fresh environment.

```bash
# 0. Tables
$P/10_build_tables.py $D $T

# 1. Model configurations (development data only; copies of the selected ones are in configs/)
$P/23_boardset_hpsearch.py phme    $T results/models
$P/23_boardset_hpsearch.py siemens $T results/models
$P/35_july_tuning.py $T results/v3 16
python -c "import json;json.dump(json.load(open('results/v3/boardset_cfg_phme_july.json'))['best'],open('results/v5_julyday/cfg_best.json','w'))"

# 2. PHME cross-fits
$P/26_crossfit.py phme $T results/cf 5
$P/27_dayblock.py phme $T results/cf 5
$P/22_boardset.py phme $T results/cf 5 ctx,noctx,single results/models/boardset_cfg_phme.json
$P/29_gbdt_tuned.py phme $T results/cf 3
$P/31_dayblock_models.py phme $T results/cf 3 results/models/boardset_cfg_phme.json two_stage,boardset_ctx
$P/31_dayblock_models.py phme $T results/v5_julyday 3 results/v5_julyday/cfg_best.json boardset_ctx
$P/36_verified_crossfit.py $D $T results/v3 results/models/boardset_cfg_phme.json
$P/40_phme_f1_official_inputs.py $T results/v5/phme_f1_official_inputs.json

# 3. Siemens cross-fits, day-boundary split
export SIEMENS_SPLIT=day
$P/26_crossfit.py siemens $T results/cf_db 3
$P/27_dayblock.py siemens $T results/cf_db 3
$P/34_siemens_groupagg.py $T results/cf_db 3 16
$P/22_boardset.py siemens $T results/cf_db 3 ctx,noctx,single results/models/boardset_cfg_siemens.json
$P/31_dayblock_models.py siemens $T results/cf_db 3 results/models/boardset_cfg_siemens.json two_stage,two_stage_dd,boardset_ctx,boardset_noctx
$P/29_gbdt_tuned.py siemens $T results/cf_db 3
for f in results/cf/phme_*; do ln -sf ../cf/$(basename $f) results/cf_db/; done
unset SIEMENS_SPLIT

# 4. Controls, widths, group rules, fixed-model control, rolling origins, Pfab protocol, latency
for k in randboard randday; do for i0 in 0 10; do
  $P/33_controls_repeat.py $T results/v3 $k $i0 10 9
  $P/37_grouping_sensitivity.py $T results/v4_db $k $i0 10 9
  $P/38_group_rules.py $T results/v4_db $k $i0 10 9
done; done
SIEMENS_SPLIT=day $P/37_grouping_sensitivity.py $T results/v4_db chrono 0 1 16
SIEMENS_SPLIT=day $P/38_group_rules.py $T results/v4_db chrono 0 1 16
$P/39_fixed_model_calib.py $T results/v5/fixed_model_chrono.json chrono 20 8
$P/39_fixed_model_calib.py $T results/v5/fixed_model_randday.json randday 20 8
$P/32_rolling_origin.py $T results/v3 lgb_base,lgb_ctx,two_stage 16
ROLL_GAP=7 $P/32_rolling_origin.py $T results/v5 lgb_base,lgb_ctx 8
$P/28_siemens_controls.py $T results/controls controls 3
$P/28_siemens_controls.py $T results/controls pfab 3
$P/62_latency.py $T results/v5/latency.json        # LAT_TRAIN_THREADS limits training threads on shared machines

# 5. Analyses and supplement tables
$P/48_analysis_v2.py results/cf_db results/controls results/analysis_v5/analysis_v2_db.json
$P/51_analysis_v3.py results/cf_db results/v3 results/analysis_v5/analysis_v3_db.json
$P/54_analysis_v4.py results/cf_db results/v3 results/v4_db $T results/analysis_v5/analysis_v4_db.json
$P/59_board_workload.py results/cf_db results/analysis_v5/board_workload_db.json
$P/57_group_rules_summary.py results/v4_db results/analysis_v5/group_rules_summary_db.json
$P/60_v5_tables.py results/v5 results/v3 results/analysis_v5/v5_summary.json
$P/61_supp_tcpmt_db.py results/analysis_v5 results/analysis_v5/supp
$P/56_supp_tables_v4.py results/analysis_v5/analysis_v4_db.json

# 6. Figures (the last two arguments select the July-configured BoardSet for the PHME temporal panel of Fig. 4)
$P/58_figs_ieee.py results/cf_db results/analysis_v5/analysis_v3_db.json results/analysis_v5/analysis_v4_db.json figures results/cf_db results/v5_julyday
```

## Aggregated results in this repository

| File (`results/main/`) | Used for |
|---|---|
| `analysis_v2_db.json` | Tables I and III (AP, oracle volume), Tables S13–S15, model-difference intervals |
| `analysis_v3_db.json` | realized slip with day-block intervals, Fig. 2, Tables S3, S4, S16 |
| `analysis_v4_db.json`, `partitions/` | Fig. 3, Tables S8, S9, S11 (per-partition grouping widths and group rules) |
| `board_workload_db.json` | board-level measures (Table III, Table S17) |
| `controls_repeat_*.json`, `rolling_origin_*.json` | Table II, Table S7, Table S20 |
| `fixed_model_*.json`, `v5_summary.json` | Tables S19, S20 |
| `latency*.json` | Section VI-D, Table S21 |
| `phme_f1_official_inputs.json` | Section S6 |
| `supplement_tables/` | generated supplement tables |

`results/earlier_round/` holds the files behind Tables S1, S2, S5, S7 and S18 (`analysis_v2d.json`, `analysis_v3b.json`, `phme_ctx_ablation.json`).

## Script index

| Script | Purpose |
|---|---|
| `fcr_eval.py` | calibration rules (expected-slip, high-confidence, mean), board metrics, Siemens split (`siemens_half`) |
| `models.py` | LightGBM settings, grouped folds, cross-fitting, two-stage neighbor model |
| `boardset.py` | BoardSet set transformer |
| `phme_data.py`, `common.py` | PHME loading and features; slip/volume helpers |
| `10_build_tables.py` | build the PHME component table and the Siemens call table (timestamp groups, context features) |
| `22`, `23` | BoardSet cross-fits; BoardSet configuration search |
| `26`, `27`, `29`, `29b`, `31`, `34` | GBDT and two-stage cross-fits; day-blocked GBDT; tuned GBDT; day-blocked two-stage and BoardSet; group-aggregate GBDT |
| `28`, `33` | Siemens control partitions and the Pfab–Rothering protocol; repeated controls |
| `32` | rolling origins (and `ROLL_GAP`) |
| `35`, `36`, `40_phme_f1_official_inputs` | July-only selection; repair-stage labels; challenge-rule F1 |
| `37`, `38`, `39` | width of calibration groups and split-conformal variant; group-level rules; fixed-model calibration control |
| `48`, `50`, `51`, `54`, `59` | analyses (AP, oracle, intervals; dependence statistics; realized slip; grouping sensitivity and model pairs; board measures) |
| `56`, `57`, `60`, `61` | supplement tables |
| `58`, `47`, `62` | figures for the IEEE version; Fig. 1 schematic (earlier version); inference latency |
| others (`00`–`25` not listed, `40_analyze`, `41`–`46`, `49`, `52`, `53`, `55`) | earlier exploratory rounds and an extended manuscript; not needed for the reported results |

## License and citation

Code: MIT License (see `LICENSE`). The data keep their own licences. If you use this code, please cite the paper (see `CITATION.cff`).
