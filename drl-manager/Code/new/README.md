# `Code/new` utilities

## `preprocess_single_turbine.py`

Converts a single turbine CSV (legacy split format with `Tmstamp` + `Patv`) into a **regular 10-minute** time series CSV suitable for **TimeCAP/ETT-style loaders**.

It implements a common wind-power preprocessing pipeline:
- **Range constraints** (e.g., negative power → 0; wind-direction/pitch ranges)
- **IQR outlier removal** (optionally only remove *adjacent* outliers)
- **Linear interpolation** for missing values (with configurable max-gap)
- Optional **Min-Max normalization** (some vars to `[-1, 1]`, others to `[0, 1]`)

### Usage

```bash
python3 preprocess_single_turbine.py \
  --input /home/joshua/rl-cloudsimplus-greenscheduling/cloudsimplus-gateway/src/main/resources/windProduction/split/Turbine_9_2021.csv \
  --output /home/joshua/rl-cloudsimplus-greenscheduling/cloudsimplus-gateway/src/main/resources/windProduction/preprocessed/Turbine_9_2021.csv
```

### Output columns (ETT-like)

- `date` + 12 features + target:
  - `Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Prtv,T2m,Sp,RelH,Wspd_w,Wdir_w,OT`
- `OT` is the target and equals input `Patv`.

### Common options

```bash
# Stronger cleaning:
python3 preprocess_single_turbine.py \
  --input ... \
  --output ... \
  --outlier-mode adjacent \
  --interp-max-gap 12 \
  --normalize minmax \
  --scaler-output /tmp/turbine_9_2021_scaler.json
```

### Notes

- The script prints a summary with counts for duplicates, created missing rows, outliers removed, and how many values were interpolated/filled.

## `merge_ett_csvs.py`

Merges multiple **ETT-style** CSVs (must have the same header and a `date` column) into one file:
- sorts by time
- deduplicates by `date` (keeps the row from the later file if both contain the same timestamp)

Example (single turbine across years):

```bash
python3 merge_ett_csvs.py \
  --inputs \
    /home/joshua/rl-cloudsimplus-greenscheduling/cloudsimplus-gateway/src/main/resources/windProduction/preprocessed/Turbine_9_2020_ett.csv \
    /home/joshua/rl-cloudsimplus-greenscheduling/cloudsimplus-gateway/src/main/resources/windProduction/preprocessed/Turbine_9_2021_ett.csv \
  --output \
    /home/joshua/rl-cloudsimplus-greenscheduling/cloudsimplus-gateway/src/main/resources/windProduction/preprocessed/Turbine_9_2020_2021_ett.csv
```

## `preprocess_all_turbines_2021.py`

Batch preprocess **all 2021** split turbine files into ETT-style CSVs.

- Input pattern: `split/Turbine_<ID>_2021.csv`
- Output:
  - `preprocessed/Turbine_<ID>_2021_ett.csv`
  - `preprocessed/Turbine_<ID>_2021_report.json`
  - `preprocessed/preprocess_2021_summary.json`

### Usage

```bash
# From repo root, using the drl-manager venv:
/home/joshua/rl-cloudsimplus-greenscheduling/drl-manager/.venv/bin/python \
  /home/joshua/rl-cloudsimplus-greenscheduling/drl-manager/Code/new/preprocess_all_turbines_2021.py \
  --year 2021 \
  --jobs 1
```

### Common options

```bash
# Faster/stronger cleaning and write per-turbine scaler JSONs
/home/joshua/rl-cloudsimplus-greenscheduling/drl-manager/.venv/bin/python \
  /home/joshua/rl-cloudsimplus-greenscheduling/drl-manager/Code/new/preprocess_all_turbines_2021.py \
  --year 2021 \
  --outlier-mode adjacent \
  --interp-max-gap 12 \
  --normalize minmax \
  --write-scalers \
  --jobs 4
```


