# 🔥 FireScope: Wildfire Detection & Intensity Analysis (2022–2024)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Data](https://img.shields.io/badge/Data-NASA%20FIRMS%20MODIS-orange)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

## 📌 Project Overview

This project analyzes NASA FIRMS MODIS satellite fire detection data for the
United States across three years (2022–2024) to uncover temporal trends,
geographic hotspots, and fire intensity patterns. The ultimate goal is to build
a regression model to predict Fire Radiative Power (FRP) — a key indicator of
wildfire severity.

---

## 📂 Dataset

| Property | Detail |
|---|---|
| Source | [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) |
| Satellite | MODIS Terra & Aqua |
| Coverage | United States, 2022–2024 |
| Raw Size | ~120,000 records |
| Cleaned Size | ~114,000 records |
| Format | CSV (one file per year, merged) |

**Key Features:**

| Column | Description |
|---|---|
| `acq_date` | Date of satellite detection |
| `latitude` / `longitude` | Fire pixel coordinates |
| `brightness` | Brightness temperature (Kelvin) |
| `frp` | Fire Radiative Power (Megawatts) |
| `confidence` | Detection reliability (0–100 integer) |
| `scan` / `track` | Pixel spatial resolution |
| `daynight` | D = daytime, N = nighttime detection |
| `year` | Added during preprocessing (2022/2023/2024) |

---

## 🧹 Data Preprocessing
**NASA FIRMS MODIS — United States (2022, 2023, 2024)**


## 1. Loading & Merging

Each year was loaded as a separate CSV and tagged with a `year` column before merging into a single unified DataFrame.
```python
df22 = pd.read_csv('modis_2022_United_States.csv')
df23 = pd.read_csv('modis_2023_United_States.csv')
df24 = pd.read_csv('modis_2024_United_States.csv')

df22['year'] = 2022
df23['year'] = 2023
df24['year'] = 2024

df_all = pd.concat([df22, df23, df24], ignore_index=True)
```


## 2. Column Standardization

All column names were stripped of whitespace, lowercased, and spaces replaced with underscores to ensure consistent access across the merged dataset.
```python
df_all.columns = df_all.columns.str.strip().str.lower().str.replace(' ', '_')
```


## 3. Schema Validation Across Years

Confirmed that all three annual files share identical column names — no column drift or schema inconsistency across years.
```python
set(df22.columns) == set(df23.columns) == set(df24.columns)  # True
```


## 4. Null Value Check

Checked for missing values across all columns both as raw counts and as a percentage of total rows.
```python
df_all.isna().sum()
(df_all.isna().sum() / len(df_all)) * 100
```


## 5. Data Type Verification

Confirmed all columns are correctly typed — numeric columns are float/int, date column is object (to be parsed if needed for time-series analysis).
```python
df_all.dtypes
```


## 6. Confidence Column Inspection

The `confidence` column in the US MODIS dataset stores integer values (0–100), unlike the global version which uses categorical strings. This was verified and noted for binning during the analysis phase.
```python
df_all['confidence'].unique()
```


## 7. Year Distribution Check

Verified that all three years are represented in the merged dataset with reasonable record counts.

```python
df_all['year'].value_counts()
```


## 📎 References

- NASA FIRMS: https://firms.modaps.eosdis.nasa.gov/
- MODIS Active Fire Documentation: https://modis-fire.umd.edu/
- 2023 Canadian Wildfire Season: https://cwfis.cfs.nrcan.gc.ca/
