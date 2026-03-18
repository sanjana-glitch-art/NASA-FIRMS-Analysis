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

Checked for missing values across all columns as a raw count of total rows.
```python
df_all.isna().sum()
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

# 📊 EDA Visualizations — NASA FIRMS US (2022–2024)

**Dataset:** NASA FIRMS MODIS — United States  
**Tool:** Plotly (Charts 1, 2, 3, 7) + Tableau (Charts 5, 6) + Plotly/Dash Dashboard  

---

## Chart 1 — Annual Fire Count Comparison (2022 vs 2023 vs 2024)

**Type:** Bar chart  
**Tool:** Plotly (Dash dashboard)  
**Purpose:** Answers "How is wildfire activity changing over time?"

**Insight:**  
2022 recorded the highest detection count at 130,358, followed by 2024 at 123,712 — a
near-recovery after 2023's significant drop to 94,000 (down 28% year-over-year). The 2023
dip likely reflects cooler, wetter conditions or reduced drought severity rather than a genuine
long-term decline, as 2024 rebounded strongly. This year-over-year volatility confirms that
both `year` and climate-driven features must be included in ML modeling.

---

## Chart 2 — Monthly / Seasonal Trend Line Chart

**Type:** Multi-line chart with 3-year average  
**Tool:** Plotly (Dash dashboard)  
**Purpose:** Shows seasonality and climate-driven patterns

**Insight:**  
Seasonality pattern is unstable across years. 2022 peaks sharply in June with a strong
secondary peak in September (~18,300). 2023 remains relatively flat with a modest August
peak. 2024 shows an extreme single spike in July. Overall, fire detections concentrate in
**summer (June–September)** with no consistent spring secondary peak. The high
year-to-year variability suggests that beyond seasonal rhythm, annual climate conditions
(e.g. drought severity) drive detection counts just as strongly as month alone. Both `year`
and `month` should be included as features in ML modeling, and models should be tested
for cross-year generalization to avoid distribution shift.

---

## Chart 3 — Fire Type Distribution

**Type:** Donut chart  
**Tool:** Plotly (Dash dashboard)  
**Purpose:** Shows composition of fire detection types in the dataset

**Insight:**  
Type 0 (presumed vegetation fire) dominates at ~95%+ of all detections. Volcanic,
static land, and offshore detections are negligible in the US dataset. This confirms
the dataset is almost entirely composed of vegetation fires, validating its use for
wildfire intensity analysis. The small fraction of non-vegetation detections were
retained in the dataset but their influence on FRP modeling will be minimal.

---

## Chart 4 — State-Level Choropleth Map (Fire Counts)

**Type:** Choropleth map  
**Tool:** Plotly  
**Purpose:** Identifies geographic hotspots across the United States

**Insight:**  
Wildfire detections show strong spatial clustering in the **western and southeastern
United States**. California and the Pacific Northwest exhibit high densities due to
dry climates and forested landscapes. The southeastern US shows frequent fire activity
likely related to agricultural burns and seasonal vegetation fires. The central US
has comparatively low fire detection density.

---

## Chart 5 — Brightness vs. Confidence Scatter Plot

**Type:** Density scatter plot  
**Tool:** Tableau  
**Purpose:** Evaluates detection reliability and intensity correlation

**Insight:**  
A clear positive relationship exists between fire brightness and detection confidence.
Most wildfire detections cluster between brightness values of approximately **310–380
Kelvin** with confidence levels above 70. Stronger thermal signals produce more reliable
satellite detections. Low-confidence detections (below 50) are associated with lower
brightness values, suggesting marginal fires near the sensor's detection threshold.

---

## Chart 6 — Heatmap of Fire Density (Latitude × Longitude)

**Type:** Spatial density heatmap  
**Tool:** Tableau  
**Purpose:** Reveals spatial clustering and natural fire zones

**Insight:**  
Wildfire detections show strong spatial clustering in the western and southeastern
United States. The highest density hotspots appear along the California coast,
Pacific Northwest, and the southeastern corridor. Sparse detections in the
central plains and northeast confirm that fire risk is geographically concentrated
rather than evenly distributed across the country.

---

## Chart 7 — Fire Intensity Over Time (FRP by Month × Year)

**Type:** Boxplot grouped by month and year  
**Tool:** Plotly  
**Purpose:** Shows how fire intensity (FRP) changes seasonally and year-to-year

**Insight:**  
FRP peaks sharply in **July–August** across all three years, confirming summer as the
high-intensity fire season. 2022 shows the widest IQR in July, indicating greater
spread in fire intensity that year. The presence of extreme outliers (FRP > 500 MW)
in summer months across all years confirms the right-skewed distribution — log
transformation will be necessary before feeding FRP into the regression model.
Year-to-year differences in median FRP are visible even within the same month,
reinforcing that `year` carries independent predictive signal beyond seasonality alone.

---

## Dashboard

**Dashboard 1 — Plotly Dash**  
File: `app_charts123.py`  
Charts included: 1, 2, 3 with interactive year and fire type filters + KPI cards  
Tool: Plotly / Dash (Python)

**Dashboard 2 — Tableau**  
Charts included: 5, 6 (Brightness vs Confidence + Fire Density Heatmap)  
Tool: Tableau Public  

## 📎 References

- NASA FIRMS: https://firms.modaps.eosdis.nasa.gov/
- MODIS Active Fire Documentation: https://modis-fire.umd.edu/
