# 🔥 Burning Patterns: NASA Satellite Fire Detections

> *"Every year, wildfires burn millions of acres, displace communities, and release decades of stored carbon back into the atmosphere. NASA's MODIS satellite sees all of it. This project uses that data to predict how powerful a fire will be — before it's contained."*

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Data Cleaning & Preprocessing](#6-data-cleaning--preprocessing)
4. [Exploratory Data Analysis](#7-exploratory-data-analysis)
   - [EDA 1 — Annual Fire Count Comparison](#eda-1--annual-fire-count-comparison)
   - [EDA 2 — Monthly Seasonal Trend Line Chart](#eda-2--monthly-seasonal-trend-line-chart)
   - [EDA 3 — Fire Type Distribution](#eda-3--fire-type-distribution)
   - [EDA 4 — State-Level Choropleth Map](#eda-4--state-level-choropleth-map)
   - [EDA 5 — Fire Density Heatmap](#eda-5--fire-density-heatmap)
   - [EDA 6 — Fire Intensity Over Time](#eda-6--fire-intensity-over-time)
   - [EDA 7 — Fire Brightness vs Detection Confidence](#eda-7--fire-brightness-vs-detection-confidence)
   - [EDA 8 — Wildfire Spatial Distribution](#eda-8--wildfire-spatial-distribution)
5. [Dashboards](#8-dashboards)
6. [Preliminary ML Direction](#9-preliminary-ml-direction)
7. [Project Structure](#10-project-structure)
8. [Requirements](#11-requirements)
9. [References](#12-references)

---

## 1. Project Overview

**Project Title:** Burning Patterns: NASA Satellite Fire Detections

**Main Question:**
> Can we predict how intense a wildfire will be, using satellite-measured features at the moment of detection?

**Goal:** Analyze three years of NASA FIRMS MODIS fire detection data for the United States to uncover temporal trends, geographic hotspots, and fire intensity patterns, and build a regression model that predicts **Fire Radiative Power (FRP)**, the most direct satellite-measurable indicator of wildfire severity.

**Key Facts:**
- 348,070 total fire detection records across 2022, 2023, 2024
- 16 features per record, including spatial, temporal, and physical measurements
- Coverage: Continental United States
- Satellite: MODIS aboard Terra and Aqua (NASA)

---

## 2. Dataset 

**Source:** NASA FIRMS - Fire Information for Resource Management System
**Instrument:** MODIS (Moderate Resolution Imaging Spectroradiometer)
**Provider:** NASA Earth Science Division
**URL:** https://firms.modaps.eosdis.nasa.gov/

### Dataset Size

| Year | Records |
|---|---|
| 2022 | 130,358 |
| 2023 | 94,000 |
| 2024 | 123,712 |
| **Total (merged)** | **348,070 rows × 16 columns** |

### Feature Descriptions

| Feature | Type | Description |
|---|---|---|
| `acq_date` | Date | Date of satellite fire detection |
| `acq_time` | Integer | UTC time of detection (HHMM format) |
| `latitude` | Float | Geographic latitude of fire pixel |
| `longitude` | Float | Geographic longitude of fire pixel |
| `brightness` | Float | Band 21 brightness temperature - measures radiant energy emitted directly by the fire flame (Kelvin) |
| `bright_t31` | Float | Band 31 brightness temperature - captures background/ambient temperature around the fire (Kelvin) |
| `frp` | Float | Fire Radiative Power - total energy released by the fire (Megawatts) - **ML target variable** |
| `confidence` | Integer | Detection reliability estimate, range 0–100. Low < 30, Nominal 30–80, High > 80 |
| `scan` | Float | Along-scan pixel size in km (spatial resolution) |
| `track` | Float | Along-track pixel size in km. Pixel area = scan × track |
| `daynight` | String | D = daytime detection, N = nighttime detection |
| `type` | Integer | 0 = presumed vegetation fire, 1 = active volcano, 2 = other static land source, 3 = offshore |
| `satellite` | String | Satellite platform - T = Terra, A = Aqua |
| `version` | String | MODIS collection version (6.0 / 6.1) |
| `year` | Integer | Added during preprocessing - 2022, 2023, or 2024 |

---

### Environmental Impact Argument

FRP directly measures:
- **Carbon release** - how much stored biomass is being converted back to CO₂ in real time
- **Smoke generation** - high FRP fires produce smoke columns affecting air quality across entire regions
- **Ecosystem loss** - FRP intensity predicts whether a landscape will recover or permanently shift to a degraded state

> Predicting FRP accurately means predicting the severity of all of these outcomes before the fire is even contained. A model trained on this data could give fire agencies earlier, more precise estimates of how destructive an active fire will become - directly informing evacuation decisions, air quality warnings, and resource deployment.

---

Data was downloaded directly from the NASA FIRMS archive:
- **URL:** https://firms.modaps.eosdis.nasa.gov/country/
- **Files downloaded:**
  - `modis_2022_United_States.csv`
  - `modis_2023_United_States.csv`
  - `modis_2024_United_States.csv`

Each annual file was loaded separately and tagged with a `year` column before concatenation. This ensures year-over-year analysis is possible on the merged dataset without ambiguity.

```python
import pandas as pd

# Load each year
df22 = pd.read_csv('modis_2022_United_States.csv')
df23 = pd.read_csv('modis_2023_United_States.csv')
df24 = pd.read_csv('modis_2024_United_States.csv')

# Tag with year
df22['year'] = 2022
df23['year'] = 2023
df24['year'] = 2024

# Merge into single DataFrame
df_all = pd.concat([df22, df23, df24], ignore_index=True)

print(df_all.shape)
# Output: (348070, 16)
```

**Schema validation** — confirmed all three files have identical column structures before merging:

```python
# Verify identical schemas across all 3 years
print(set(df22.columns) == set(df23.columns) == set(df24.columns))
# Output: True
```

**Finding:** No column drift or schema inconsistency was detected across the three annual files. Safe to merge directly.

---

## 3. Data Cleaning & Preprocessing

### Steps Performed

```python
# Step 1 — Standardize column names
df_all.columns = df_all.columns.str.strip().str.lower().str.replace(' ', '_')

# Step 2 — Inspect structure
df_all.info()
df_all.head()

# Step 3 — Check null values (raw count)
df_all.isna().sum()

# Step 4 — Check null values (as percentage)
(df_all.isna().sum() / len(df_all)) * 100

# Step 5 — Verify data types
df_all.dtypes

# Step 6 — Inspect confidence column format
df_all['confidence'].unique()

# Step 7 — Verify year distribution
df_all['year'].value_counts()

# Step 8 — Export merged dataset
df_all.to_csv('NASA_FIRMS_2022-24.csv', index=False)
```
---

**Total visualizations:** 8

### EDA 1 — Annual Fire Count Comparison

**Type:** Bar chart
**Tool:** Plotly Dash (Dashboard 1)
**Purpose:** Answers "How is wildfire activity changing over time?"

**Insight:**
2022 recorded the highest detection count at **130,358**, followed by 2024 at **123,712** — a near-recovery after 2023's significant drop to **94,000** (down 28% year-over-year). The 2023 dip likely reflects cooler, wetter conditions or reduced drought severity rather than a genuine long-term decline. The strong 2024 rebound confirms that wildfire frequency remains elevated and volatile — not trending downward.

**ML Connection:**
Year-level variation confirms that `year` must be included as an independent feature in the model. The model cannot treat all years as interchangeable.

---

### EDA 2 — Monthly Seasonal Trend Line Chart

**Type:** Multi-line chart with 3-year average
**Tool:** Plotly Dash (Dashboard 1)
**Purpose:** Shows seasonality and climate-driven patterns

**Insight:**
Seasonality is unstable across years. 2022 peaks sharply in **June** with a strong secondary peak in September (~18,300). 2023 remains relatively flat with a modest August peak. 2024 shows an extreme single spike in **July**. Overall, detections concentrate in **summer (June–September)** with no consistent spring secondary peak.

The high year-to-year variability suggests that beyond seasonal rhythm, **annual climate conditions (e.g. drought severity) drive detection counts just as strongly as month alone**. Both `year` and `month` should be included as features in ML modeling, and models should be tested for cross-year generalization to avoid distribution shift.

**ML Connection:**
Month is confirmed as a meaningful predictor. Irregular seasonal shapes across years also suggest an interaction term `month × year` may improve model performance.

---

### EDA 3 — Fire Type Distribution

**Type:** Donut chart
**Tool:** Plotly Dash (Dashboard 1)
**Purpose:** Shows the composition of fire detection types

**Insight:**
The distribution is overwhelmingly dominated by **Type 0 — Vegetation fire at 94.2%** (327,899 detections). The remaining three types collectively account for less than 6% of all records.

This extreme imbalance means a naive classifier could achieve over 94% accuracy by simply predicting Type 0 for every observation — a meaningless result. For any ML classification task using `type` as a target, **SMOTE (Synthetic Minority Over-sampling Technique)** or `class_weight="balanced"` must be applied to prevent the model from ignoring minority classes entirely.

The derived feature `brightness − bright_t31` (thermal contrast) is a strong candidate for distinguishing volcano detections (Type 1) from vegetation fires (Type 0), since volcanic heat sources maintain elevated background temperatures, resulting in a smaller contrast ratio compared to wildfire hotspots.

**ML Connection:**
The `type` column is dropped from the FRP regression feature set due to near-zero variance. However, the class imbalance finding is important context for any future classification extension of this project.

---

### EDA 4 — State-Level Choropleth Map

**Type:** Choropleth map (fire count by state)
**Tool:** Plotly
**Purpose:** Identifies geographic hotspots across the United States

**Insight:**
Wildfire activity is unevenly distributed across the United States with clear clustering in the **southern and western regions**. Significant hotspots are observed in **Texas, California, and the southeastern states**. Areas with higher FRP values indicate more intense fires, suggesting these regions are more vulnerable due to environmental and climatic conditions — dry heat in the west, agricultural burns in the southeast.

**ML Connection:**
Geographic location carries independent predictive signal for FRP. Western US fires (California, Pacific Northwest) are both more frequent and more intense — `latitude` and `longitude` are meaningful features for the regression model.

---

### EDA 5 — Fire Density Heatmap (Latitude × Longitude)

**Type:** Spatial density heatmap
**Tool:** Plotly
**Purpose:** Reveals spatial clustering and natural fire zones

**Insight:**
Wildfire occurrences are **highly concentrated in specific geographic regions** rather than evenly distributed. The highest density is observed in the **southern United States, particularly around Texas and nearby regions**. Additional clusters appear in **California and parts of the southeastern US**, indicating recurring fire-prone zones driven by climate, vegetation type, and land use patterns.

**ML Connection:**
Spatial clustering across all three years confirms that `latitude` and `longitude` are stable, meaningful features. High-density regions also correlate with higher FRP values — geographic location is a proxy for fire regime type.

---

### EDA 6 — Fire Intensity Over Time (FRP by Month × Year)

**Type:** Line chart / time series of FRP
**Tool:** Plotly
**Purpose:** Shows how fire intensity (FRP) changes across time

**Insight:**
Fire intensity varies across days and months, showing noticeable fluctuations in FRP values. Peaks in FRP indicate days or periods with significantly higher wildfire intensity. These short-term and seasonal trends highlight how wildfire severity can fluctuate dramatically over time, reinforcing that temporal features are essential for FRP prediction.

> **Note for final presentation:** This chart should be updated to show FRP aggregated by **month × year** (boxplot or multi-line) rather than individual daily dates. The current version showing only March 2026 dates indicates a date filter was left active during export.

**ML Connection:**
Seasonal FRP variation confirms that `month` carries signal not just for fire count but for fire intensity. The `year` feature captures inter-annual differences in baseline intensity levels.

---

### EDA 7 — Fire Brightness vs Detection Confidence

**Type:** Density scatter plot
**Tool:** Tableau (Dashboard 2)
**Purpose:** Evaluates detection reliability and its relationship to fire intensity

**Insight:**
A clear **positive relationship** exists between fire brightness and detection confidence. Most wildfire detections cluster between brightness values of approximately **310–380 Kelvin** and confidence levels above **70**. Stronger thermal signals produce more reliable satellite detections. Low-confidence detections (below 50) are associated with lower brightness values, suggesting marginal fires near the sensor's detection threshold.

**ML Connection:**
`brightness` and `confidence` are positively correlated. Both are valid predictors of FRP, but **multicollinearity between them must be checked** before including both in the regression model:

```python
df[['brightness', 'bright_t31', 'confidence', 'frp', 'scan', 'track']].corr()
```

If correlation between two features exceeds 0.85, the weaker predictor should be dropped.

---

### EDA 8 — Wildfire Spatial Distribution

**Type:** Density map (geographic heatmap)
**Tool:** Tableau (Dashboard 2)
**Purpose:** Geographic distribution of all wildfire detections 2022–2024

**Insight:**
Wildfires are **strongly clustered in specific regions** rather than evenly spread across the US. Major hotspots appear in:
- **California** — dry climate, forested landscapes, Santa Ana wind events
- **Pacific Northwest** — dense coniferous forests, summer drought
- **Texas** — largest state by land area, high vegetation fire frequency
- **Southeastern United States** — agricultural burns, high humidity but dense vegetation

These areas are known for dry climates, dense vegetation, and seasonal fire activity — all conditions that drive both detection frequency and FRP intensity.

**ML Connection:**
The geographic clustering visible in this map directly justifies including `latitude` and `longitude` as features. The spatial patterns are consistent across all three years, making location a stable and reliable predictor.

---

## 5. Dashboards

### Dashboard 1 — Plotly Dash

**Tool:** Plotly / Dash (Python)
**Charts included:** EDA 1, EDA 2, EDA 3

**Features:**
- Interactive **year filter** (checklist: 2022, 2023, 2024)
- Interactive **fire type filter** (multi-select dropdown)
- **KPI cards:** Total detections, Avg FRP (MW), Avg Confidence (%), Night detections (%)
- All charts update dynamically based on filter selection
- Single-callback architecture for performance

**Run locally:**
```bash
pip install dash dash-bootstrap-components plotly pandas
python app.py
# Visit: http://127.0.0.1:8050
```

---

### Dashboard 2 — Tableau

**Tool:** Tableau Public
**Charts included:** EDA 7 (Brightness vs Confidence) + EDA 8 (Wildfire Spatial Distribution)

**Features:**
- Brightness vs Detection Confidence density scatter with year slider
- Wildfire Density Map (Latitude × Longitude) with year range filter
- Combined dashboard view: "Spatial Distribution and Detection Confidence of Wildfires in the United States"

**Key findings from dashboard:**
- Fire brightness and detection confidence show a positive relationship
- Wildfire activity is geographically clustered rather than evenly distributed
- Both geographic location and thermal intensity are important predictors of wildfire activity

---

## 6. Preliminary ML Direction

### One-Line Summary

> *"Our data doesn't just tell us where fires happen — it tells us how powerful they are. Predicting that power is our machine learning goal."*

---

### Proposed Task: Supervised Regression

**Target variable:** `frp` — Fire Radiative Power (continuous, Megawatts)
**Transformed target:** `frp_log = np.log1p(frp)` — applied before training due to right-skewed distribution

---

### Why Regression — Not Classification or Clustering

| Task | Decision | Reasoning |
|---|---|---|
| **Regression** | ✅ Selected | FRP is continuous — its exact value carries operational meaning. The difference between 400 MW and 800 MW is the difference between a manageable incident and a regional emergency |
| Classification | ❌ Ruled out | Binning FRP into low/medium/high discards the most valuable information in the dataset |
| Clustering | ❌ Ruled out | Clustering is unsupervised and finds hidden groups. We already have a defined target variable and a predictive question |

---

### Model Pipeline

| Stage | Model | Purpose |
|---|---|---|
| 1 | Linear Regression | Interpretable baseline — every subsequent model must beat this |
| 2 | Random Forest Regressor | Handles non-linearity, provides feature importance scores |
| 3 | XGBoost / LightGBM | Expected best performer — gradient boosting handles skewed targets and outliers well on tabular data |

## 7. Project Structure

```
NASA-FIRMS-Analysis/
│
├── data/
│   ├── modis_2022_United_States.csv      
│   ├── modis_2023_United_States.csv      
│   ├── modis_2024_United_States.csv      
│   └── NASA_FIRMS_2022-24.csv                      
│
├── dashboards/
│   ├── plotly_visuals.py                                                         
│
└── README.md                            
```

---

## 8. Requirements

```bash
pip install pandas numpy matplotlib seaborn plotly dash dash-bootstrap-components
```

| Library | Version | Use |
|---|---|---|
| `pandas` | 1.5+ | Data loading, merging, cleaning |
| `numpy` | 1.23+ | Numerical operations, log transform |
| `plotly` | 5.0+ | Interactive visualizations |
| `dash` | 2.0+ | Dashboard 1 framework |
| `dash-bootstrap-components` | 1.0+ | Dashboard layout |
| `matplotlib` / `seaborn` | Any | Static EDA charts |
| Tableau Public | Desktop | Dashboard 2 |

---

## 9. References

| Resource | URL |
|---|---|
| NASA FIRMS Data Download | https://firms.modaps.eosdis.nasa.gov/country/ |
| MODIS Active Fire Documentation | https://modis-fire.umd.edu/ |
| MODIS Collection 6 User Guide | https://earthdata.nasa.gov/ |
| Plotly Dash Documentation | https://dash.plotly.com/ |
| Tableau Public | https://public.tableau.com/ |
| 2023 Canadian Wildfire Season | https://cwfis.cfs.nrcan.gc.ca/ |
*DATA 230 — San José State University — College of Information, Data and Society*
*Group 6 | Sanjana Thummalapalli · Elina Yin · Mansi Verma · Xuanhua (Carol) Li*
