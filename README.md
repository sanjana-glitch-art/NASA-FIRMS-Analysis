# 🔥 Burning Patterns: NASA Satellite Fire Detections
### DATA 230 — Mid-Presentation | Group 6 | San José State University

> *"Every year, wildfires burn millions of acres, displace communities, and release decades of stored carbon back into the atmosphere. NASA's MODIS satellite sees all of it. This project uses that data to predict how powerful a fire will be — before it's contained."*

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Introduction](#2-dataset-introduction)
3. [Dataset Motivation](#3-dataset-motivation)
4. [Data Collection](#4-data-collection)
5. [Data Merging](#5-data-merging)
6. [Data Cleaning & Preprocessing](#6-data-cleaning--preprocessing)
7. [Exploratory Data Analysis](#7-exploratory-data-analysis)
   - [EDA 1 — Annual Fire Count Comparison](#eda-1--annual-fire-count-comparison)
   - [EDA 2 — Monthly Seasonal Trend Line Chart](#eda-2--monthly-seasonal-trend-line-chart)
   - [EDA 3 — Fire Type Distribution](#eda-3--fire-type-distribution)
   - [EDA 4 — State-Level Choropleth Map](#eda-4--state-level-choropleth-map)
   - [EDA 5 — Fire Density Heatmap](#eda-5--fire-density-heatmap)
   - [EDA 6 — Fire Intensity Over Time](#eda-6--fire-intensity-over-time)
   - [EDA 7 — Fire Brightness vs Detection Confidence](#eda-7--fire-brightness-vs-detection-confidence)
   - [EDA 8 — Wildfire Spatial Distribution](#eda-8--wildfire-spatial-distribution)
8. [Dashboards](#8-dashboards)
9. [Preliminary ML Direction](#9-preliminary-ml-direction)
10. [Project Structure](#10-project-structure)
11. [Requirements](#11-requirements)
12. [References](#12-references)

---

## 1. Project Overview

**Project Title:** Burning Patterns: NASA Satellite Fire Detections

**Course:** DATA 230 — Data Analytics and Visualization

**Main Question:**
> Can we predict how intense a wildfire will be, using satellite-measured features at the moment of detection?

**Goal:** Analyze three years of NASA FIRMS MODIS fire detection data for the United States to uncover temporal trends, geographic hotspots, and fire intensity patterns — and build a regression model that predicts **Fire Radiative Power (FRP)**, the most direct satellite-measurable indicator of wildfire severity.

**Key Facts:**
- 348,070 total fire detection records across 2022, 2023, 2024
- 16 features per record including spatial, temporal, and physical measurements
- Coverage: Continental United States
- Satellite: MODIS aboard Terra and Aqua (NASA)

---

## 2. Dataset Introduction

**Source:** NASA FIRMS — Fire Information for Resource Management System
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
| `brightness` | Float | Band 21 brightness temperature — measures radiant energy emitted directly by the fire flame (Kelvin) |
| `bright_t31` | Float | Band 31 brightness temperature — captures background/ambient temperature around the fire (Kelvin) |
| `frp` | Float | Fire Radiative Power — total energy released by the fire (Megawatts) — **ML target variable** |
| `confidence` | Integer | Detection reliability estimate, range 0–100. Low < 30, Nominal 30–80, High > 80 |
| `scan` | Float | Along-scan pixel size in km (spatial resolution) |
| `track` | Float | Along-track pixel size in km. Pixel area = scan × track |
| `daynight` | String | D = daytime detection, N = nighttime detection |
| `type` | Integer | 0 = presumed vegetation fire, 1 = active volcano, 2 = other static land source, 3 = offshore |
| `satellite` | String | Satellite platform — T = Terra, A = Aqua |
| `version` | String | MODIS collection version (6.0 / 6.1) |
| `year` | Integer | Added during preprocessing — 2022, 2023, or 2024 |

### Important Feature Notes

- **`brightness` vs `bright_t31`:** Band 21 captures the fire flame temperature directly. Band 31 captures the surrounding background. The difference `brightness - bright_t31` (thermal contrast) is a derived feature useful for distinguishing fire types — volcanic sources maintain elevated background temperatures resulting in smaller contrast ratios compared to vegetation fires.
- **`frp`:** Not an estimate. It is a direct physical measurement of radiative energy from the satellite sensor. This makes it the most reliable continuous target for intensity prediction.
- **`confidence`:** In the US MODIS dataset, this is stored as an integer (0–100), not categorical strings as seen in the global version. This was verified during preprocessing.

---

## 3. Dataset Motivation

### The Scale of the Problem

Wildfires are no longer rare disasters — they are becoming the new normal. Between 2022 and 2024, the United States recorded over **348,000 fire detection events**. Climate scientists now classify wildfire as a **primary feedback loop**: fires release CO₂, which warms the planet, which creates drier conditions, which creates more fires.

Key real-world events during the dataset period:
- **2023:** Canada experienced its largest wildfire season ever recorded — over 18 million hectares burned
- **2022:** The US saw its highest detection count in this dataset at 130,358 events
- **2024:** Strong rebound to 123,712 events after 2023's climate-driven dip, signaling no long-term decline

### Why NASA FIRMS MODIS Specifically?

- **No blind spots** — MODIS orbits the entire continental US twice daily, detecting fires in locations no ground crew ever reaches, including remote forests, mountains, and uninhabited regions at 3am
- **Operationally used** — CAL FIRE, the USDA Forest Service, and FEMA all use FIRMS data for real-time fire response and resource allocation. Our analysis is built on the same foundation
- **Three physical dimensions simultaneously** — location (lat/lon), time (date/hour), and energy (FRP + brightness) in a single dataset
- **Ground truth intensity** — FRP is not modeled or estimated. It is a direct physical measurement from the satellite sensor itself

### Why It Fits Our ML Goal

| What the ML Model Needs | What the Dataset Provides |
|---|---|
| A continuous, meaningful target variable | FRP in Megawatts — directly interpretable and operationally significant |
| Strong predictor features | Brightness, scan/track geometry, confidence score |
| Temporal variation | 3 full years — captures seasonal and inter-annual patterns |
| Spatial variation | Full continental US — diverse fire regimes from California to Florida |
| Sufficient scale | 348,070 records — large enough for robust ML training and testing |

### Environmental Impact Argument

FRP directly measures:
- **Carbon release** — how much stored biomass is being converted back to CO₂ in real time
- **Smoke generation** — high FRP fires produce smoke columns affecting air quality across entire regions
- **Ecosystem loss** — FRP intensity predicts whether a landscape will recover or permanently shift to a degraded state

> Predicting FRP accurately means predicting the severity of all of these outcomes before the fire is even contained. A model trained on this data could give fire agencies earlier, more precise estimates of how destructive an active fire will become — directly informing evacuation decisions, air quality warnings, and resource deployment.

---

## 4. Data Collection

Data was downloaded directly from the NASA FIRMS archive:
- **URL:** https://firms.modaps.eosdis.nasa.gov/country/
- **Instrument:** MODIS — Collection 6 / 6.1
- **Region:** United States
- **Format:** CSV, one file per calendar year
- **Files downloaded:**
  - `modis_2022_United_States.csv`
  - `modis_2023_United_States.csv`
  - `modis_2024_United_States.csv`

Each file contains all fire pixels detected by the MODIS Terra and Aqua satellites over the continental United States for that calendar year, at the satellite's native detection resolution.

---

## 5. Data Merging

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

## 6. Data Cleaning & Preprocessing

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

### Summary of Findings

| Step | Action | Finding |
|---|---|---|
| Column standardization | Strip, lowercase, underscore | Clean — consistent across all years |
| Schema validation | `set(df.columns)` comparison | Identical — no column drift |
| Null check | `isna().sum()` | Minimal to zero missing values |
| Null percentage | `isna().sum() / len(df)` | No column exceeds meaningful threshold |
| Data types | `.dtypes` | All columns correctly typed |
| Confidence format | `.unique()` | Integer 0–100 (US MODIS format, not categorical strings) |
| Year distribution | `.value_counts()` | All 3 years present |

### Key Finding on Complexity

The dataset required **no major cleaning or imputation**. Its complexity lies in:

- **Scale** — 348,070 records with simultaneous temporal, spatial, and physical dimensions
- **FRP distribution** — heavily right-skewed with extreme outlier values (megafires > 500 MW). Log transformation will be required before ML modeling: `df['frp_log'] = np.log1p(df['frp'])`
- **Confidence as integer** — US MODIS format stores confidence as 0–100, not categorical. Will be binned for categorical analysis: Low (0–30), Nominal (31–80), High (81–100)
- **Class imbalance in fire types** — 94.2% of records are Type 0 (vegetation). This imbalance has direct implications for any classification extension of this work
- **Spatial density** — lat/lon granularity creates clustering challenges for state-level aggregation

---

## 7. Exploratory Data Analysis

**Total visualizations:** 8
**Tools used:** Plotly / Plotly Dash (Charts 1–6), Tableau (Charts 7–8)
**Main question all charts connect to:** *How are wildfire patterns changing over time, and what features best predict fire intensity (FRP)?*

---

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

## 8. Dashboards

### Dashboard 1 — Plotly Dash

**File:** `app_charts123.py`
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
python app_charts123.py
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

### Tool Diversity

| Dashboard | Tool | Grade Tier |
|---|---|---|
| Dashboard 1 | Plotly Dash (Python) | — |
| Dashboard 2 | Tableau Public | — |
| **Combined** | **Two different tools** | ✅ **Excellent** |

---

## 9. Preliminary ML Direction

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

### EDA-to-Feature Justification

Every feature included in the model traces back to a specific EDA finding:

| EDA Finding | Feature Decision |
|---|---|
| FRP is heavily right-skewed with extreme outliers (EDA 6) | Log-transform target: `np.log1p(frp)` |
| Brightness strongly correlates with FRP (EDA 7) | `brightness` = primary predictor |
| Seasonality unstable — year and month both matter (EDA 1, 2) | Include both `month` AND `year` as features |
| Western US fires consistently more intense (EDA 4, 5, 8) | `latitude`, `longitude` carry spatial signal |
| Type column is 94.2% vegetation (EDA 3) | Drop `type` — near-zero variance |
| Brightness and confidence are correlated (EDA 7) | Check multicollinearity before including both |

---

### Feature Set

```python
features = [
    'brightness',    # Primary thermal signal — direct physical correlation with FRP
    'bright_t31',    # Background temperature — helps compute thermal contrast
    'confidence',    # Detection quality — subject to multicollinearity check vs brightness
    'scan',          # Pixel width — affects how much area the sensor captures
    'track',         # Pixel height — same reason as scan
    'month',         # Seasonal fire patterns
    'year',          # Inter-annual climate variation
    'latitude',      # Spatial fire intensity signal
    'longitude',     # Geographic fire regime
    'daynight',      # Night fires behave differently (less wind, humidity effects)
]

target = 'frp_log'  # np.log1p(df['frp'])
```

**Dropped:** `type` — 94.2% single class, near-zero variance, no predictive value

---

### Model Pipeline

| Stage | Model | Purpose |
|---|---|---|
| 1 | Linear Regression | Interpretable baseline — every subsequent model must beat this |
| 2 | Random Forest Regressor | Handles non-linearity, provides feature importance scores |
| 3 | XGBoost / LightGBM | Expected best performer — gradient boosting handles skewed targets and outliers well on tabular data |

---

### Train / Test Split

```python
import numpy as np

# Log-transform target
df['frp_log'] = np.log1p(df['frp'])

# Time-based split — simulates real deployment
# Train on historical data, test on most recent year
train = df[df['year'].isin([2022, 2023])]
test  = df[df['year'] == 2024]

X_train = train[features]
y_train = train['frp_log']
X_test  = test[features]
y_test  = test['frp_log']
```

> **Why time-based and not random?** A random split leaks future fire data into training. Training on 2022–2023 and testing on 2024 simulates predicting next season's fires from historical data — a much more honest evaluation of model generalization.

---

### Evaluation Metrics

| Metric | What It Measures | Why It Matters |
|---|---|---|
| **RMSE** | Root Mean Squared Error — average error in log(MW) | Penalizes large prediction errors heavily |
| **MAE** | Mean Absolute Error — average absolute error | More interpretable, less sensitive to outliers |
| **R²** | Proportion of FRP variance explained | Overall model fit — pair with RMSE for full picture |

---

### Anticipated Challenges

| Challenge | Mitigation |
|---|---|
| Right-skewed FRP | Log transformation before training |
| Megafire outliers (FRP > 500 MW) | Evaluate model performance separately on extreme events |
| Spatial autocorrelation | Time-based train/test split partially mitigates inflation |
| Multicollinearity (brightness vs confidence) | Correlation matrix check before finalizing feature set |
| Cross-year generalization | Train on 2022–2023, test on 2024 — tests true out-of-distribution performance |

---

## 10. Project Structure

```
burning-patterns/
│
├── data/
│   ├── modis_2022_United_States.csv      # Raw download from NASA FIRMS
│   ├── modis_2023_United_States.csv      # Raw download from NASA FIRMS
│   ├── modis_2024_United_States.csv      # Raw download from NASA FIRMS
│   └── NASA_FIRMS_2022-24.csv            # Merged + cleaned dataset
│
├── notebooks/
│   └── eda_analysis.ipynb                # Full EDA notebook (Google Colab)
│
├── dashboards/
│   ├── app_charts123.py                  # Dashboard 1 — Plotly Dash
│   └── tableau/                          # Dashboard 2 — Tableau workbook
│
├── docs/
│   ├── preprocessing.md                  # Preprocessing steps detail
│   └── visualizations.md                 # All 8 charts with full insights
│
└── README.md                             # This file
```

---

## 11. Requirements

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

## 12. References

| Resource | URL |
|---|---|
| NASA FIRMS Data Download | https://firms.modaps.eosdis.nasa.gov/country/ |
| MODIS Active Fire Documentation | https://modis-fire.umd.edu/ |
| MODIS Collection 6 User Guide | https://earthdata.nasa.gov/ |
| Plotly Dash Documentation | https://dash.plotly.com/ |
| Tableau Public | https://public.tableau.com/ |
| 2023 Canadian Wildfire Season | https://cwfis.cfs.nrcan.gc.ca/ |

---

*DATA 230 — San José State University — College of Information, Data and Society*
*Group 6 | Sanjana Thummalapalli · Elina Yin · Mansi Verma · Xuanhua (Carol) Li*
