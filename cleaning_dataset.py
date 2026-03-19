""" Original file is located at
    https://colab.research.google.com/drive/1WD7386dH-ry8mfN2o9S6_mkrqhMZEW9q
"""

import pandas as pd

# Load each year's dataset
df22 = pd.read_csv('/content/modis_2022_United_States.csv')
df23 = pd.read_csv('/content/modis_2023_United_States.csv')
df24 = pd.read_csv('/content/modis_2024_United_States.csv')

# Add a column to track the year (optional but useful)
df22['year'] = 2022
df23['year'] = 2023
df24['year'] = 2024

# Merge all three into one dataframe
df_all = pd.concat([df22, df23, df24], ignore_index=True)

# Show shape
df_all.shape

df_all.columns = df_all.columns.str.strip().str.lower().str.replace(' ', '_')

df_all.info()
df_all.head()
df_all['year'].value_counts()

df_all.to_csv('NASA_FIRMS_2022-24', index=False)

df_all.isna().sum()

(df_all.isna().sum() / len(df_all)) * 100

df_all.dtypes

df_all['confidence'].unique()

set(df22.columns), set(df23.columns), set(df24.columns)

