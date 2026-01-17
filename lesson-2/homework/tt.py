# =========================================================
# Data Preprocessing for Machine Learning
# California Housing Dataset (90-point solution)
# =========================================================

import os
import tarfile
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ---------------------------------------------------------
# 1. Download & Load Dataset
# ---------------------------------------------------------

DOWNLOAD_ROOT = "https://github.com/ageron/data/raw/main/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=housing_path)

fetch_housing_data()

df = pd.read_csv(os.path.join(HOUSING_PATH, "housing.csv"))

# ---------------------------------------------------------
# Part 1: Exploratory Data Analysis (EDA)
# ---------------------------------------------------------

print("\nFIRST 10 ROWS:")
print(df.head(10))

print("\nDATASET INFO:")
print(df.info())

print("\nSUMMARY STATISTICS:")
print(df.describe())

print("\nCATEGORICAL VALUE COUNTS:")
print(df["ocean_proximity"].value_counts())

# Identify numerical and categorical columns
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=["object"]).columns

print("\nNUMERICAL FEATURES:")
print(num_cols.tolist())

print("\nCATEGORICAL FEATURES:")
print(cat_cols.tolist())

# ---------------------------------------------------------
# Part 2: Missing Values
# ---------------------------------------------------------

print("\nMISSING VALUES PER COLUMN:")
print(df.isnull().sum())

# Missing report function
def missing_report(dataframe):
    missing_count = dataframe.isnull().sum()
    missing_percent = (missing_count / len(dataframe)) * 100
    report = pd.DataFrame({
        "column": missing_count.index,
        "missing_count": missing_count.values,
        "missing_percent": missing_percent.values
    })
    return report[report["missing_count"] > 0]

print("\nMISSING REPORT:")
print(missing_report(df))

# Median imputation for total_bedrooms
df["total_bedrooms"] = df["total_bedrooms"].fillna(
    df["total_bedrooms"].median()
)

# ---------------------------------------------------------
# Part 3: Encoding Categorical Variables
# ---------------------------------------------------------

df_encoded = pd.get_dummies(df, columns=["ocean_proximity"])

print("\nDATA AFTER ONE-HOT ENCODING:")
print(df_encoded.head())

# ---------------------------------------------------------
# Part 4: Feature Scaling
# ---------------------------------------------------------

scale_features = [
    "median_income",
    "housing_median_age",
    "population",
    "median_house_value"
]

# Histograms before scaling
df[scale_features].hist(bins=30, figsize=(10, 6))
plt.suptitle("Before Scaling")
plt.show()

# Standard Scaling
std_scaler = StandardScaler()
df_std = df_encoded.copy()
df_std[scale_features] = std_scaler.fit_transform(df_std[scale_features])

# Min-Max Scaling
mm_scaler = MinMaxScaler()
df_mm = df_encoded.copy()
df_mm[scale_features] = mm_scaler.fit_transform(df_mm[scale_features])

# Histograms after Standard Scaling
df_std[scale_features].hist(bins=30, figsize=(10, 6))
plt.suptitle("After Standard Scaling")
plt.show()

# Histograms after Min-Max Scaling
df_mm[scale_features].hist(bins=30, figsize=(10, 6))
plt.suptitle("After Min-Max Scaling")
plt.show()

# ---------------------------------------------------------
# Part 5: Feature Engineering (Optional)
# ---------------------------------------------------------

df_fe = df_encoded.copy()

df_fe["rooms_per_household"] = (
    df_fe["total_rooms"] / df_fe["households"]
)

df_fe["bedrooms_per_room"] = (
    df_fe["total_bedrooms"] / df_fe["total_rooms"]
)

df_fe["population_per_household"] = (
    df_fe["population"] / df_fe["households"]
)

print("\nFEATURE ENGINEERING SAMPLE:")
print(df_fe[[
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household"
]].head())

print("\nPreprocessing completed successfully.")
