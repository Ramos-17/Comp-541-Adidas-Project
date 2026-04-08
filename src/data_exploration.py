import os
import pandas as pd
import numpy as np
from pathlib import Path

def load_data() -> pd.DataFrame:
    path = Path(__file__).parent.parent /"data" / "adidas_usa.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset at {path}")
    return pd.read_csv(path)

def outlier_counts_using_iqr (df:pd.DataFrame) -> pd.Series:
    counts = {}
    number_cols = df.select_dtypes(include = ("number")).columns
    for col in number_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        counts[col] = len(outliers)
    return pd.Series(counts)

# Function to fill missing values in 'original_price' with 'selling_price' and convert to numeric
def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df['original_price'] = df['original_price'].fillna(df['selling_price'])
    df['original_price'] = pd.to_numeric(
        df['original_price'].astype(str).str.replace('[^0-9.]', '', regex=True),
        errors='coerce'
    )
    df['selling_price'] = pd.to_numeric(
        df['selling_price'].astype(str).str.replace('[^0-9.]', '', regex=True),
        errors='coerce'
    )
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(df.info())
    print(df.isnull().sum())
    print("--------------Outlier Counts using IQR Method----------------")
    print(outlier_counts_using_iqr(df))
    print("--------------Filling Missing Values----------------")
    df = fill_missing_values(df)
    print(df.isnull().sum())
    
