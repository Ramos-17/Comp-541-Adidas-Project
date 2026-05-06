import os
import pandas as pd
import numpy as np
from pathlib import Path

def load_data() -> pd.DataFrame:
    path = Path(__file__).parent.parent /"data" / "AdidasDataset2023to2025.csv"
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


if __name__ == "__main__":
    df = load_data()
    print(df.shape)

    
