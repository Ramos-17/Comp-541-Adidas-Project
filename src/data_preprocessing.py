# This file will contain functions for data preprocessing
# It will include functions for data cleaning, transformation, feature selection, and feature engineering.
# This is for assignment 4

import os
import pandas as pd
import numpy as np
from pathlib import Path

def load_data() -> pd.DataFrame:
    path = Path(__file__).parent.parent /"data" / "AdidasDataset2023to2025.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset at {path}")
    return pd.read_csv(path)



def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    return df

def data_transformation(df: pd.DataFrame) -> pd.DataFrame:
    # Convert Order_Date from string type to pandas datatime type
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    return df

def feature_selection(df: pd.DataFrame) -> pd.DataFrame:
    # Remove customer_age'
    df.drop("Customer_Age", axis = 1, inplace=True)
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Extract Quarter feature from Order_Date column, this new feature will identify which quarter the product is bought in
    df['Quarter'] = df['Order_Date'].dt.quarter.map({1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'})

    # Extract Avg_Product_Sales feature from Category and Revenue columns, this new feature will identify the average sales of each product
    df['Avg_Product_Sales'] = df.groupby('Category')['Revenue'].transform('mean')

    return df