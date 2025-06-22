import pandas as pd
import numpy as np
import pytest
from src.data_processing import clean_stock_data, split_train_test_data

def test_clean_stock_data():
    data = {
        'Open' : [100, 101, 102, 102, 104, 500],
        'High' : [102, 103, 104, 104, 106, 505],
        'Low' : [99, 100, 101, 101, 103, 495],
        'Close' : [101, 102, np.nan, 103, 105, 499], 
        'Volume': [1000, 1100, 1200, 1200, 1300, 5000]
    }

    df = pd.DataFrame(data)
    duplicate_row = pd.DataFrame([df.iloc[1]], columns = df.columns)
    df = pd.concat([df, duplicate_row], ignore_index=True)
    df.index = pd.to_datetime(pd.date_range(start = '2023-01-01', periods= len(df)))

    cleaned_df = clean_stock_data(df.copy())

    assert not cleaned_df.isnull().values.any(), "Test Failed: NaN values were not filled."

    assert len(cleaned_df) == 5, f"Test Failed: Expected 5 rows after cleaning, but got  {len(cleaned_df)}."

    assert 499 not in cleaned_df['Close'].values, "Test Failed: Outlier was not removed."

    expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    assert all(col in cleaned_df.columns for col in expected_columns),  "Test Failed: Columns are incorrect."

def test_split_train_test_data():

    data = {'Close' : range(100)}
    df = pd.DataFrame(data)

    train_data, test_data = split_train_test_data(df, split_ratio=0.8)

    assert len(train_data) == 80, f"Test Failed: Expected 80 training rows, got {len(train_data)}"
    assert len(test_data) == 20, f"Test Failed: Expected 20 testing rows, got {len(test_data)}"

    assert train_data.index.max() < test_data.index.min(), "Test Failed: Data is not sequential."
    