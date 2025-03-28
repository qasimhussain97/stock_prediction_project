import pandas as pd
from scipy.stats import zscore

def clean_stock_data(df):
    df.ffill(inplace=True)
    df.drop_duplicates(inplace=True)
    df['Z-Score'] = zscore(df['Close'])
    df = df[(df['Z-Score'] < 3) & (df['Z-Score'] > -3)]
    df.drop(columns=['Z-Score'])
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.index = pd.to_datetime(df.index)
    return df

def split_train_test_data(data, split_ratio=0.8):
    split_index = int(len(data) * split_ratio)
    training_data = data[:split_index]
    testing_data = data[split_index:]
    return training_data, testing_data

