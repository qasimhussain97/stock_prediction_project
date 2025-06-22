import pandas as pd
from scipy.stats import zscore

def clean_stock_data(df):
    df.ffill(inplace=True)
    df.drop_duplicates(inplace=True)
    Q1 = df['Close'].quantile(0.25)
    Q3 = df['Close'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df = df[(df['Close'] >= lower_bound) & (df['Close'] <= upper_bound)]
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.index = pd.to_datetime(df.index)
    return df

def split_train_test_data(data, split_ratio=0.8):
    split_index = int(len(data) * split_ratio)
    training_data = data[:split_index]
    testing_data = data[split_index:]
    return training_data, testing_data
