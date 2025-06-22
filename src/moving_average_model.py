import pandas as pd
from sklearn.metrics import mean_squared_error

def moving_average_baseline(training_Data, test_Data, window=5):
    if len(training_Data) + len(test_Data) < window:
        return pd.DataFrame(columns=['Moving_Average_Prediction'], index=test_Data.index)

    combined_data = pd.concat([training_Data, test_Data])

    combined_data['Moving_Average_Prediction'] = (
        combined_data['Close'].rolling(window=window).mean().shift(1)
    )

    combined_data.dropna(subset=['Moving_Average_Prediction'], inplace=True)
    
    predictions = combined_data.loc[test_Data.index.intersection(combined_data.index)]
    return predictions

def calculate_mse(actual_values, predicted_values):
    if predicted_values.empty:
        return float('nan')

    aligned_actuals = actual_values.loc[predicted_values.index]
    
    mse = mean_squared_error(aligned_actuals, predicted_values)
    return mse