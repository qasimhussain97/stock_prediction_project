import pandas as pd
from sklearn.metrics import mean_squared_error

def moving_average_baseline(training_Data, test_Data, window=5):

    combined_data = pd.concat([training_Data, test_Data]).copy()

    if 'Close' not in combined_data.columns: 
        raise ValueError("'Close' column not found in data.")
    
    combined_data['Moving_Average_Prediction'] = (
        combined_data['Close'].rolling(window=window).mean().shift(1)
    )

    if 'Moving_Average_Prediction' not in combined_data.columns:
        print("Column names:", combined_data.columns)
        raise ValueError("Prediction column is missing!")
    
    if combined_data['Moving_Average_Prediction'].isnull().all():
        raise ValueError("Prediction column exists but contains only NaNs.")

    print("Combined data columns:", combined_data.columns)
    print("Preview of prediction column:")
    print(combined_data['Moving_Average_Prediction'].head(10))
    
    combined_data = combined_data.dropna(subset=['Moving_Average_Prediction'])
   
    
    valid_index = test_Data.index.intersection(combined_data.index)
    predictions = combined_data.loc[valid_index]
    # Return only the testing period predictions
    return predictions


# def moving_average_baseline(training_Data, test_Data, window=5):

#     combined_data = pd.concat([training_Data, test_Data])
#     combined_data = combined_data.copy()

#     combined_data['Moving_Average_Prediction'] = (
#         combined_data['Close'].rolling(window=window).mean().shift(1)
#     )

#     combined_data = combined_data.dropna(subset=['Moving_Average_Prediction'])
    
#     # Return only the testing period predictions
#     predictions = combined_data.loc[test_Data.index.intersection(combined_data.index)]
#     return predictions



def calculate_mse(actual_values, predicted_values):
    # Calculate the Mean Squared Error
    mse = mean_squared_error(actual_values, predicted_values)
    return mse


