from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

def arima_predict(training_data, testing_data, order=(5,1,0)):
    history = [x for x in training_data['Close']]
    predictions = []

    for t in range(len(testing_data)):
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = testing_data['Close'].iloc[t]
        history.append(obs)
    
    result = testing_data.copy()
    result['ARIMA_Prediction'] = predictions
    return result

def calculate_mse_for_arima(testing_data, predictions_column='ARIMA_Prediction'):
    # Calculate the Mean Squared Error between the actual and predicted values
    mse = mean_squared_error(testing_data['Close'], testing_data[predictions_column])
    return mse

def evaluate_arima_model(train, test, arima_order):
    history = [x for x in train['Close']]
    predictions = list()
    
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test['Close'].iloc[t])
    
    error = mean_squared_error(test['Close'], predictions)
    return error

def grid_search_arima(train, test, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_arima_model(train, test, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print(f'ARIMA{order} MSE={mse}')
                except:
                    continue
    print(f'Best ARIMA{best_cfg} MSE={best_score}')
    return best_cfg
