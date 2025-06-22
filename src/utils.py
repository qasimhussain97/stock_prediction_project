import json
import os
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from logger import logger 

def evaluate_deep_learning_model(model, model_name, symbol, X_test, scaler, test_close_prices, window_size):
    predictions_scaled = model.predict(X_test)
    predictions_inversed = scaler.inverse_transform(predictions_scaled)

    actual = test_close_prices[window_size:].values
    rmse = np.sqrt(mean_squared_error(actual, predictions_inversed))
    mae = mean_absolute_error(actual, predictions_inversed)
    
    logger.info(f"RMSE ({model_name}) {symbol}: {rmse}")
    logger.info(f"MAE ({model_name}) {symbol}: {mae}")
    
    return predictions_inversed.flatten(), rmse, mae

def log_metrics(metrics_table, stock, model_name, rmse, mae):
    metrics_table.append({
        'Stock': stock,
        'Model': model_name, 
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4)
    })

def save_model_metadata(model_dir, template, stock, model_name, rmse, mae): 
    metadata_path = os.path.join(model_dir, template.format(symbol=stock.lower()))
    metadata = {
        "stock" : stock,
        "model_name": model_name, 
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(metadata_path, "w") as f: 
        json.dump(metadata, f, indent = 4)


