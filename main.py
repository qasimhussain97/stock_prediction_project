import yaml
import os
import yfinance as yf
import numpy as np
import pandas as pd
import json
import argparse
from datetime import datetime
from tensorflow.keras.models import load_model
from logger import logger
from src.utils import log_metrics, save_model_metadata
from src.moving_average_model import moving_average_baseline, calculate_mse
from src.plotting import plot_predictions, plot_residuals, plot_model_predictions
from src.arima_model import arima_predict, grid_search_arima
from src.lstm_model import preprocess_data_for_lstm, build_lstm_model
from src.lstm_cnn_model import build_cnn_lstm_model
from src.data_processing import clean_stock_data, split_train_test_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

def parse_args():
    parser = argparse.ArgumentParser(description = "Stock Price Prediction CLI")
    parser.add_argument("--stock", type=str, help="Stock symol (e.g., AAPL)", default = None)
    parser.add_argument("--model", type=str, choices=["moving_average", "arima", "lstm", "cnn_lstm", "all"], default="all", help="Model to run")
    parser.add_argument("--predict_only", action="store_true", help="Skip training and only predict")
    return parser.parse_args()

args = parse_args()

with open("config.yaml", "r") as f: 
    config = yaml.safe_load(f)

logger.info("Loaded config:", config)
logger.info(f"Stock symbols: {config['data']['stock_symbols']}")

window_size = config["data"]["window_size"]
all_symbols = config["data"]["stock_symbols"]
stock_symbols = [args.stock] if args.stock else all_symbols
batch_size = config["model"]["lstm"]["batch_size"]
epochs = config["model"]["lstm"]["epochs"]
model_dir = config["paths"]["model_dir"]
model_name_template = config["paths"]["templates"]["lstm"]["model_name"]
metadata_name_template = config["paths"]["templates"]["lstm"]["metadata_name"]
plots_dir = config["paths"]["plots_dir"]
prediction_plot_template = config["paths"]["templates"]["plots"]["prediction_plot"]
residual_plot_template = config["paths"]["templates"]["plots"]["residual_plot"]
cnn_config = config["model"]["cnn_lstm"]
cnn_model_name_template = config["paths"]["templates"]["cnn_lstm"]["model_name"]
cnn_metadata_template = config["paths"]["templates"]["cnn_lstm"]["metadata_name"]

# Data preprocessing
metrics_table = []


logger.info("Downloading and preprocessing data...")
data = {symbol: yf.download(symbol, start="2019-01-01", end="2025-03-25") for symbol in stock_symbols}
for symbol in data:
    if isinstance(data[symbol].columns, pd.MultiIndex):
        data[symbol].columns = data[symbol].columns.get_level_values(0)

cleaned_data = {s: clean_stock_data(df) for s, df in data.items()}
split_data = {s: split_train_test_data(df) for s, df in cleaned_data.items()}

for symbol, (train, test) in split_data.items():
    predictions_dict = {}

    # Moving_average
    if args.model in ["moving_average", "all"]: 
        ma_predictions = moving_average_baseline(train, test, window=5)  # You can parametrize window size if you like
        ma_rmse = np.sqrt(calculate_mse(test['Close'], ma_predictions['Moving_Average_Prediction']))
        logger.info(f"RMSE (Moving_Average) {symbol}: {ma_rmse}")
        ma_mae = mean_absolute_error(test['Close'], ma_predictions['Moving_Average_Prediction'])
        logger.info(f"MAE (Moving_Average) {symbol}: {ma_mae}")

        log_metrics(metrics_table, symbol, "Moving_Average", ma_rmse, ma_mae)
        save_model_metadata(model_dir, metadata_name_template, symbol, "Moving_Average", ma_rmse, ma_mae)

        predictions_dict["Moving_Average"] = ma_predictions

    # Arima 
    if args.model in ["arima", "all"]:
        # Adjust order (p, d, q) as needed
        arima_predictions = arima_predict(train, test, order = (5, 1, 0))
        arima_rmse = np.sqrt(mean_squared_error(test['Close'], arima_predictions['ARIMA_Prediction']))
        logger.info(f"RMSE (Arima) {symbol}: {arima_rmse}")
        arima_mae = mean_absolute_error(test['Close'], arima_predictions['ARIMA_Prediction'])
        logger.info(f"MAE (Arima) {symbol}: {arima_mae}")

        log_metrics(metrics_table, symbol, "Arima", arima_rmse, arima_mae)
        save_model_metadata(model_dir, metadata_name_template, symbol, "Arima", arima_rmse, arima_mae)

        predictions_dict["Arima"] = arima_predictions



    # LSTM starts here
    if args.model in ["lstm", "all"]:
        scaler = MinMaxScaler()
        scaled_train = scaler.fit_transform(train[['Close']])
        scaled_test = scaler.transform(test[['Close']])

        X_train, y_train = preprocess_data_for_lstm(scaled_train, window_size)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        model_path = os.path.join(model_dir, model_name_template.format(symbol=symbol.lower()))

        # Load or train model
        if os.path.exists(model_path):
            logger.info("Loading existing LSTM model...")
            model = load_model(model_path)
        else:
            logger.info("Training new LSTM model...")
            model = build_lstm_model((X_train.shape[1], 1))
            model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
            model.save(model_path)
            logger.info("Model saved to disk.")

        # Prepare test data
        X_test, y_test = preprocess_data_for_lstm(scaled_test, window_size)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Predict
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)

        actual = test['Close'][window_size:]
        lstm_rmse = np.sqrt(mean_squared_error(actual, predictions))
        logger.info(f"RMSE (LSTM) {symbol}: {lstm_rmse}")
        lstm_mae = mean_absolute_error(actual, predictions) 
        logger.info(f"MAE (LSTM) {symbol}: {lstm_mae}")

        log_metrics(metrics_table, symbol, "LSTM", lstm_rmse, lstm_mae)
        save_model_metadata(model_dir, metadata_name_template, symbol, "LSTM", lstm_rmse, lstm_mae)

        predictions_dict["LSTM"] = predictions.flatten()

    # CNN_LSTM
    if args.model in ["cnn_lstm", "all"]:
        X_train, y_train = preprocess_data_for_lstm(scaled_train, window_size)
        X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  

        cnn_model_path = os.path.join(model_dir, cnn_model_name_template.format(symbol=symbol.lower()))
        if os.path.exists(cnn_model_path):
            logger.info("Loading existing CNN-LSTM model...")
            cnn_model = load_model(cnn_model_path)
        else: 
            logger.info("Training CNN-LSTM model...")
            cnn_model = build_cnn_lstm_model(
                input_shape=(X_train_cnn.shape[1], 1),
                filters=cnn_config["filters"],
                kernel_size=cnn_config["kernel_size"],
                pool_size=cnn_config["pool_size"],
                lstm_units=cnn_config["lstm_units"],
                dropout=cnn_config["dropout"],
                dense_units=cnn_config["dense_units"]
            )
            cnn_model.fit(
                X_train_cnn, y_train,
                batch_size=cnn_config["batch_size"],
                epochs=cnn_config["epochs"],
                verbose=0
            )
            cnn_model.save(cnn_model_path)
            logger.info("CNN-LSTM model saved.")

        # Test data
        X_test, y_test = preprocess_data_for_lstm(scaled_test, window_size)
        X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        cnn_predictions = cnn_model.predict(X_test_cnn)
        cnn_predictions = scaler.inverse_transform(cnn_predictions)

        actual = test['Close'][window_size:]
        cnn_rmse = np.sqrt(mean_squared_error(actual, cnn_predictions))
        logger.info(f"RMSE (CNN-LSTM) {symbol}: {cnn_rmse}")
        cnn_mae = mean_absolute_error(actual, cnn_predictions)
        logger.info(f"MAE (CNN-LSTM) {symbol}: {cnn_mae}")

        log_metrics(metrics_table, symbol, "CNN-LSTM", cnn_rmse, cnn_mae)
        save_model_metadata(model_dir, cnn_metadata_template, symbol, "CNN-LSTM", cnn_rmse, cnn_mae)

        predictions_dict["CNN-LSTM"] = cnn_predictions.flatten()


    # Align index with test_data by trimming the beginning (like LSTM)
    if "CNN-LSTM" in predictions_dict:
        aligned_index = test.index[window_size:]
        cnn_lstm_df = pd.DataFrame({
            'CNN_LSTM_Prediction': predictions_dict["CNN-LSTM"]
        }, index=aligned_index)
        predictions_dict["CNN-LSTM"] = cnn_lstm_df["CNN_LSTM_Prediction"]

    os.makedirs(plots_dir, exist_ok=True)
    plot_model_predictions(
        test_data=test,
        predictions_dict=predictions_dict,
        symbol=symbol,
        plots_dir=plots_dir,
        prediction_template=prediction_plot_template,
        residual_template=residual_plot_template
    )

metrics_table_df = pd.DataFrame(metrics_table)

metrics_table_df.to_csv("metrics_summary.csv", index=False)
metrics_table_df.to_json("metrics_summary.json", orient = "records", indent= 4)

logger.info(metrics_table_df.to_string(index=False))