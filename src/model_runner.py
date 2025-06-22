import os as os
import tensorflow as tf
import numpy as np
import yfinance as yf
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from src.data_processing import clean_stock_data, split_train_test_data
from sklearn.preprocessing import MinMaxScaler


class ModelRunner: 
    def __init__(self, config):
        # Initialise config
        self.config = config

        self.window_size = config["data"]["window_size"]
        self.model_dir = config["paths"]["model_dir"]
        self.model_name_template = config["paths"]["templates"]["lstm"]["model_name"]

    def load_model(self, symbol, model_type):
        symbol = symbol.upper()
        model_type = model_type.lower()
        model_path = os.path.join(self.model_dir, self.model_name_template.format(symbol = symbol.lower()))
        return load_model(model_path)

    def fetch_data(self, symbol,  period="1y"):
        return yf.download(symbol, period=period)

    def preprocess_data(self, data, mode = 'train'):
        clean_data = clean_stock_data(data)

        if mode =='train':
            train, test = split_train_test_data(clean_data)
            return train, test
        elif mode == 'predict':
            last_window = clean_data["Close"].values[-self.window_size:]
            return last_window

        

    def predict(self, symbol, model_type):
        stock_data = yf.download(symbol, period = "1y")
        close_prices = self.preprocess_data(stock_data, mode='predict')
        scaler = MinMaxScaler()
        scaler_data = scaler.fit_transform(close_prices.reshape(-1, 1))
        X_input = scaler_data.reshape(1, self.window_size, 1)
        model_path = os.path.join(self.model_dir, self.model_name_template.format(symbol = symbol.lower()))

        model = load_model(model_path)

        predictions_scaled = model.predict(X_input)

        predictions = scaler.inverse_transform(predictions_scaled)

        return {
            "symbol": symbol,
            "model" : model,
            "predicted_close_price" : predictions[0][0] 
        }



