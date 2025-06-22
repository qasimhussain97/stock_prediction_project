from fastapi import FastAPI, HTTPException
import tensorflow as tf
import numpy as np
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="Stock Prediction API")

class StockInput(BaseModel):
    data: list[float] 

@app.post("/predict/{stock_symbol}")
def predict_stock(stock_symbol: str, input_data: StockInput):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "..", "models", f"lstm_model_{stock_symbol.lower()}.h5")
    scaler_path = os.path.join(base_dir, "..", "models", f"scaler_lstm_{stock_symbol.lower()}.joblib")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise HTTPException(status_code=404, detail=f"Model or scaler for {stock_symbol} not found.")

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    input_array = np.array(input_data.data).reshape(-1, 1)
    if input_array.shape[0] != model.input_shape[1]:
        raise HTTPException(status_code=400, detail=f"Input data must have {model.input_shape[1]} elements.")

    scaled_data = scaler.transform(input_array)
    reshaped_data = scaled_data.reshape(1, -1, 1)

    prediction_scaled = model.predict(reshaped_data)
    prediction = scaler.inverse_transform(prediction_scaled)

    return {"stock_symbol": stock_symbol, "prediction": prediction.tolist()[0][0]}