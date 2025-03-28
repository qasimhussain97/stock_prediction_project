from fastapi import FastAPI
import tensorflow as tf
import numpy as np
from pydantic import BaseModel

app = FastAPI()

class StockInput(BaseModel):
    data: list

model = tf.keras.models.load_model("models/lstm_model.h5")

@app.post("/predict")
def predict_stock(input_data: StockInput):
    data = np.array(input_data.data).reshape(1, -1, 1)
    prediction = model.predict(data).tolist()
    return {"prediction": prediction}
