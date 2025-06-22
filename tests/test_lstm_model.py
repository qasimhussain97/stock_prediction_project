import numpy as np
import tensorflow as tf
from src.lstm_model import preprocess_data_for_lstm, build_lstm_model

def test_preprocess_data_for_lstm():
    
    data = np.array(range(20)) # A simple sequence from 0 to 19
    window_size = 5

    X, y = preprocess_data_for_lstm(data, window_size)

    expected_samples = 15
    assert X.shape == (expected_samples, window_size), f"Test Failed: Shape of X is {X.shape}, expected ({expected_samples}, {window_size})."
    assert y.shape == (expected_samples,), f"Test Failed: Shape of y is {y.shape}, expected ({expected_samples},)."

    assert np.array_equal(X[0], np.array([0, 1, 2, 3, 4])), "Test Failed: First sequence in X is incorrect."
    assert y[0] == 5, f"Test Failed: First label in y is {y[0]}, expected 5."

    assert np.array_equal(X[-1], np.array([14, 15, 16, 17, 18])), "Test Failed: Last sequence in X is incorrect."
    assert y[-1] == 19, f"Test Failed: Last label in y is {y[-1]}, expected 19."

def test_build_lstm_model():
    input_shape = (60, 1) 

    model = build_lstm_model(input_shape)

    assert isinstance(model, tf.keras.Sequential), "Test Failed: Model is not a Keras Sequential model."

    assert len(model.layers) == 6, f"Test Failed: Expected 6 layers, but model has {len(model.layers)}."

    assert model.output_shape == (None, 1), f"Test Failed: Model output shape is {model.output_shape}, expected (None, 1)."