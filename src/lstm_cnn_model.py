import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras_tuner import RandomSearch, Hyperband
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense

# Preprocessing function for CNN-LSTM
def preprocess_data_for_cnn_lstm(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Function to build the CNN-LSTM model
def build_cnn_lstm_model(input_shape, filters, kernel_size, pool_size, lstm_units, dropout, dense_units):
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(units=lstm_units))
    model.add(Dropout(dropout))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(1))  # Final output
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Function for hyperparameter tuning using Keras Tuner
def build_tuner(input_shape, project_name, max_epochs=100):  # Increased max_epochs
    def model_builder(hp):
        model = tf.keras.Sequential()

        # Hyperparameter tuning for the Conv1D layer
        hp_filters = hp.Int('filters', min_value=32, max_value=256, step=32)  # Wider range for filters
        model.add(tf.keras.layers.Conv1D(filters=hp_filters, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

        # Hyperparameter tuning for the LSTM layers
        hp_units = hp.Int('units', min_value=50, max_value=200, step=50)  # Wider range for LSTM units
        model.add(tf.keras.layers.LSTM(units=hp_units, return_sequences=True))
        model.add(tf.keras.layers.Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1)))  # Tuning dropout rate
        model.add(tf.keras.layers.LSTM(units=hp_units, return_sequences=False))
        model.add(tf.keras.layers.Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1)))  # Tuning dropout rate

        # Dense layers
        model.add(tf.keras.layers.Dense(units=hp.Int('dense_units', min_value=25, max_value=100, step=25)))  # Tuning Dense layer units
        model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))  # Ensure sigmoid activation

        # Compile model
        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),  # Tuning learning rate
                      loss='mean_squared_error')
        return model
    
    tuner = Hyperband(
        model_builder,
        objective='val_loss',
        max_epochs=max_epochs,
        factor=3,
        directory='my_dir',
        project_name=project_name
    )
    return tuner

# Function to train the model
def train_model(model, X_train, y_train, batch_size=32, epochs=50):  # Increased epochs
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

# Function to make predictions
def predict_with_model(model, X_test, scaler):
    predicted_prices = model.predict(X_test)
    return scaler.inverse_transform(predicted_prices)

# Function to evaluate the model
def evaluate_model(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(y_true, y_pred)

# Function to plot the results
def plot_results(actual_prices, predicted_prices, dates, title='CNN-LSTM Model: Actual vs. Predicted Prices'):
    plt.figure(figsize=(14, 7))
    plt.plot(dates, actual_prices, label='Actual Prices', color='black')
    plt.plot(dates, predicted_prices, label='CNN-LSTM Predictions', color='green')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
