import tensorflow as tf
from keras_tuner import RandomSearch, Hyperband

def build_lstm_model(hp, window_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=hp.Int('units', min_value=32, max_value=128, step=16), 
                                   return_sequences=True, 
                                   input_shape=(window_size, 1)))
    model.add(tf.keras.layers.Dropout(hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(tf.keras.layers.LSTM(units=hp.Int('units', min_value=32, max_value=128, step=16), 
                                   return_sequences=False))
    model.add(tf.keras.layers.Dropout(hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(tf.keras.layers.Dense(units=hp.Int('dense_units', min_value=10, max_value=50, step=10)))
    model.add(tf.keras.layers.Dense(1))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='mean_squared_error')
    return model

def tune_lstm_hyperparameters(X_train, y_train, window_size):
    tuner = RandomSearch(
        lambda hp: build_lstm_model(hp, window_size),  # Pass window_size here
        objective='val_loss',
        max_trials=5,
        executions_per_trial=3,
        directory='lstm_tuning',
        project_name='stock_prediction'
    )

    tuner.search(X_train, y_train, epochs=10, validation_split=0.2)
    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model


