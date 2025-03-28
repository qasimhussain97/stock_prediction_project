import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_predictions(test_data, moving_avg_predictions, arima_predictions, lstm_predictions, cnn_lstm_predictions=None,symbol=None, save_path=None):
    plt.figure(figsize=(14, 7))
    plt.figure(figsize=(14, 7))
    plt.plot(test_data.index, test_data['Close'], label='Actual Prices', color='black')

    if moving_avg_predictions is not None:
        plt.plot(moving_avg_predictions.index, moving_avg_predictions['Moving_Average_Prediction'], label='Moving Average Prediction', color='blue')

    if arima_predictions is not None:
        plt.plot(arima_predictions.index, arima_predictions['ARIMA_Prediction'], label='ARIMA Prediction', color='red')

    if lstm_predictions is not None:
        plt.plot(test_data.index[-len(lstm_predictions):], lstm_predictions, label='LSTM', color='green')

    if cnn_lstm_predictions is not None:
        plt.plot(test_data.index[-len(cnn_lstm_predictions):], cnn_lstm_predictions, label='CNN-LSTM', color='orange')

    plt.title(f'{symbol} Stock Price Prediction: Actual vs. Models' if symbol else 'Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_residuals(test_data, moving_average_predictions, arima_predictions, lstm_predictions, cnn_lstm_predictions=None, symbol=None, save_path=None):
    plt.figure(figsize=(14, 7))

    if moving_average_predictions is not None:
        moving_avg_residuals = test_data['Close'] - moving_average_predictions['Moving_Average_Prediction']
        plt.plot(test_data.index[-len(moving_avg_residuals):], moving_avg_residuals, label='Moving Average Residuals', color='blue')

    if arima_predictions is not None:
        arima_residuals = test_data['Close'] - arima_predictions['ARIMA_Prediction']
        plt.plot(test_data.index[-len(arima_residuals):], arima_residuals, label='ARIMA Residuals', color='red')

    if lstm_predictions is not None:
        lstm_actuals = test_data['Close'][-len(lstm_predictions):]
        plt.plot(lstm_actuals.index, lstm_actuals - lstm_predictions, label='LSTM Residuals', color='green')

    if cnn_lstm_predictions is not None:
        cnn_lstm_actuals = test_data['Close'][-len(cnn_lstm_predictions):]
        plt.plot(cnn_lstm_actuals.index, cnn_lstm_actuals - cnn_lstm_predictions, label='CNN-LSTM Residuals', color='orange')


    plt.title(f'{symbol} Residual Errors by Model' if symbol else 'Residual Errors by Model')
    plt.xlabel('Date')
    plt.ylabel('Residual (Actual - Prediction)')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
        
def plot_model_predictions(test_data, predictions_dict, symbol, plots_dir, prediction_template, residual_template):

    os.makedirs(plots_dir, exist_ok=True)

    if not any(val is not None for val in predictions_dict.values()):
        print(f"No predictions to plot for {symbol}")
        return

    plot_path = os.path.join(plots_dir, prediction_template.format(symbol=symbol.lower()))
    residual_path = os.path.join(plots_dir, residual_template.format(symbol=symbol.lower()))

    plot_predictions(
        test_data=test_data,
        moving_avg_predictions=predictions_dict.get("Moving_Average"),
        arima_predictions=predictions_dict.get("Arima"),
        lstm_predictions=predictions_dict.get("LSTM"),
        cnn_lstm_predictions=predictions_dict.get("CNN-LSTM"),
        symbol=symbol,
        save_path=plot_path
    )

    plot_residuals(
        test_data=test_data,
        moving_average_predictions=predictions_dict.get("Moving_Average"),
        arima_predictions=predictions_dict.get("Arima"),
        lstm_predictions=predictions_dict.get("LSTM"),
        cnn_lstm_predictions=predictions_dict.get("CNN-LSTM"),
        symbol=symbol,
        save_path=residual_path
    )


