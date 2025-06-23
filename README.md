# Stock Price Prediction: A Comparative Study
This project started as a curiosity driven dive into stock price forecasting. I wanted to move beyond just training a single model and instead build a complete, end-to-end pipeline to compare classical statistical models against more modern deep learning approaches.

The project downloads historical stock data, processes it, trains four distinct models (Moving Average, ARIMA, LSTM, and CNN-LSTM), and evaluates their performance to see which one proves most effective. To complete the workflow, it also includes a REST API to serve predictions from the trained models.

# Key Features
Multi-Model Comparison: Systematically trains and evaluates four different models: a simple Moving Average, a classical ARIMA model, a Long Short-Term Memory (LSTM) network, and a hybrid CNN-LSTM network.
Configuration Driven: All important parameters, from stock symbols to model hyperparameters, are managed in a central config.yaml file, making it easy to run new experiments.
Automated Evaluation: The project automatically calculates and saves key performance metrics (RMSE and MAE) for every model and every stock, saving the results to metrics_summary.csv and metrics_summary.json.
Prediction API: Includes a working REST API built with FastAPI that can load a trained model and its corresponding data scaler to serve live predictions.
Tested & Reliable: The core logic is validated with a suite of unit and integration tests using pytest to ensure the data processing and API endpoints are reliable.
Results & Key Takeaway
After running the pipeline, one of the most interesting takeaways was that for this dataset, the simpler, classical ARIMA model consistently outperformed the more complex deep learning models like LSTM and CNN-LSTM. This is a great reminder that the newest or most complex tool isn't always the best one for the job.

Here is the full performance summary:

| Stock | Model | RMSE | MAE |
| :--- | :--- | :--- | :--- |
| AAPL | MovingAverage | 4.9406 | 3.8631 |
| AAPL | Arima | 3.1748 | 2.3160 |
| AAPL | LSTM | 7.3517 | 6.0099 |
| AAPL | CNN-LSTM | 13.9706 | 12.6807 |
| TSLA | MovingAverage | 16.2289 | 11.3117 |
| TSLA | Arima | 10.8803 | 7.6139 |
| TSLA | LSTM | 18.9292 | 13.7508 |
| TSLA | CNN-LSTM | 22.2921 | 15.6918 |
| AMZN | MovingAverage | 4.7857 | 3.7258 |
| AMZN | Arima | 3.3536 | 2.5255 |
| AMZN | LSTM | 9.5505 | 8.1363 |
| AMZN | CNN-LSTM | 6.1335 | 4.9396 |
| KO | MovingAverage | 0.8866 | 0.6570 |
| KO | Arima | 0.5878 | 0.4340 |
| KO | LSTM | 1.1620 | 0.8758 |
| KO | CNN-LSTM | 2.4258 | 2.1197 |
| JPM | MovingAverage | 4.7256 | 3.4514 |
| JPM | Arima | 3.1471 | 2.0606 |
| JPM | LSTM | 8.4861 | 7.1901 |
| JPM | CNN-LSTM | 13.1713 | 12.0391 |
| NVDA | MovingAverage | 5.5101 | 4.2224 |
| NVDA | Arima | 3.9560 | 2.8564 |
| NVDA | LSTM | 23.8635 | 22.2872 |
| NVDA | CNN-LSTM | 15.1134 | 13.6888 |

Export to Sheets
(Results sourced from metrics_summary.csv)

# How to Get Started
Getting the project running is straightforward.

1. Set up your environment:

Bash

# Clone the repository
git clone [your-github-repo-url]
cd stock-prediction-project

# Install the required packages
pip install -r requirements.txt
2. Run the Training Pipeline:

This command will download the data, train all models for all stocks, and save the results, plots, and models.

Bash

python main.py
You can also run it for a specific stock or model:

Bash

# Run all models for only Apple
python main.py --stock aapl

# Run only the ARIMA model for all stocks
python main.py --model arima
3. Run the Prediction API:

First, make sure you've run the training script for a stock (e.g., aapl) so that a model exists.

Bash

# Launch the API server
uvicorn api.app:app --reload
The API will be available at http://127.0.0.1:8000. You can visit http://127.0.0.1:8000/docs in your browser to see the interactive API documentation.

A Look at the Project Structure
I organized the project to keep the code clean and easy to navigate.

api/: Contains the FastAPI application for serving predictions.
models/: This is where the trained model files (.h5, .pkl) and scaler objects (.joblib) are saved.
plots/: Stores the generated plots comparing model predictions.
src/: Holds all the core Python modules for data processing, model definitions, plotting, and other utilities.
tests/: Contains all the unit and integration tests.
main.py: The main script that orchestrates the entire training and evaluation pipeline.
config.yaml: The central configuration file.
.gitignore: Standard file to exclude unnecessary files from version control.