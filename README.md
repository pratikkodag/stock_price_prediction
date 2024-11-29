<img width="960" alt="image" src="https://github.com/user-attachments/assets/74b77378-fbd1-4d71-a729-e7d964a3d1f2">

## Stock Price Prediction 

The Stock Price Prediction System is a data-driven application designed to analyze historical stock data, provide technical insights, and forecast future stock trends. Below is a detailed explanation of the process and techniques used:

## Data Source
Yahoo Finance API

Data Source: Stock data such as historical prices, financial metrics, and market trends are fetched using the Yahoo Finance API (yfinance).

Real-Time Updates: The API ensures the app always provides up-to-date financial data.

Stock Ticker Input: Users select stock tickers to analyze and predict.

## Data Preprocessing

To ensure accurate analysis and forecasting, raw stock data underwent preprocessing:

Missing Data Handling: Missing values in stock prices and metrics were imputed using interpolation or forward-filling techniques.
# Feature Engineering:

Daily Returns: Calculated percentage changes in stock prices for volatility analysis.

Stationarity Checks: Used rolling means and differencing to make the time series stationary for forecasting.

# Feature Extraction

Key features extracted for analysis include:

Stock Prices: Open, high, low, close, and adjusted close prices.

Volume: Total shares traded during each period.

# Technical Indicators:

Relative Strength Index (RSI): Identifies overbought or oversold conditions.

MACD (Moving Average Convergence Divergence): Highlights momentum shifts.

Simple Moving Averages (SMA-50, SMA-200): Tracks long-term trends.

# Prediction Model
The ARIMA model was used for time-series forecasting:

Data Transformation: Rolling mean and differencing were applied to remove trends and achieve stationarity.

# Parameter Selection:
Auto-ARIMA: Selected optimal parameters (p, d, q) based on AIC/BIC criteria.

# Model Training: Trained on historical stock prices to learn patterns and trends.

Forecasting: Predicted closing prices for the next 30 days.

Scaling:
Min-Max Scaling: Scaled the data to fit within a specific range for better model performance.

Inverse Transformation: Applied to rescale predictions back to their original values.

# Evaluation Metrics:
RMSE (Root Mean Squared Error): Measured model accuracy by comparing predicted prices with actual values.

Visualization
Interactive and intuitive charts provide insights into stock performance:

Candlestick Charts: Visualize intraday price movements.

Line Charts: Overlay open, close, high, and low prices with technical indicators.

Forecast Charts: Display predicted price trends for the next 30 days.

User Interaction
Stock Selection: Users can choose stocks from a dropdown menu.

Timeframes: Select predefined periods like 5D, 1M, 6M, YTD for analysis.

Real-Time Updates: Outputs are updated dynamically based on user input.

# Libraries and Tools Used
Pandas: For handling and preprocessing tabular stock data.

NumPy: For numerical operations and feature engineering.

Scikit-learn: For scaling, evaluation metrics, and preprocessing.

Statsmodels: For implementing the ARIMA model.

TA-Lib (pandas-ta): For calculating technical indicators.

Plotly: For creating interactive and visually appealing charts.

Streamlit: For building an intuitive user interface.

Yahoo Finance API (yfinance): For fetching stock data.

# Output and Recommendations

The app provides users with:

Detailed analysis of stock trends using historical data and technical indicators.
Reliable predictions of future stock prices based on ARIMA models.
User-friendly visualizations to make data interpretation straightforward.

