Stock Price Prediction App

Overview
The Stock Price Prediction App is a dynamic and user-friendly web application built using Streamlit. It offers tools for stock analysis, technical insights, and price forecasting. Leveraging data visualization and advanced predictive modeling, the app equips users with actionable information about stock trends to aid in making informed financial decisions.


Features
1. Home Page

Explore key information about the selected stock, including:

Stock Overview: Sector, market cap, EPS, PE ratio, beta, profit margins, and other financial metrics.

Historical Data: User-specified date ranges for stock prices.

Daily Price Changes: Insights into price fluctuations.

Interactive Timeframes: Quick access to 5D, 1M, 6M, and YTD data.

2. Stock Analysis Page
   
Delve deeper into stock performance using advanced visualizations and indicators:

Candlestick Charts: Examine price movements across chosen time periods.

Line Charts: Overlay high, low, open, and close prices.

Technical Indicators:

RSI (Relative Strength Index): Highlights overbought or oversold conditions.

MACD (Moving Average Convergence Divergence): Tracks price momentum.

SMA-50 (Simple Moving Average): Monitors long-term trends.

3. Stock Prediction Page

Predict future stock prices using the ARIMA model for time-series forecasting:

30-Day Forecast: Generates future closing price predictions.

Preprocessing Techniques: Implements rolling mean, differencing, and stationarity checks.

Scaling: Ensures accurate predictions with scaling and inverse transformations.

Evaluation: Utilizes RMSE to measure model accuracy.

Visualization: Displays forecasted trends with clear, interactive charts.

Technical Overview

Tech Stack

Frontend: Streamlit for an intuitive and interactive UI.

Data Source: Yahoo Finance API (yfinance) for fetching stock data.

Visualizations:

Plotly for creating interactive charts.

TA-Lib (pandas-ta) for technical indicator calculations.

Data Processing:

Pandas for data manipulation and preprocessing.

Scikit-learn for scaling and performance evaluation.

Forecasting:

ARIMA for time-series modeling and forecasting.

Installation

Prerequisites

Python 3.7 or higher.

Required Python libraries (detailed in requirements.txt).

Setup Instructions

Clone the repository:

bash

Copy code

git clone https://github.com/your-repo-name/stock-price-prediction.git  

cd stock-price-prediction  

Install the dependencies:

bash
Copy code
pip install -r requirements.txt  
Run the application:

bash
Copy code
streamlit run app.py  
Open the app in your browser at http://localhost:8501.

How to Use

Home Page: Explore stock summaries and historical data.

Stock Analysis Page: Analyze trends and visualize technical indicators.

Stock Prediction Page: Forecast stock prices using machine learning models.

Visualizations

Candlestick Chart: Captures intraday price movements.

Line Chart: Tracks metrics like RSI, MACD, and SMA over time.

Forecast Chart: Projects stock trends for the next 30 days.

Future Enhancements

Expanded Forecasting Models: Incorporate advanced models like LSTM and Prophet.

Sentiment Analysis: Analyze market news for stock sentiment trends.

