import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error,r2_score
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st 
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Stock Prediction",
    page_icon="chart_with_upwords_trend",
    layout= "wide"
)

st.title("Stock Prediction")
col1,clo2,clo3 = st.columns(3)

with col1:
    ticker = st.text_input('Stock Ticker','AAPL')

def get_data(ticker):
    stock_data = yf.download(ticker,start='2024-01-01')
    return stock_data[['Close']]


def get_rollng_mean(close_price):
    rolling_price = close_price.rolling(window=7).mean().dropna()
    return rolling_price

def stationary_check(close_price):
    adf_test=adfuller(close_price)
    p_value=round(adf_test[1],3)
    return p_value


def get_differencing_order(close_price):
    p_value = stationary_check(close_price)
    d=0
    while True:
        if p_value > 0.05:
            d=d+1
            close_price=close_price.diff().dropna()
            p_value = stationary_check(close_price)
        else:
            break
    return d


def scaling(close_price):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(np.array(close_price).reshape(-1,1))
    return scaled_data,scaler

rmse =0
st.subheader('Predicting Next 30 days Close Price for :-'+ticker)
close_price = get_data(ticker)
rolling_price = get_rollng_mean(close_price)

differencing_order= get_differencing_order(rolling_price)

def fit_model(data,differencing_order):
    model =ARIMA(data,order=(7,differencing_order,7))
    model_fit =model.fit()
    forcast_steps=30
    forcast = model_fit.get_forecast(steps=forcast_steps)
    predictions = forcast.predicted_mean
    return predictions

def evaluate_model(original_price,differencing_order):
    train_data, test_data =original_price[:-30], original_price[-30:]
    predictions = fit_model(train_data,differencing_order)
    rmse= np.sqrt(mean_squared_error(test_data,predictions))
    return round(rmse,2)

scaled_data,scaler=scaling(rolling_price)
rmse = evaluate_model(scaled_data,differencing_order)

def get_forecast(orignal_price,differencing_order):
    predictions = fit_model(orignal_price,differencing_order)
    start_date = datetime.now().strftime('%Y-%m-%d')
    end_date = (datetime.now() + timedelta(days=29)).strftime('%Y-%m-%d')
    forecast_index = pd.date_range(start=start_date,end=end_date,freq='D')
    forecast_df= pd.DataFrame(predictions,index=forecast_index,columns= ['Close'])
    return forecast_df

def inverse_scaling(scaler,scaled_data):
    close_price = scaler.inverse_transform(np.array(scaled_data).reshape(-1,1))
    return close_price


st.write("**Model RMSE Score**",rmse)

forecast= get_forecast(scaled_data,differencing_order)

forecast['Close']= inverse_scaling(scaler,forecast['Close'])

st.write('#### Forecast Data (Next 30 days)')

df_1=forecast.sort_index(ascending=True).round(3)
st.dataframe(df_1.style.set_properties(**{'font-size': '20px'}),width=1000)

forecast = pd.concat([rolling_price,forecast])


# def Moving_average_forecast(forecast):
#     # Ensure enough data exists
#     if len(forecast) <= 30:
#         return "Not enough data to plot."

#     # Convert index to datetime if needed
#     if not isinstance(forecast.index, pd.DatetimeIndex):
#         forecast.index = pd.to_datetime(forecast.index)

#     fig = go.Figure()

#     # First trace: Past close prices
#     if len(forecast.index[:-30]) > 0:
#         fig.add_trace(go.Scatter(
#             x=forecast.index[:-30],
#             y=forecast['Close'].iloc[:-30],
#             mode='lines',
#             name='Close Price',
#             line=dict(width=2, color='blue')
#         ))
#     else:
#         st.warning("No data available for past close prices.")

#     # Second trace: Future close prices
#     if len(forecast.index[-31:]) > 0:
#         fig.add_trace(go.Scatter(
#             x=forecast.index[-31:],
#             y=forecast['Close'].iloc[-31:],
#             mode='lines',
#             name='Future Close Price',
#             line=dict(width=2, color='red')
#         ))
#     else:
#         st.warning("No data available for future close prices.")

#     fig.update_layout(
#         title="Moving Average Forecast",
#         xaxis_title="Date",
#         yaxis_title="Close Price",
#         height=500,
#         plot_bgcolor='white',
#         paper_bgcolor='#E1EFFF',
#         legend=dict(yanchor='top', xanchor='right')
#     )

#     return fig

# # Plot with debug checks
# fig = Moving_average_forecast(forecast.iloc[150:])
# if isinstance(fig, str):
#     st.warning(fig)
# else:
#     st.plotly_chart(fig, use_container_width=True)
