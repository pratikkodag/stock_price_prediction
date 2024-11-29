import dateutil.relativedelta
import streamlit as st 
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import datetime
import dateutil
import ta
import plotly.express as px
import pandas_ta as pta
import numpy as np
npNaN = np.nan

st.set_page_config(
    page_title="Stock Analysis",
    page_icon="page_with_curl",
    layout='wide'
)

st.title("Stock Analysis")

col1,col2,col3 =st.columns(3)

today= datetime.date.today()

with col1:
    ticker =st.text_input("Stocker Ticker","TSLA")
with col2:
    start_date = st.date_input("choose Start Date",datetime.date(today.year -1,today.month,today.day))
with col3:
    end_date = st.date_input("choose End Date",datetime.date(today.year,today.month,today.day))
     
st.subheader(ticker)
stock = yf.Ticker(ticker)
try:
    info = stock.get_info()
    st.write(stock.info['longBusinessSummary'])
    st.write("**Sector:**",stock.info['sector'])
    st.write("**Full Time Employees:**",stock.info['fullTimeEmployees'])
    st.write("**Website:**",stock.info['website'])
except Exception as e:
    print(f"Error: {e}")
    
col1= st.columns(1)[0]
with col1:
    df = pd.DataFrame({
    "Metric": ["Market Cap", "Beta", "EPS", "PE Ratio","profitMargins",'revenuePerShare','financialCurrency'],
    "Value": [
        stock.info['marketCap'],
        stock.info['beta'],
        stock.info['trailingEps'],
        stock.info['trailingPE'],
        stock.info['profitMargins'],
        stock.info['revenuePerShare'],
        stock.info['financialCurrency']
        
        
     ]
})
    st.dataframe(df.style.set_properties(**{'font-size': '20px'}),width=1000)

data = yf.download(ticker,start=start_date,end= end_date)

col1,col2,col3 = st.columns(3)
daily_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
col1.metric("Daily Change",str(round(data['Close'].iloc[-1],2)),str(round(daily_change,2)))

# Display historical data based on start and end date
filtered_data = data[(data.index >= pd.to_datetime(start_date)) & (data.index <= pd.to_datetime(end_date))]

st.title(f"Historical Data ({start_date} to {end_date})")
st.dataframe(filtered_data.sort_index(ascending=False).round(3).style.set_properties(**{'font-size': '20px'}), width=1000)

col1,col2,col3,col4,col5,col6,col7=st.columns([1,1,1,1,1,1,1])

num_period=''
with col1:
    if st.button("5D"):
        num_period="5d"
with col2:
    if st.button("1M"):
        num_period="1mo"
with col3:
    if st.button("6M"):
        num_period="6mo"
with col4:
    if st.button("YTD"):
        num_period='ytd'
with col5:
    if st.button("1Y"):
        num_period='1y'
with col6:
    if st.button("5Y"):
        num_period='5y'
with col7:
    if st.button("MAX"):
        num_period= 'max'

##  
def filter_data(dataframe,num_period):
    if num_period =='1mo':
        date = dataframe.index[-1] + dateutil.relativedelta.relativedelta(months=-1)
    elif num_period == '5d':
        date =dataframe.index[-1] + dateutil.relativedelta.relativedelta(days=-5)
    elif num_period == '6mo':
        date =dataframe.index[-1] + dateutil.relativedelta.relativedelta(months=-6)
    elif num_period == '1y':
        date =dataframe.index[-1] + dateutil.relativedelta.relativedelta(years=-1)
    elif num_period == '5y':
        date =dataframe.index[-1] + dateutil.relativedelta.relativedelta(years=-5)
    elif num_period == 'ytd':
        date = datetime.datetime(dataframe.index[-1].year,1,1).strftime("%Y-%m-%d")
    else:
        date = dataframe.index[0]
    dataframe_reset = dataframe.reset_index()  # Reset index to create a 'Date' column
    return dataframe_reset[dataframe_reset['Date'] > date]

def close_chart(dataframe,num_period=False):
    if num_period:
        dataframe =filter_data(dataframe,num_period)
    fig =go.Figure()
    fig.add_trace(go.Scatter(
        x=dataframe['Date'],y= dataframe['Open'],
        mode='lines',
        name ='Open',line =dict(width=2,color='#5ab7ff')
    ))
    fig.add_trace(go.Scatter(
        x=dataframe['Date'],y= dataframe['Close'],
        mode='lines',
        name ='Close',line =dict(width=2,color='black')
    ))
    fig.add_trace(go.Scatter(
        x=dataframe["Date"],y= dataframe['High'],
        mode='lines',
        name ='High',line =dict(width=2,color='#0078ff')
    ))
    
    fig.add_trace(go.Scatter(
        x=dataframe["Date"],y= dataframe['Low'],
        mode='lines',
        name ='Low',line =dict(width=2,color='red')
    ))
    fig.update_xaxes(rangeslider_visible = True)
    fig.update_layout(height=500,margin=dict(l=0,r=20,t=20,b=0), plot_bgcolor='white',
                      paper_bgcolor ='#E1EFFF', legend=dict(yanchor='top',
                      xanchor='right'))
    return fig


def candlestick(dataframe,num_period):
    dataframe = filter_data(dataframe,num_period)
    fig=go.Figure()
    fig.add_trace(go.Candlestick(x=dataframe['Date'],
                   open =dataframe['Open'],high=dataframe['High'],
                   low = dataframe['Low'],close=dataframe['Close']              
        ))
    fig.update_layout(xaxis_title='Date',
        yaxis_title='Price',showlegend=False,height =500,
        margin=dict(l=0,r=20,t=20,b=0),plot_bgcolor ='white',
        paper_bgcolor ='#E1EFFF')
    return fig

def RSI(dataframe,num_period):
    dataframe['RSI']= pta.rsi(dataframe['Close'])
    dataframe = filter_data(dataframe,num_period)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dataframe['Date'],
        y=dataframe.RSI,name='RSI',marker_color='orange',line= dict(width=2,color ='orange'),
        ))
    fig.add_trace(go.Scatter(
        x=dataframe['Date'],
        y=[70]*len(dataframe),name='Overbought',marker_color='red',line= dict(width=2,color ='red',dash='dash'),
        ))
    fig.add_trace(go.Scatter(
        x=dataframe['Date'],
        y=[30]*len(dataframe),fill='tonexty',name='Oversold',marker_color='#79da84',
        line= dict(width=2,color ='#79da84',dash='dash')
        ))
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='RSI',
        height=500,
        margin=dict(l=0, r=20, t=20, b=0),
        plot_bgcolor='white',
        paper_bgcolor='#E1EFFF'
    )
    return fig


def Moving_average(dataframe,num_period):
    dataframe['SMA_50']=pta.sma(dataframe['Close'],50)
    dataframe = filter_data(dataframe,num_period)
    if dataframe['SMA_50'].isna().sum() > 0:
        st.warning("Not enough data to calculate the moving average.")
    fig=go.Figure()
    fig.add_trace(go.Scatter(
        x=dataframe['Date'],y= dataframe['Open'],
        mode='lines',
        name ='Open',line =dict(width=2,color='#5ab7ff')
    ))
    fig.add_trace(go.Scatter(
        x=dataframe['Date'],y= dataframe['Close'],
        mode='lines',
        name ='Open',line =dict(width=2,color='black')
    ))
    fig.add_trace(go.Scatter(
        x=dataframe["Date"],y= dataframe['High'],
        mode='lines',
        name ='Open',line =dict(width=2,color='#0078ff')
    ))
    
    fig.add_trace(go.Scatter(
        x=dataframe["Date"],y= dataframe['Low'],
        mode='lines',
        name ='Open',line =dict(width=2,color='red')
    ))
    fig.add_traces(go.Scatter(
        x=dataframe["Date"],y= dataframe['SMA_50'],
        mode='lines',
        name ='SMA 50',line =dict(width=2,color='purple')
    ))
    fig.update_xaxes(rangeslider_visible = True)
    fig.update_layout(height=500,margin=dict(l=0,r=20,t=20,b=0),plot_bgcolor='white',paper_bgcolor ='#E1EFFF',legend=dict(
    yanchor='top',
    xanchor='right'))
    return fig


def MACD(dataframe,num_period):
    macd = pta.macd(dataframe['Close']).iloc[:,0]
    macd_signal = pta.macd(dataframe['Close']).iloc[:,1]
    macd_hist =pta.macd(dataframe['Close']).iloc[:,2]
    dataframe['MACD'] = macd
    dataframe['MACD-Signal']=macd_signal
    dataframe['MACD-Hist']=macd_hist
    dataframe = filter_data(dataframe,num_period)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dataframe['Date'],
        y=dataframe['MACD'],name='RSI',marker_color='orange',line=dict(width=2,color='orange'),
    ))
    fig.add_trace(go.Scatter(
        x=dataframe['Date'],
        y=dataframe['MACD-Signal'],name='Overbought',marker_color ='red',line =dict(width=2,color='red',dash='dash'),
    ))
    c=['red' if cl<0 else 'green' for cl in macd_hist]
    
    return fig
  
      
col1,col2,col3 = st.columns([1,1,4])

with col1:
    char_type =st.selectbox('',('Candle','line'))
with col2:
    if char_type == 'Candle':
        
        indicators = st.selectbox('',('RSI','MACD'))
    else:
        indicators = st.selectbox('',('RSI','Moving Average','MACD'))
        
ticker_ = yf.Ticker(ticker)
new_df1=ticker_.history(period='max')
data1= ticker_.history(period='max')
if num_period == '':
    
    if char_type == 'Candle' and indicators == 'RSI':
        st.plotly_chart(candlestick(data1,'1y'),use_container_width=True)
        st.plotly_chart(RSI(data1,'1y'),use_container_width=True)
        
    if char_type == 'Candle' and indicators == 'MACD':
        st.plotly_chart(candlestick(data1,'1y'),use_container_width=True)
        st.plotly_chart(MACD(data1,'1y'),use_container_width=True)
        
    if char_type == 'line' and indicators == 'RSI':
        st.plotly_chart(close_chart(data1,'1y'),use_container_width=True)
        st.plotly_chart(RSI(data1,'1y'),use_container_width=True)
        
    if char_type == 'line' and indicators == 'Moving Average':
        st.plotly_chart(Moving_average(data1,'1y'),use_container_width=True)
        
    if char_type == 'line' and indicators == 'MACD':
        st.plotly_chart(close_chart(data1,'1y'),use_container_width=True)
        st.plotly_chart(MACD(data1,'1y'),use_container_width=True)
        
else:
    if char_type == 'Candle' and indicators == 'RSI':
        st.plotly_chart(candlestick(new_df1,num_period),use_container_width=True)
        st.plotly_chart(RSI(new_df1,num_period),use_container_width=True)
        
    if char_type == 'Candle' and indicators == 'MACD':
        st.plotly_chart(candlestick(new_df1,num_period),use_container_width=True)
        st.plotly_chart(MACD(new_df1,num_period),use_container_width=True)
        
    if char_type == 'line' and indicators == 'RSI':
        st.plotly_chart(close_chart(new_df1,num_period),use_container_width=True)
        st.plotly_chart(RSI(new_df1,num_period),use_container_width=True)
        
    if char_type == 'line' and indicators == 'Moving Average':
        st.plotly_chart(Moving_average(new_df1,num_period),use_container_width=True)
        
    if char_type == 'line' and indicators == 'MACD':
        st.plotly_chart(close_chart(new_df1,num_period),use_container_width=True)
        st.plotly_chart(MACD(new_df1,num_period),use_container_width=True)
        
    
