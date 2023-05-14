
import streamlit as st
from datetime import date
from tweepy import models

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import matplotlib.pyplot as plt
# import os

import pandas as pd 
import numpy as np
import requests
import tweepy
import config 


import logging
logger = logging.getLogger('fbprophet')
logger.setLevel(logging.DEBUG)

m = Prophet()
print(m.stan_backend)

import pkg_resources
model_file = pkg_resources.resource_filename('fbprophet', 'stan_model/prophet_model.pkl')
print(model_file)

import pickle
with open(model_file, 'rb') as f:
    stan_model = pickle.load(f)


#connect to tweepy
#auth = tweepy.OAuthHandler(config.TWITTER_CONSUMER_KEY, 
#config.TWITTER_CONSUMER_SECRET)
#auth.set_access_token(config.TWITTER_ACCESS_TOKEN, 
#/Users/jorgemelchor/Desktop/myadd #config.TWITTER_ACCESS_TOKEN_SECRET)

#get the api tweepy social media link 
#api = tweepy.API(auth)

#this is the title 
st.title("Financial Stocks")
# st.header("Welcome Financial Stocks ")

#writting the title 
st.write("""
#Google.
Open stocks in Google.

""")
#PREDICTION SECTION 
#Start day data 
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

#creating the seacrh 
st.title("Financial Prediction")
stocks = ('BAC','MRNA','BLK','ACB','SPWR','TSLA','AAPL','MSFT','BTTC-USD','ETH-USD','PEP')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider("Years of prediction:", 1 ,4)
period = n_years * 365

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...done!")

st.subheader('Raw data')
st.header('Raw data {}'.format(selected_stock))
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

#prediction using fbProphet()
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns ={"Date": "ds","Close": "y"})

# #see the data 
# st.write(df_train)

# #plot the data
# df_train.set_index('ds').y.plot()
# st.write(df_train.tail())


#running Prophet
m = Prophet()
m.fit(df_train)
future =m.make_future_dataframe(periods=period)
forecast = m.predict(future)

#Show and plot forecast 
st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

#####

# ticker_input = st.text_input('Please enter your company ticker:')
# search_button = st.button('Search')

# if search_button:
#     tickerData = yf.Ticker(ticker_input)
           


# #define the ticker symbol
# tickerSymbol = 'GOOGL'

# #get data on this ticker
# tickerData = yf.Ticker(tickerSymbol)
# #get the historical prices for this ticker
# tickerDf = tickerData.history(period='1d', start='2010-5-31', end='2021-5-31')
# #Open	High	Low	Close	Volume	Dividends	Stock Splits


# # df = st.dataframe(tickerDf)
# # df['100ma'] = df['Close'].rolling(window=100, min_periods=0).mean()
# # df.head()

# st.line_chart(tickerDf.Open)
# st.line_chart(tickerDf.Close)


#Graphic Chart
# st.line_chart(tickerDf.Open)
# st.dataframe(tickerDf)

# st.beta_expander('Profit margins (as of {})'.format(tickerData))



# profit_margin_df = pd.DataFrame(tickerData, index = open)
# st.table(open)
# st.bar_chart()


 # st.subheader('Financial Data')
option = st.sidebar.selectbox("Dasboard",('stocktwits','Walk street', 'chart'))
st.header(option)

st.write("""
# Twitter community 
What the community say about the market.

""")

# tickerSymbol = 'AAPL'
# #get data on this ticker
# tickerData = yf.Ticker(tickerSymbol)
# #get the historical prices for this ticker
# tickerDf = tickerData.history(period='1d', start='2010-5-31', end='2021-5-31')
# # Open	High	Low	Close	Volume	Dividends	Stock Splits

# st.line_chart(tickerDf.Open)
# st.line_chart(tickerDf.Volume)

#if option == 'twitter':
#    for username in config.TWITTER_USERNAMES:
#        user = api.get_user(username)
#        tweets = api.user_timeline(username)

#       st.subheader(username)
#       st.image(user.profile_image_url)
       
#        for tweet in tweets:
#           if '$' in tweet.text:
#               words = tweet.text.split(' ')
#               for word in words:
#                   if word.startswith('$') and word[1:].isalpha():
#                       symbol = word[1:]
#                       st.write(symbol)
#                       st.write(tweet.text)
#                       st.image(f"https://finviz.com/chart.ashx?t={symbol}")



if option == 'stocktwits':
    symbol = st.sidebar.text_input("Symbol", value='AAPL', max_chars=5)

    r = requests.get(f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json")

    data = r.json()

    for message in data['messages']:
        st.image(message['user']['avatar_url'])
        st.write(message['user']['username'])
        st.write(message['created_at'])
        st.write(message['body'])


