import streamlit as st
import pandas as pd

st.write('# Finding features')

st.write('### Sectors analysis')

st.write('The companies present in the dataset are distributed among 11 sectors.')

from PIL import Image
image = Image.open('./img/3_sector_repartition.png')
st.image(image)

st.write("Information Technology has the best performance, but also an important volatility shown by the large size of the box.")

image = Image.open('./img/4_sector_boxplot.png')
st.image(image)

st.write("The extreme values on the Energy boxplot are explained by an important rise in 2022.")

image = Image.open('./img/5_energy_plot.png')
st.image(image)


#import du csv avec 500 valeurs du SP500
#colonnes (ticker,name,sector,subsector)
sectors=pd.read_csv('./csv/sp500_sectors.csv',encoding='cp1252')
sectors['ticker']=sectors['ticker'].map(lambda x:x.lower())
sectors.index = sectors.index.map(str)
sectors=sectors.set_index('ticker')
#import données
data100=pd.read_csv('./csv/data100_09042022.csv',index_col=0,header=0,parse_dates=True)

tickers_by_sector={k:[] for k in sectors['sector'].unique()}
for ticker in data100.columns.to_list():
    tickers_by_sector[sectors.loc[ticker]['sector']].append(ticker)

df_by_sector={k:pd.DataFrame() for k in sectors['sector'].unique()}
for key in tickers_by_sector:
    for ticker in tickers_by_sector[key]:
        df_by_sector[key][ticker]=data100[ticker]

coefficients={}
coefficients['global']=data100.corr().mean().mean()

for key in df_by_sector:
    coefficients[key]=df_by_sector[key].corr().mean().mean()

df=pd.DataFrame.from_dict([coefficients]).T
df=df.rename({0:'Correlation coefficient'},axis=1)

st.write('The global average of the correlation coefficients is 0.68, which shows that the global market is rising.')
st.write('Only 2 sectors have a lower correlation coefficient. For our models, we will incorporate when possible the sectors values as covariates to help the models.')

st.table(df)

#RSI
st.write('## Relative Strength Index (RSI)')
st.write('The relative strength index (RSI) is a momentum indicator used in technical analysis that measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of a stock or other asset.')
st.write('The below example shows an application on the Apple and Microsoft stocks. When the rsi is low, we can expect a rise because the stock is likely to be under evaluated.')
st.write('On the contrary, when the rsi is high, we can expect a fall because the stock is likely to be over evaluated.')
st.write('This metric can be useful for return prediction.')

image = Image.open('./img/6_aapl_rsi.png')
st.image(image)


#SMA
st.write('## Simple Moving Average (SMA) and Exponential Moving Average (EMA)')
from sklearn.metrics import mean_squared_error 
tickerlist=['aapl']
for ticker in tickerlist:
    df_s = data100[ticker].to_frame()
    df_s['20D-SMA'] = df_s[ticker].rolling(window=20).mean()
    df_s['50D-SMA'] = df_s[ticker].rolling(window=50).mean()
    df_s['100D-SMA'] = df_s[ticker].rolling(window=100).mean()
    df_s=df_s.dropna()
    MSE_20D_SMA=mean_squared_error(df_s[ticker],df_s['20D-SMA'])
    MSE_50D_SMA=mean_squared_error(df_s[ticker],df_s['50D-SMA'])
    MSE_100D_SMA=mean_squared_error(df_s[ticker],df_s['100D-SMA'])

#EMA
tickerlist=['aapl']
for ticker in tickerlist:
    df_s = data100[ticker].to_frame()
    df_s['20D-EMA'] = df_s[ticker].ewm(span=20,adjust=False).mean()
    df_s['50D-EMA'] = df_s[ticker].ewm(span=50,adjust=False).mean()
    df_s['100D-EMA'] = df_s[ticker].ewm(span=100,adjust=False).mean()
    MSE_20D_EMA=mean_squared_error(df_s[ticker],df_s['20D-EMA'])
    MSE_50D_EMA=mean_squared_error(df_s[ticker],df_s['50D-EMA'])
    MSE_100D_EMA=mean_squared_error(df_s[ticker],df_s['100D-EMA'])


image = Image.open('./img/7_aapl_sma.png')
st.image(image)
st.write(f"MSE_20D_SMA : {MSE_20D_SMA}")
st.write(f"MSE_50D_SMA : {MSE_50D_SMA}")
st.write(f"MSE_100D_SMA : {MSE_100D_SMA}")

image = Image.open('./img/8_aapl_ema.png')
st.image(image)
st.write(f"MSE_20D_EMA : {MSE_20D_EMA}")
st.write(f"MSE_50D_EMA : {MSE_50D_EMA}")
st.write(f"MSE_100D_EMA : {MSE_100D_EMA}")