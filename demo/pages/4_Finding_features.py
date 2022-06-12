import streamlit as st
import pandas as pd

st.write('# Finding features')

st.write('### Sectors analysis')

from PIL import Image
image = Image.open('./img/4_sector_boxplot.png')
st.image(image)

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
st.table(df)

image = Image.open('./img/6_aapl_rsi.png')
st.image(image)

#SMA
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