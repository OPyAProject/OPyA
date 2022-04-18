#library import
import pandas as pd
import bt
import datetime
import numpy as np
from scipy.optimize import minimize

#Creation of a ticker_list_100 with format 'ticker1,ticker2,...,tickerN'

df=pd.read_csv('/data/sp500_100first.csv')
ticker_list_100=df.iloc[0][0].lower()
for index in df.index[1:]:
    ticker_list_100+=','+df.iloc[index][0].lower()

#Creation of data100 dataset
data100 = bt.get(ticker_list_100, start='2017-01-01')
data100.head(10)