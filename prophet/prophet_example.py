import numpy as np
import pandas as pd

from enum import Enum

#bt
import bt
import sklearn.manifold
import sklearn.cluster
import sklearn.covariance
from scipy.optimize import minimize
import scipy.stats
from pypfopt.expected_returns import mean_historical_return, ema_historical_return, capm_return
from prophet import Prophet
from datetime import datetime
import warnings

import matplotlib.pyplot as plt
import matplotlib as mp
import seaborn as sns

from prophet.plot import plot_plotly, plot_components_plotly

print("#################################################")
print("# TIME SERIES FORECASTING WITH FACEBOOK PROPHET LIBRARY ")
print("#################################################")

PATH_DATA = 'data/'

PERIOD_START = '2019-01-01'
PERIOD_END = '2022-04-30'

FILE_NAME_TICKER_LIST = PATH_DATA + "sp500_100first.csv"
FILE_NAME_TICKER_DATA = PATH_DATA + 'data100_'+ PERIOD_START +'_'+PERIOD_END+'.csv'


# GET THE TICKERS FOR SP500 100ST STOCKS
df=pd.read_csv(FILE_NAME_TICKER_LIST)

ticker_list_100=df.iloc[0][0].lower()
for index in df.index[1:]:
    ticker_list_100+=','+df.iloc[index][0].lower()

print('#################################################')
print('# GET LAST SAVED STOCK DATA')
print('#################################################')
stocks_dataset =pd.read_csv(FILE_NAME_TICKER_DATA,index_col=0,header=0,parse_dates=True)
#stocks_dataset =pd.read_csv("data/workspace_files/data100.csv",index_col=0,header=0,parse_dates=True)

stocks_dataset.head()

from prophet.plot import plot_cross_validation_metric 
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation



ticker_list_for_forecast = ['aapl']
data_to_forecast = stocks_dataset
#we forecast Ajd Close Prices for the example

ticker_models = {}
ticker_cross_validations = {}
ticker_performances ={}
ticker_forecasts = {}
forecast_dataset = pd.DataFrame()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for i,ticker in enumerate(ticker_list_for_forecast):
        ticker_data=data_to_forecast[ticker].to_frame()
        ticker_data.reset_index(inplace=True)
        
        ticker_data.columns=["ds","y"]

        nb_obs = len(ticker_data)
        
        train_data, test_data = ticker_data[:int(nb_obs*0.7)], ticker_data[int(nb_obs*0.3):]
        
        # Creation of Prophet model with default parameters, but adding US Country Holidays
        ticker_models[ticker] = Prophet()
        ticker_models[ticker].add_country_holidays(country_name='US')
        ticker_models[ticker].fit(train_data)
        #future = ticker_models[ticker].make_future_dataframe(periods=int(nb_obs*0.1))
        #forecast = ticker_models[ticker].predict(future)
        
        #applying predict to the last 30% dates and get the 70% fitted in sample datas in a single dataframe
        ticker_forecast = ticker_models[ticker].predict(ticker_data)
        ticker_forecasts[ticker] = ticker_forecast
        
        #we save all forecast in a single dataframe for all stocks selected
        if (i==0):
            forecast_dataset['ds'] = ticker_forecast['ds']
        forecast_dataset[ticker] = ticker_forecast['yhat']
        
        # Display graphic of Actual vs Predicted values
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(15, 5))
        
        fig.suptitle("Actual Vs Forecast for " + ticker)
        #using Prophet plotting
        ticker_models[ticker].plot(ticker_forecast,ax=ax1)

        #using standard plotting
        ticker_data.plot(x='ds',y='y',ax=ax2)
        ticker_forecast.plot(x='ds',y='yhat', ax=ax2)
        
        #display prophet decomposition of the series (trend, seasonnalities, holidays)
        ticker_models[ticker].plot_components(ticker_forecast)
        
        #get and display performance metrics of the model (cross-validation adding 30 days by fold, and prediction horizon of 30 days)
        ticker_cross_validations[ticker] = cross_validation(ticker_models[ticker], period='30 days', horizon = '30 days')
        
        ticker_performances[ticker] = performance_metrics(ticker_cross_validations[ticker])
        
        #display rmse and mae metrics
        fig = plot_cross_validation_metric(ticker_cross_validations[ticker], metric='rmse')
        fig = plot_cross_validation_metric(ticker_cross_validations[ticker], metric='mae')
        
        plt.show();
    
    forecast_dataset.set_index('ds')


ticker_cross_validations['aapl'].groupby('cutoff').count()


ticker_cross_validations['aapl'][ticker_cross_validations['aapl']['cutoff'] == '2020-01-06']


ticker_cross_validations['aapl'][ticker_cross_validations['aapl']['cutoff'] == '2021-03-31']

plot_plotly(ticker_models['aapl'], ticker_forecasts['aapl'])

plot_components_plotly(ticker_models['aapl'], ticker_forecasts['aapl'])

ticker_performances['aapl']

