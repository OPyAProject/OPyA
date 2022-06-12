# -*- coding: utf-8 -*-

# -- Sheet --

# # OPyA project : Online Portfolio Allocation benchmarking with Python
# 
# *Data Scientist promo Sep21 - Sandra CHARLERY, Maxime VANPEENE*


#general
import gc
import os
import copy

#data science / calculation
import pandas as pd
import bt
import datetime
import numpy as np
from scipy.optimize import minimize

#pyportfolioopt
import pypfopt

#plotting
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from pprint import pprint
import seaborn as sns
sns.set_style("whitegrid")
sns.set(rc={'figure.figsize':(15.7,6)})
from IPython.core.display import display, HTML
%matplotlib inline

#warnings
import warnings
warnings.filterwarnings("ignore")

#Création d'une string ticker_list_100 sous le format 'ticker1,ticker2,...,tickerN'

#Lecture du CSV avec tous les tickers concernés
df=pd.read_csv('/data/workspace_files/sp500_100first.csv')
#initialisation boucle
ticker_list_100=df.iloc[0][0].lower()
#ajout des tickers suivants
for index in df.index[1:]:
    ticker_list_100+=','+df.iloc[index][0].lower()

#Création du fichier de données data100.csv en utilisant bt

#data100 = bt.get(ticker_list_100, start='2017-01-01')
#data100.head(10)
#data100.to_csv('/data/workspace_files/data100_09042022.csv')

#Utilities
def display_dataframe(df, title = ""):
    #tdstring = f'<td style="text-align: left; vertical-align: middle; font-size:1.2em;">{v}</td>'
    if (title != ""):
        text = f'<h2>{title}</h2><table><tr>'
    else:
        text = '<table><tr>'
    text += ''.join([f'<td style="text-align: left; vertical-align: middle; font-size:1.2em;"><b>{col}</b></td>' for col in df.columns.values]) + '</tr>'
    for row in df.itertuples():
        #text +=  '<tr>' + ''.join([f'<td valign="top">{v}</td>' for v in row[1:]]) + '</tr>'
        text +=  '<tr>' + ''.join([ f'<td style="text-align: left; vertical-align: middle; font-size:1.1em;">{v}</td>' for v in row[1:]]) + '</tr>'
    text += '</table>'
    display(HTML(text))

def rsi(series, period):
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
    u = u.drop(u.index[:(period-1)])
    d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
    d = d.drop(d.index[:(period-1)])
    rs = pd.DataFrame.ewm(u, com=period-1, adjust=False).mean() / \
         pd.DataFrame.ewm(d, com=period-1, adjust=False).mean()
    return 100 - 100 / (1 + rs)

def rsi_class(x):
    ret = "low"
    if x < 50:
        ret = "low"
    if x > 50:
        ret = "med"
    if x > 70:
        ret = "hi"
    return(ret)

# # Project introduction
# The financial and banking world is one of the sectors that generates the most data and is most often freely accessible. The portfolio allocation domain is particularly interesting in the sense that it is abundant in data and can be systematized.
# 
# The objective of this project is to create a portfolio allocation model that adapts its strategy online.
# 
# We decided to focus on the 100 largest capitalizations of the New York Stock Exchange.


# # Data import
# 
# The dataset contains the 100 largest capitalizations close values of the New York Stock Exchange at the start of the project.<br>
# Each action is represented by a 'ticker' which is a unique code for each action.<br>
# The dataset is obtained using the bt library and its .get method based on the **Yahoo Finance API**.<br>
# By default, .get (alias for ffn.get) downloads the **Adjusted Close**.<br>
# The adjusted closing price considers other factors like dividends, stock splits, and new stock offerings.
# It is a more accurate measure of stocks' value.


data100=pd.read_csv('/data/workspace_files/data100_09042022.csv',index_col=0,header=0,parse_dates=True)
lst_data = []
lst_data.append({"Item": "No. of Tickers(columns)", "Value": len(data100.columns.to_list())})
lst_data.append({"Item": "No. of Observations(rows)", "Value": (data100.shape[0])})
lst_data.append({"Item": "Start date", "Value": (data100.index.to_list()[0].date())})
lst_data.append({"Item": "End date", "Value": (data100.index.to_list()[-1].date())})
#lst_data.append({"Item": "Data availability for", "Value": "1202 Days"})
#lst_data.append({"Item": "No. of Securities with full availability", "Value": "1865"})
df_display = pd.DataFrame(data = lst_data)
display_dataframe(df_display, "Basic Statistics of Stock Data")

# # Data cleaning


# The data is extracted from an API so the dataset is clean.


lst_data = []
lst_data.append({"Item": "No. of NaN values", "Value": data100.isna().sum().sum()})
lst_data.append({"Item": "No. of Duplicates", "Value": data100.duplicated().sum()})
lst_data.append({"Item": "Index format", "Value": "DatetimeIndex"})
lst_data.append({"Item": "Value format", "Value": "float64"})
#lst_data.append({"Item": "Data availability for", "Value": "1202 Days"})
#lst_data.append({"Item": "No. of Securities with full availability", "Value": "1865"})
df_display = pd.DataFrame(data = lst_data)
display_dataframe(df_display, "Basic Statistics of Stock Data")

# The following boxplot shows that there is no visible outlier


data100.boxplot();
plt.xticks(rotation=90);
fig=plt.gcf()
fig_width, fig_height = plt.gcf().get_size_inches()
fig.savefig('all_stocks_boxplot',format='png',bbox_inches='tight')
print(fig_width, fig_height)

# <img src="all_stocks_boxplot" width=885 height=400>


# # Data exploration


# ## Market global tendency


# Let's display a global view of all the stocks to detect trends


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
plt.rcParams['axes.grid'] = True
%matplotlib inline

# define subplot grid
fig, axs = plt.subplots(nrows=20, ncols=5, figsize=(15, 30))
fig.suptitle("Daily closing prices", fontsize=18)

# loop through tickers and axes
for ticker, ax in zip(data100.columns.tolist(), axs.ravel()):
    # filter df for ticker and plot on specified axes
    data100[ticker].plot(ax=ax)

    # chart formatting
    ax.set_title(ticker.upper())
    ax.set_xlabel("")
    ax.xaxis.set_major_locator(mdates.YearLocator())
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig('all_stocks_subplots',format='png',bbox_inches='tight')
plt.show()

# <img src="all_stocks_subplots" width=800 height=1600>


# Over the studied time period, a global uptrend can be observed.<br>
# This is confirmed by simulating a naive portfolio with equal weights.<br>


average_index=data100.div(data100.iloc[0]/100).mean(axis=1)
average_index.plot(figsize=(16,9));
ax=plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
plt.title('Naive portfolio with equal weights, initial investment 100');
plt.grid(visible=True)
fig=plt.gcf()
fig.savefig('naive_portolio',format='png',bbox_inches='tight')

# <img src='naive_portolio'>


# # Finding features


# ## Sectors analysis


# The companies present in the dataset are distributed among 11 sectors. <br>


#import du csv avec 500 valeurs du SP500
#colonnes (ticker,name,sector,subsector)
sectors=pd.read_csv('/data/workspace_files/sp500_sectors.csv',encoding='cp1252')

#tickers en minuscules
sectors['ticker']=sectors['ticker'].map(lambda x:x.lower())
sectors.index = sectors.index.map(str)


#on filtre sur les 100 actions qui nous intéressent et on met le ticker comme index
sectors=sectors[sectors['ticker'].isin(data100.columns.to_list())]
sectors=sectors.set_index('ticker')

#Plotting
sectors.groupby('sector').\
    agg('count').\
    sort_values(
                by='name',
                ascending=False).\
    plot.pie(y='name',
            figsize=(16,9),
            title='100 Stocks sectors',
            autopct= lambda x: '{:.0f}'.format(x*sectors.shape[0]/100),
            legend=False,
            textprops={'fontsize': 12},
            pctdistance=0.9);
plt.axis('off');

"""
fig=plt.gcf()
fig=plt.gcf()
fig.savefig('sector_repartition',format='png',bbox_inches='tight')
"""

# <img src='sector_repartition'>


data100_by_sector=pd.DataFrame(0, index=data100.index, columns=sectors['sector'].unique())
for ticker in data100.columns.to_list():
    data100_by_sector[sectors.loc[ticker,'sector']]+=data100[ticker]
data100_by_sector_inv=data100_by_sector.div(data100_by_sector.iloc[0]/100)
order = data100_by_sector_inv.median().sort_values().index.to_list()
data100_by_sector_inv.boxplot(column=order,rot=90);
plt.title('Sectors portfolio values, initial investment 100');


fig=plt.gcf()
fig.savefig('sector_boxplot',format='png',bbox_inches='tight')

# Information Technology has the best performance, but also an important volatility shown by the large size of the box.


# <img src='sector_boxplot'>


# The extreme values on the Energy boxplot are explained by an important rise in 2022.


data100_by_sector_inv['Energy'].plot();
pl
fig=plt.gcf()
fig.savefig('energy_plot',format='png',bbox_inches='tight')

# <img src='energy_plot'>


# The global average of the correlation coefficients is 0.68, which shows that the global market is rising.<br>
# Only 2 sectors have a lower correlation coefficient. For our models, we will incorporate when possible the sectors values as covariates to help the models.


tickers_by_sector={k:[] for k in sectors['sector'].unique()}
for ticker in data100.columns.to_list():
    tickers_by_sector[sectors.loc[ticker]['sector']].append(ticker)

df_by_sector={k:pd.DataFrame() for k in sectors['sector'].unique()}
for key in tickers_by_sector:
    for ticker in tickers_by_sector[key]:
        df_by_sector[key][ticker]=data100[ticker]

print(f"Global average correlation coefficient : {data100.corr().mean().mean()}")
for key in df_by_sector:
    print(f"{key} : {df_by_sector[key].corr().mean().mean()}")

# ## Relative Strength Index (RSI)


# The relative strength index (RSI) is a momentum indicator used in technical analysis that measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of a stock or other asset.<br>
# The below example shows an application on the Apple and Microsoft stocks.
# When the rsi is low, we can expect a rise because the stock is likely to be under evaluated.
# On the contrary, when the rsi is high, we can expect a fall because the stock is likely to be over evaluated.
# This metric can be useful for return prediction.


sns.set(rc={'figure.figsize':(14.7,4)})
sns.set_style("whitegrid")
tickerlist = ['aapl','msft']
for ticker in tickerlist:
    df_stock = data100[ticker].to_frame()
    df_stock['rsi'] = rsi(series=data100[ticker], period=28)
    df_stock['rsicat'] = list(map(rsi_class, df_stock['rsi']))
    plt.title(f"Examining RSI on movement of Price for:{ticker}")
    ax = sns.scatterplot(x = df_stock.index, y = df_stock[ticker], hue = df_stock["rsicat"]);
    fig=plt.gcf()
    fig.savefig(f"{ticker}_rsi",format='png',bbox_inches='tight')
    plt.show()

# <img src='aapl_rsi'>


# <img src='msft_rsi'>


# ## Analyzing SMA (Simple Moving Averages) and EMA (Exponential Moving Averages)


# We are now going to analyze the SMAs(Simple Moving Averages) and the EMA(Exponential Moving Averages).
# The purpose of these metrics is to catch global tendencies.
# 
# SMA is a rolling average while EMA is an exponentially weighted average giving more weight to the most recent values.
# The below examples seem to show that the 50D timeframe is the best compromise because the curve catches the trends without too much overfitting.
# 
# A quick RMSE analysis shows that the EMA 50D is the most interesting.



sns.set_style("whitegrid")
sns.set(rc={'figure.figsize':(14.7,4)})
# sns.set_style("whitegrid")
from sklearn.metrics import mean_squared_error 
tickerlist=['aapl','msft']
for ticker in tickerlist:
    df_s = data100[ticker].to_frame()
    df_s['20D-SMA'] = df_s[ticker].rolling(window=20).mean()
    df_s['50D-SMA'] = df_s[ticker].rolling(window=50).mean()
    df_s['100D-SMA'] = df_s[ticker].rolling(window=100).mean()
    df_s=df_s.dropna()
    sns.set_style("whitegrid")
    df_s.plot(title = "SMA analysis for Security Code:" + str(ticker));
    fig=plt.gcf()
    fig.savefig(f"{ticker}_sma",format='png',bbox_inches='tight')
    print(mean_squared_error(df_s[ticker],df_s['50D-SMA']))

# <img src="aapl_sma">


# <img src="msft_sma">


sns.set_style("whitegrid")
sns.set(rc={'figure.figsize':(14.7,4)})
# sns.set_style("whitegrid")
from sklearn.metrics import mean_squared_error 
tickerlist=['aapl','msft']
for ticker in tickerlist:
    df_s = data100[ticker].to_frame()
    df_s['20D-EMA'] = df_s[ticker].ewm(span=20,adjust=False).mean()
    df_s['50D-EMA'] = df_s[ticker].ewm(span=50,adjust=False).mean()
    df_s['100D-EMA'] = df_s[ticker].ewm(span=100,adjust=False).mean()
    sns.set_style("whitegrid")
    df_s.plot(title = "EMA analysis for Security Code:" + str(ticker));
    fig=plt.gcf()
    fig.savefig(f"{ticker}_ema",format='png',bbox_inches='tight')
    print(mean_squared_error(df_s[ticker],df_s['50D-EMA']))



# <img src="aapl_ema">


# 


# <img src="msft_ema">


# # Strategies available for benchmarking


# ## Risk Parity
# 
# Risk parity is a portfolio allocation strategy that uses risk to determine allocations across various components of an investment portfolio. The risk parity strategy modifies the modern portfolio theory (MPT) approach to investing through the use of leverage.


# ## Momentum
# 
# Momentum investing is a system of buying stocks or other securities that have had high returns over the past three to twelve months, and selling those that have had poor returns over the same period.


# ## Markowitz portolio allocation method


# ### Method
# 
# The Markowitz portfolio allocation method was introduced in 1952 by Harry Markowitz, and is based on the expected return of the assets E(Ri) and the volatility.<br>
# <br>
# The volatility is assimilated to the sample standard deviation.<br>
# The expected return is the methods' main flaw as it is very difficult to predict asset returns.<br>
# 
# The Sharpe ratio is defined by the portfolio return divided by the volatility, it is then maximized.


# ### Example
# 
# For memory considerations, we limit the example to 3 stocks.
# We generate 10000 random portfolios and display their volatility and return on a graph.
# The upper frontier is called the 'efficient frontier', the portfolios on this frontier have the best Sharpe ratios.


import copy
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt import risk_models
from pypfopt import plotting

data3=data100.iloc[:,:3]

mu = mean_historical_return(data3)
S = CovarianceShrinkage(data3).ledoit_wolf()
ef=EfficientFrontier(mu,S)

fig, ax = plt.subplots()
ef_max_sharpe = copy.deepcopy(ef)
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False,linewidth=100)
for line in plt.gca().lines:
    line.set_linewidth(3)
    line.set_color('black')

# Find the tangency portfolio

ef_max_sharpe.max_sharpe()
ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
ax.scatter(std_tangent, ret_tangent, marker="*", s=500, c="r", label="Max Sharpe")

# Generate random portfolios
n_samples = 5000
w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
rets = w.dot(ef.expected_returns)
stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
sharpes = rets / stds
ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

# Output
ax.set_title("Efficient Frontier with random portfolios")
ax.legend()
plt.tight_layout()
plt.savefig("ef_scatter.png", dpi=200)
plt.show()

# <img src='ef_scatter.png'>


# ### Description of the different ways implemented to estimate the expected returns used in Markowitz Strategy


# #### Mean historical return
# The mean historical return of an asset  is its past rate of return and performance.


# #### EMA historical return (Exponential Moving Average)
# 
# The EMA is a kind of moving average that gives more importance to recent data.
# 
# ​**EMA=Return(t)×k+EMA(y)×(1−k)**
# 
# - **t**=today
# - **y**=yesterday
# - **N**=number of days in EMA
# - **k**=2÷(N+1)​


# #### CAPM return (Capital Asset Pricing Model)
# 
# The CAPM model gives the expected return of an asset based on its level of risk.
# 
# **E(Ra) = RFR + 𝛃a x (Rm -RFR)**
# 
# - **E(Ra)**: the expected return of the asset a that we want to calculate<br>
# - **RFR**: the risk free rate<br>
# - **𝛃a**: the beta of asset a<br>
# - **Rm**: the average profitability of the market<br>
# 
# According to the first part of the sum, **RFR**, the expected return of the asset will be at least equal to the risk-free rate RFR. 
# Any risky asset must provide a return greater than the risk-free rate according to the risk-return principle.
# The next term of the sum of the CAPM brings up two other notions: the **beta 𝛃** and the **profitability of the market Rm**.
# 
# The beta is the measure of the correlation between the profitability of the asset a and the market m, that is to say the trend of evolution of our asset compared to that of the market.
# The higher the beta, the more the asset and the market tend to move in the same direction. 
# 
# The closer the beta is to 0, the higher the independence of the asset from the market. 
# Whatever the evolution of the market, we will not be able to predict the evolution of the asset.
# 
# A negative beta means that our asset and the market are moving in opposite directions.
# 
# Beta is a measure of asset risk as it measures the volatility (up or down) of our asset relative to the market.
# 
# The next term of the sum, is the difference between the market return and the risk-free rate. 
# It is like the risk premium when investing in the market compared to a return without risk. This is the **market risk premium**.
# 
# The product of beta multiplied by the market risk premium gives a risk premium for the asset.
# The higher the beta, the more volatile our asset is relative to the market, and the higher the risk premium  will be.


# #### Predicting expected return with Facebook Prophet Machine Learning
# 
# Prophet is an open-source library developed by Facebook and designed for automatic forecasting of univariate time series data.
# 
# It is easy to use and designed to automatically find a good set of hyper parameters for the model to make forecasts for data with trends and seasonal structure by default.
# 
# The general idea of ​​the model is similar to a generalized additive model. The "prophet's equation" corresponds to trend, seasonality and holidays. This is given by,
# 
# **y(t)=g(t)+s(t)+h(t)+e(t)**
# 
# - y(t) is the forecast
# - g(t) refers to the trend (changes over a long period)
# - s(t) refers to seasonality (periodic or short-term changes)
# - h(t) refers to the effects of holidays on the forecast
# - e(t) refers to unconditional changes. It is called the error term.
# 
# Prophet is used here with the **piecewise linear model** to predict data (but it can also work with the logistic growth model if specified).
# 
# y = 𝛃0 + 𝛃1*x if x <= c
# 
# y = 𝛃0 - 𝛃2*c + (𝛃1 + 𝛃2)*x if x>c  
# 
# (c is the trend change point)
# 
# #### - Example of Prophet prediction with Apple stock ####
# 
# Prophet only predicts data when it is in a certain format.<br>
# The dataframe should have a column name '**ds**' in datetime format, and a column named '**y**' for the data to forecast.<br><br>
# The image on the left below is generated with Prophet and shows the basic prediction.<br>
# The light blue is the level of uncertainty ( yhat_upper and yhat_lower)<br> 
# The dark blue is the prediction ( yhat ), and the black dots are the original data.<br><br>
# 
# The image on the right simply plot actual versus forecast Apple stock prices.


from prophet import Prophet
from prophet.plot import plot_cross_validation_metric 
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation



ticker_list_for_forecast = ['aapl']
data_to_forecast = data100
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

plt.savefig("prophet_apple_forecast.png", dpi=200)
   


# ### Prophet forecast components
# 
# The images below show the trends and seasonalities of the time series data. We can see that there is an upward trend.


#display prophet decomposition of the series (trend, seasonnalities, holidays)
ticker_models[ticker].plot_components(ticker_forecast)

plt.savefig("prophet_apple_components.png", dpi=200)


# #### Prophet cross validation and metrics
# 
# Prophet includes functionality for time series cross validation to measure forecast error by comparing the predicted values with the actual values.
# 
# To apply the cross_validation function, we specify the forecast horizon (horizon), then optionally the size of the initial training period (initial) and the spacing between cutoff dates (period).
# 
# Here, we display the RMSE metric (Root Mean Squared Error), which increases as the forecast horizon increases.


#get and display performance metrics of the model (cross-validation adding 30 days by fold, and prediction horizon of 30 days)
ticker_cross_validations[ticker] = cross_validation(ticker_models[ticker], period='30 days', horizon = '30 days')

ticker_performances[ticker] = performance_metrics(ticker_cross_validations[ticker])
#display rmse and mae metrics
fig = plot_cross_validation_metric(ticker_cross_validations[ticker], metric='rmse')
        
fig.savefig("prophet_apple_rmse.png")

# # Results
# 
# To benchmark the different strategies, we use the bt library.<br>


"""
Mettre ici les données à afficher sous format DataFrame
Dates : 01/01/2021 au 30/04/2022 
"""

# #### Benchmark S&P 500


#SP500 DF 
PATH_DATA = '/data/workspace_files/'
df_benchmark = pd.read_csv(PATH_DATA+'benchmark.csv',index_col=0,parse_dates=True)
df_mean['sBenchmarkSP500'].plot()
plt.xlim(['1-1-2021','8-4-2022'])
plt.legend();
fig=plt.gcf()
fig.savefig(f"benchmark",format='png',bbox_inches='tight')

df_benchmark = pd.read_csv('benchmark.csv',index_col=0,parse_dates=True)

##############
#Stratégies hors Markowitz
##############

#Risk Parity
"""
s1 = bt.Strategy('Risk Parity 3m',[bt.algos.RunMonthly(),
                        bt.algos.RunAfterDate('1-1-2021'),
                        bt.algos.SelectAll(),
                        bt.algos.WeighERC(lookback=pd.DateOffset(months=3)),
                        bt.algos.Rebalance()])
test1 = bt.Backtest(s1, data100)
res=bt.run(test1)
res.plot();
df_risk_parity=res.prices
df_risk_parity=df_risk_parity.loc['1-1-2021':]
df_risk_parity.to_csv('risk_parity.csv')
"""
df_risk_parity=pd.read_csv('risk_parity.csv',index_col=0,parse_dates=True)

#Momentum

"""
s1 = bt.Strategy('Momentum 3m',[bt.algos.RunMonthly(),
                        bt.algos.RunAfterDate('1-1-2021'),
                        bt.algos.SelectAll(),
                        bt.algos.SelectMomentum(n=10,lookback=pd.DateOffset(months=3)),
                        bt.algos.WeighEqually(),
                        bt.algos.Rebalance()])
test1 = bt.Backtest(s1, data100)
res=bt.run(test1)
res.plot();
df_momentum=res.prices
df_momentum=df_momentum.loc['1-1-2021':]
df_momentum.to_csv('momentum.csv')
"""

df_momentum=pd.read_csv('momentum.csv',index_col=0,parse_dates=True)

# # Customization of bt implementation of Markowitz Strategy (WeighMeanVar)
# 
# The customized strategy (**OPAMarkowitz**) has been implemented in another notebook (see bt_expected_returns)
# 
# The aim is to :
# - configure estimation method for the expected return (Historical mean, EMA, CAPM, Forecast with Prophet)
# - take into account benchmark data in some cases (ex: CAPM)
# 
# Each diagram below shows the evolution of the Portfolio dynamic strategy with rebalancing weights monthly:
# - against the static strategy of holding position without rebalancing (*Once)
# - against the benchmark (S&P 500 Index)


#Mean historical return DF

df_mean = pd.read_csv(PATH_DATA+'markowitz_mean.csv',index_col=0,parse_dates=True)
df_mean['sBenchmarkSP500'].plot()
df_mean['sOPAMarkowitzMeanOnce'].plot()
df_mean['sOPAMarkowitzMean'].plot()
plt.xlim(['1-1-2021','8-4-2022'])
plt.legend();
fig=plt.gcf()
fig.savefig(f"markowitz_mean",format='png',bbox_inches='tight')

#EMA historical return DF
df_ema = pd.read_csv(PATH_DATA+'markowitz_ema.csv',index_col=0,parse_dates=True)
df_ema['sBenchmarkSP500'].plot()
df_ema['sOPAMarkowitzEMAOnce'].plot()
df_ema['sOPAMarkowitzEMA'].plot()
plt.xlim(['1-1-2021','8-4-2022'])
plt.legend();
fig=plt.gcf()
fig.savefig(f"markowitz_ema",format='png',bbox_inches='tight')

#CAPM DF ->Sandra
df_capm = pd.read_csv(PATH_DATA+'markowitz_capm.csv',index_col=0,parse_dates=True)
df_capm['sBenchmarkSP500'].plot()
df_capm['sOPAMarkowitzCAPMOnce'].plot()
df_capm['sOPAMarkowitzCAPM'].plot()
plt.xlim(['1-1-2021','8-4-2022'])
plt.legend();
fig=plt.gcf()
fig.savefig(f"markowitz_capm",format='png',bbox_inches='tight')

#Prophet DF
df_ml = pd.read_csv(PATH_DATA+'markowitz_ml.csv',index_col=0,parse_dates=True)
df_ml['sBenchmarkSP500'].plot()
df_ml['sOPAMarkowitzMLOnce'].plot()
df_ml['sOPAMarkowitzML'].plot()
plt.xlim(['1-1-2021','8-4-2022'])
plt.legend();
fig=plt.gcf()
fig.savefig(f"markowitz_ml",format='png',bbox_inches='tight')

# We see that **dynamic strategy** with rebalancing the weights of each stock is **more efficient**.


df_benchmark['sBenchmarkSP500'].plot()
df_mean['sOPAMarkowitzMean'].plot()
df_ema['sOPAMarkowitzEMA'].plot()
df_capm['sOPAMarkowitzCAPM'].plot()
df_ml['sOPAMarkowitzML'].plot()
df_risk_parity['Risk Parity 3m'].plot()
df_momentum['Momentum 3m'].plot()
plt.xlim(['1-1-2021','8-4-2022'])
plt.legend();
fig=plt.gcf()
fig.savefig(f"final_benchmark",format='png',bbox_inches='tight')

# <img src='final_benchmark'>


# # Conclusion
# 
# It is very difficult to predict markets.
# This study shows however that several dynamic allocation strategies can outperform the market on our dataset.<br>
# 
# We have not performed a strategy using the RSI but this metrics seems to give interesting results.
# Another lead would be to use deep learning models instead of Prophet. We made several trials on the Google Temporal Fusion Transformer (TFT), but due to computing limitations we were not able to get exploitable results.
# 
# To confirm the generalization of the strategies, other studies should be performed on other datasets with other trends.<br>
# Our models don't include transaction costs which will influence returns depending on the strategies and the actualization frequencies.<br>
# Our models focus on returns, but another approach could be to better estimate the risk other than with covariance matrix.


