#library import
#General
import pandas as pd
import bt
import datetime
import numpy as np
from scipy.optimize import minimize

#Plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams['axes.grid'] = True
import seaborn as sns

#Data import

#Creation of a ticker_list_100 with format 'ticker1,ticker2,...,tickerN'
df=pd.read_csv("data/sp500_100first.csv")
ticker_list_100=df.iloc[0][0].lower()
for index in df.index[1:]:
    ticker_list_100+=','+df.iloc[index][0].lower()

"""
#Creation of data100 dataset
data100 = bt.get(ticker_list_100, start='2017-01-01')
data100.head(10)
data100.to_csv('/data/data100_XXXXXX.csv')
"""

#Import from local .csv
data100=pd.read_csv('data/data100_09042022.csv',index_col=0,header=0,parse_dates=True)

#Data exploration
print('Rows '+str(data100.shape[0]))
print('Columns '+str(data100.shape[1]))
print(data100.index)

#Data cleaning
print(data100.isna().sum().sum()) #no NaN
print(data100.duplicated().sum()) #no duplicates
print(data100.dtypes) #All values are float64
print(data100.describe()) #No visible outliers


#Dataviz

#Plot all stocks #1

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
plt.show()

#Plot index made of the sum of all stocks #2

index_fictif=data100.sum(axis=1)
index_fictif.plot();
ax=plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
plt.title('Index fictif somme des valeurs');
plt.show()

#Plot sectors repartition in the portfolio #3

#import csv
#columns = (TICKER,name,sector,subsector)
sectors=pd.read_csv('data/sp500_sectors.csv',encoding='cp1252')
#tickers to lower
sectors['ticker']=sectors['ticker'].map(lambda x:x.lower())
#filtering on 100 actions and set index to ticker
sectors=sectors[sectors['ticker'].isin(data100.columns.to_list())]
sectors=sectors.set_index('ticker')
#Plotting
sectors.groupby('sector')\
    .agg('count')\
    .sort_values(
                by='name',
                ascending=False)\
    .plot.pie(y='name',
            figsize=(16,9),
            title='100 Stocks sectors',
            autopct= lambda x: '{:.0f}'.format(x*sectors.shape[0]/100),
            legend=False,
            textprops={'fontsize': 12},
            pctdistance=0.9);
plt.axis('off');
plt.show()

#Plot sectors indexes in the portfolio (lineplot) #4

data100_by_sector=pd.DataFrame(0, index=data100.index, columns=sectors['sector'].unique())
for ticker in data100.columns.to_list():
    data100_by_sector[sectors.loc[ticker,'sector']]+=data100[ticker]
data100_by_sector['sum']=index_fictif
data100_by_sector.drop('sum',axis=1).plot(figsize=(16,16));
plt.show()

#Plot sectors return with a 100 initial investment (lineplot) #5

data100_by_sector_inv=data100_by_sector.div(data100_by_sector.iloc[0]/100)

fig,ax = plt.subplots(figsize=(16,16))
data100_by_sector_inv.plot(ax=ax, x_compat=True)
ax.xaxis.set_major_locator(mdates.YearLocator())
for line in ax.get_lines():
    line.alpha=0.2
    if line.get_label() == 'sum':
        line.set_linewidth(3)
plt.ylabel('Return')
plt.show()

#Plot sectors return with a 100 initial investment (boxplot) #6

order = data100_by_sector_inv.median().sort_values().index.to_list()
data100_by_sector_inv.boxplot(column=order,rot=90);
plt.show()