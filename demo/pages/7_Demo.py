import streamlit as st
import datetime

#générales
import pandas as pd
import numpy as np
from enum import Enum
#bt
import bt
import sklearn.manifold
import sklearn.cluster
import sklearn.covariance
from scipy.optimize import minimize
import scipy.stats

from pypfopt.expected_returns import ema_historical_return, capm_return
from prophet import Prophet
from ffn.core import deannualize
import json 
from prophet.serialize import model_to_json, model_from_json
import seaborn as sns
import warnings
import matplotlib.pyplot as plt

PATH_DATA = "/data/workspace_files/"
PERIOD_START = '2018-12-07'
PERIOD_END = '2022-04-08'
FILE_NAME_TICKER_DATA = '/data/workspace_files/data100_09042022.csv'
FILE_NAME_BENCHMARK_DATA = PATH_DATA + 'benchmark_'+ PERIOD_START +'_'+PERIOD_END+'.csv'
FILE_NAME_JSON_MODEL = '_serialized_prophet_model.json'
FILE_NAME_TICKER_EXPECTED_RETURN = '_expected_return.csv'
BENCHMARK_TICKER = 'gspc'
DEANNUALIZATION_PERIOD = 252
risk_free_rate = 0.0093
data_all=pd.read_csv('./csv/data100_09042022.csv',index_col=0,header=0,parse_dates=True)
benchmark_all = pd.read_csv('./csv/benchmark_data.csv',index_col=0,header=0,parse_dates=True)
data=data_all
benchmark=benchmark_all
select_estimation_method = None
st.write('# Demo')
select_mode = st.sidebar.selectbox('Select Mode :',  ('Static (Buy and Hold)','Dynamic'))
select_strategy = st.sidebar.selectbox('Select Strategy :',  ('Markowitz','Risk Parity', 'Momentum'))
if select_strategy == 'Markowitz':
    select_estimation_method = st.sidebar.selectbox('Select E(R) Estimation method:',  ('Historical Mean','EMA', 'CAPM'))#,  'Forecast'))

min_value = datetime.date(2021, 1, 1)
if select_estimation_method == 'Forecast':
    min_value = datetime.date(2022, 2, 1)

start_date = st.sidebar.date_input(
        "Start date",
        min_value,min_value=min_value)

risk_free_rate=st.sidebar.number_input(label='Risk free rate',min_value=0.0,step=0.001,value=0.0093,format="%.4f")


ER_ESTIMATION_METHOD = Enum('ER_ESTIMATION_METHOD', 'MEAN EMA CAPM PROPHET')
class OPAMarkowitz(bt.Algo):
    def __init__(self, lookback=pd.DateOffset(months=3), rf=0.0,covar_method="standard",lag=pd.DateOffset(days=0), 
                 er_estimation_method=ER_ESTIMATION_METHOD.MEAN,benchmark_ticker=None):
        """
        * rf (float): Risk-free rate used in utility calculation
        * covar_method (str): Covariance matrix estimation method.
             Currently supported:
                 - ledoit-wolf
                 - standard
        """
        super(OPAMarkowitz, self).__init__()
        self.lookback = lookback
        self.lag = lag
        self.er_estimation_method = er_estimation_method
        self.benchmark_ticker = benchmark_ticker
        self.covar_method = covar_method
        self.rf = rf        

    def expected_returns(self,historical_returns):
        if self.er_estimation_method == ER_ESTIMATION_METHOD.MEAN:
            return historical_returns.mean()
        elif self.er_estimation_method == ER_ESTIMATION_METHOD.EMA:
            return deannualize(ema_historical_return(historical_returns,span=30,returns_data=True),DEANNUALIZATION_PERIOD)
        elif self.er_estimation_method == ER_ESTIMATION_METHOD.CAPM:
            benchmark_returns = None
            main_returns = historical_returns
            if self.benchmark_ticker is not None:
                benchmark_returns = historical_returns[self.benchmark_ticker]
                main_returns = historical_returns.drop(self.benchmark_ticker,axis=1)
                return deannualize(capm_return(main_returns,
                               benchmark_returns.rename('mkt'), risk_free_rate=self.rf,
                               returns_data=True),DEANNUALIZATION_PERIOD)
        elif self.er_estimation_method == ER_ESTIMATION_METHOD.PROPHET:
            return self.forecast_returns(historical_returns)
        else:
            raise NotImplementedError("Unknown estimation method for Expected Returns ")
            
    def forecast_returns(self,historical_returns):
        forecasted_returns = pd.DataFrame()
        for i,ticker in enumerate(historical_returns.columns):
            data_ticker=historical_returns[ticker].to_frame()
            data_ticker.reset_index(inplace=True)
            data_ticker.columns=["ds","y"]
            with open('./json/'+ticker+FILE_NAME_JSON_MODEL, 'r') as fin:
                ticker_model = model_from_json(json.load(fin))  # Load model
            future = pd.date_range(data_ticker['ds'].max()+pd.DateOffset(days=1), periods=30)
            future = pd.DataFrame(future)
            future.columns = ['ds']
            future['ds']= pd.to_datetime(future['ds'])
            forecast = ticker_model.predict(future)
            # forecast.to_csv(PATH_DATA+ticker
            #                       +str(data_ticker['ds'].max().date())+FILE_NAME_TICKER_EXPECTED_RETURN)
            if (i==0):
                forecasted_returns['ds'] = forecast['ds']
            forecasted_returns[ticker] = forecast['yhat']
        
        # forecasted_returns.to_csv(PATH_DATA+'by_step_'
        #                           +str(forecasted_returns['ds'].max().date())+FILE_NAME_TICKER_EXPECTED_RETURN)
                    
        forecasted_returns.set_index('ds')
        forecasted_returns_mean = forecasted_returns.mean()
        return forecasted_returns_mean

    def __call__(self, target):
        selected = target.temp["selected"]

        if len(selected) == 0:
            target.temp["weights"] = {}
            return True

        if len(selected) == 1:
            target.temp["weights"] = {selected[0]: 1.0}
            return True

        t0 = target.now - self.lag
        prc = target.universe.loc[t0 - self.lookback : t0, selected]
        tw = self.calc_mean_var_weights(prc.to_returns().dropna())
    
        target.temp["weights"] = tw.dropna()
        return True
    
    def calc_mean_var_weights(self,historical_returns, weight_bounds=(0., 1.)):
        """
        Calculates the mean-variance weights given a DataFrame of returns.

        Args:
            * returns (DataFrame): Returns for multiple securities.
            * weight_bounds ((low, high)): Weigh limits for optimization.


        Returns:
            Series {col_name: weight}

        """
        def fitness(weights, exp_rets, covar, rf):
            # portfolio mean
            mean = sum(exp_rets * weights)
            # portfolio var
            var = np.dot(np.dot(weights, covar), weights)
            # utility - i.e. sharpe ratio
            util = (mean - (rf/252)) / np.sqrt(var)
            # negative because we want to maximize and optimizer
            # minimizes metric
            return -util


        # expected return defaults to mean return by default
        exp_rets = self.expected_returns(historical_returns)
#        exp_rets.to_csv(PATH_DATA+str(self.er_estimation_method)+'_'+FILE_NAME_TICKER_EXPECTED_RETURN)
        if self.benchmark_ticker is not None:
            historical_returns = historical_returns.drop(self.benchmark_ticker,axis=1)
        

        n = len(historical_returns.columns)
                        
        #print("EXPECTED RETURNS WITH " + str(self.er_estimation_method))
        #print(exp_rets)
        # calc covariance matrix
        if self.covar_method == 'ledoit-wolf':
            covar = sklearn.covariance.ledoit_wolf(historical_returns)[0]
        elif self.covar_method == 'standard':
            covar = historical_returns.cov()
        else:
            raise NotImplementedError('covar_method not implemented')

        weights = np.ones([n]) / n
        bounds = [weight_bounds for i in range(n)]
        # sum of weights must be equal to 1
        constraints = ({'type': 'eq', 'fun': lambda W: sum(W) - 1.})
        optimized = minimize(fitness, weights, (exp_rets, covar, self.rf),
                             method='SLSQP', constraints=constraints,
                             bounds=bounds)
        # check if success
        if not optimized.success:
            raise Exception(optimized.message)

        # return weight vector
        return pd.Series({historical_returns.columns[i]: optimized.x[i] for i in range(n)})


if (select_mode=='Dynamic' and select_strategy=='Markowitz' and select_estimation_method=='Historical Mean'):
    strategy = bt.Strategy('sOPAMarkowitzMean', [bt.algos.RunAfterDate(start_date),bt.algos.RunMonthly(),
                        bt.algos.SelectHasData(),
                        OPAMarkowitz(rf=risk_free_rate),
                        bt.algos.Rebalance()])
if select_mode=='Static (Buy and Hold)' and select_strategy=='Markowitz' and select_estimation_method=='Historical Mean':                    
    strategy = bt.Strategy('sOPAMarkowitzMeanOnce', [bt.algos.RunAfterDate(start_date),bt.algos.RunOnce(),
                        bt.algos.SelectHasData(),
                        OPAMarkowitz(rf=risk_free_rate),
                        bt.algos.Rebalance()])
if select_mode=='Dynamic' and select_strategy=='Markowitz' and select_estimation_method=='EMA':
    strategy = bt.Strategy('sOPAMarkowitzEMA', [bt.algos.RunAfterDate(start_date),bt.algos.RunMonthly(),
                        bt.algos.SelectHasData(),
                        OPAMarkowitz(er_estimation_method=ER_ESTIMATION_METHOD.EMA,rf=risk_free_rate),
                        bt.algos.Rebalance()])
if select_mode=='Static (Buy and Hold)' and select_strategy=='Markowitz' and select_estimation_method=='EMA':
    strategy = bt.Strategy('sOPAMarkowitzEMAOnce', [bt.algos.RunAfterDate(start_date),bt.algos.RunOnce(),
                        bt.algos.SelectHasData(),
                        OPAMarkowitz(er_estimation_method=ER_ESTIMATION_METHOD.EMA,rf=risk_free_rate),
                        bt.algos.Rebalance()])
if select_mode=='Dynamic' and select_strategy=='Markowitz' and select_estimation_method=='CAPM':
    strategy = bt.Strategy('sOPAMarkowitzCAPM', [bt.algos.RunAfterDate(start_date),bt.algos.RunMonthly(),
                        bt.algos.SelectHasData(),
                        OPAMarkowitz(er_estimation_method=ER_ESTIMATION_METHOD.CAPM,benchmark_ticker=BENCHMARK_TICKER,rf=risk_free_rate),
                        bt.algos.Rebalance()])
if select_mode=='Static (Buy and Hold)' and select_strategy=='Markowitz' and select_estimation_method=='CAPM':
    strategy = bt.Strategy('sOPAMarkowitzCAPMOnce', [bt.algos.RunAfterDate(start_date),bt.algos.RunOnce(),
                        bt.algos.SelectHasData(),                                    
                        OPAMarkowitz(er_estimation_method=ER_ESTIMATION_METHOD.CAPM,benchmark_ticker=BENCHMARK_TICKER,rf=risk_free_rate),
                        bt.algos.Rebalance()])
if select_mode=='Dynamic' and select_strategy=='Markowitz' and select_estimation_method=='Forecast':
    strategy = bt.Strategy('sOPAMarkowitzML', [bt.algos.RunAfterDate(start_date),bt.algos.RunMonthly(),
                        bt.algos.SelectHasData(),
                        OPAMarkowitz(er_estimation_method=ER_ESTIMATION_METHOD.PROPHET,rf=risk_free_rate),
                        bt.algos.Rebalance()])
if select_mode=='Static (Buy and Hold)' and select_strategy=='Markowitz' and select_estimation_method=='Forecast':
    strategy = bt.Strategy('sOPAMarkowitzMLOnce', [bt.algos.RunAfterDate(start_date),bt.algos.RunOnce(),
                        bt.algos.SelectHasData(),                                    
                        OPAMarkowitz(er_estimation_method=ER_ESTIMATION_METHOD.PROPHET,rf=risk_free_rate),
                        bt.algos.Rebalance()])
#Risk Parity
if select_mode=='Dynamic' and select_strategy=='Risk Parity':
    strategy = bt.Strategy('sRiskParity3M',[bt.algos.RunAfterDate(start_date),
                        bt.algos.RunMonthly(),
                        bt.algos.SelectAll(),
                        bt.algos.WeighERC(lookback=pd.DateOffset(months=3)),
                        bt.algos.Rebalance()])
if select_mode=='Static (Buy and Hold)' and select_strategy=='Risk Parity':
    strategy = bt.Strategy('sRiskParity3MOnce',[bt.algos.RunAfterDate(start_date),
                        bt.algos.RunOnce(),
                        bt.algos.SelectAll(),
                        bt.algos.WeighERC(lookback=pd.DateOffset(months=3)),
                        bt.algos.Rebalance()])
if select_mode=='Dynamic' and select_strategy=='Momentum':
    strategy = bt.Strategy('sMomentum3M',[bt.algos.RunAfterDate(start_date),
                        bt.algos.RunMonthly(),
                        bt.algos.SelectAll(),
                        bt.algos.SelectMomentum(n=10,lookback=pd.DateOffset(months=3)),
                        bt.algos.WeighEqually(),
                        bt.algos.Rebalance()])
if select_mode=='Static (Buy and Hold)' and select_strategy=='Momentum':
    strategy = bt.Strategy('sMomentum3MOnce',[bt.algos.RunAfterDate(start_date),
                        bt.algos.RunOnce(),
                        bt.algos.SelectAll(),
                        bt.algos.SelectMomentum(n=10,lookback=pd.DateOffset(months=3)),
                        bt.algos.WeighEqually(),
                        bt.algos.Rebalance()])
    
if select_strategy=='Markowitz' and select_estimation_method=='CAPM':
        data=data.join(benchmark)

test=bt.Backtest(strategy,data)
res=bt.run(test)
res.plot()
plt.xlim([start_date,'4-8-2022'])
fig=plt.gcf()
st.pyplot(fig)