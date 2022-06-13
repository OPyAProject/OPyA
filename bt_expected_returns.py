# -*- coding: utf-8 -*-

# -- Sheet --

# # PROJET ONLINE PORTFOLIO ALLOCATION
# ## Mise en place d'une stratégie de type Markowitz avec différentes méthodes d'estimation des rendements E(Ri)
# 
# ### Méthodes d'estimation implémentées avec les librairies PyPortfolioOpt et Prophet
# ####  - Mean historical return
# ####  - EMA historical return (Exponential Moving Average)
# ####  - CAPM return (Capital Asset Pricing Model)
# ####  - ML return (with Facebook Prophet Library)
# 
# ### Stratégie Markowitz en surchargeant l'algo WeighMeanVar de la librairie bt
# 
# *Data Scientist promo Sep21 - Sandra CHARLERY, Maxime VANPEENE*


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
#%matplotlib inline

# ## Chargement des données depuis le 12/07/2018
# ### 100 premières actions
# ### Benchmark (indice SP500, ticker = GSPC)


PATH_DATA = "/data/workspace_files/"

PERIOD_START = '2018-12-07'
PERIOD_END = '2022-04-08'

#FILE_NAME_TICKER_DATA = PATH_DATA + 'data100_'+ PERIOD_START +'_'+PERIOD_END+'.csv'
FILE_NAME_TICKER_DATA = '/data/workspace_files/data100_09042022.csv'
FILE_NAME_BENCHMARK_DATA = PATH_DATA + 'benchmark_'+ PERIOD_START +'_'+PERIOD_END+'.csv'
FILE_NAME_JSON_MODEL = '_serialized_prophet_model.json'
FILE_NAME_TICKER_EXPECTED_RETURN = '_expected_return.csv'
BENCHMARK_TICKER = 'gspc'

DEANNUALIZATION_PERIOD = 252

#https://ycharts.com/indicators/10_year_treasury_rate January 04, 2021 => 0.93%

RISK_FREE_RATE = 0.0093
data_all=pd.read_csv(FILE_NAME_TICKER_DATA,index_col=0,header=0,parse_dates=True)
benchmark_all = pd.read_csv(FILE_NAME_BENCHMARK_DATA,index_col=0,header=0,parse_dates=True)

# ## Entrainement du modèle sur les 2 premières années


data = data_all.loc['2018':'2020']
benchmark = benchmark_all.loc['2018':'2020']

#data = data_all
#benchmark = benchmark_all

#data.tail()



#benchmark.tail()

# ## Entrainement et sauvegarde d'un modèle Prophet pour chaque action



returns = data.to_returns().dropna()



for i,ticker in enumerate(returns.columns):
    data_ticker=returns[ticker].to_frame()
    data_ticker.reset_index(inplace=True)
    data_ticker.columns=["ds","y"]
    ticker_model = Prophet()
    ticker_model.fit(data_ticker)
    with open(PATH_DATA+ticker+FILE_NAME_JSON_MODEL, 'w') as fout:
        json.dump(model_to_json(ticker_model), fout)  # Save model

# ax = returns.hist(figsize=(20, 30))

# returns.corr().as_format('.2f')

# forecasted_returns

# data.columns[:25]

#data = data_all.loc['2021':,:'acn']
#benchmark = benchmark_all.loc['2021':]

# data = data_all.loc['2021-01-01':'2021-06-30']
# benchmark = benchmark_all.loc['2021-01-01':'2021-06-30']

data = data_all
benchmark = benchmark_all

# ## Adaptation de l'algo WeighMeanvar pour :
# ###    - paramétrer la méthode d'estimation des rendements
# ###    - prendre en compte les données de benchmark dans certains cas (ex: CAPM)


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
            with open(PATH_DATA+ticker+FILE_NAME_JSON_MODEL, 'r') as fin:
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

# #  Création des différentes stratégies (benchmark, moyenne historique,CAPM,ML with Prophet)


startDateForStaticStrategy = '1-1-2021'
sBenchmarkSP500 = bt.Strategy('sBenchmarkSP500', [bt.algos.RunAfterDate(startDateForStaticStrategy),bt.algos.RunOnce(),
                       bt.algos.SelectAll(),
                       bt.algos.WeighEqually(),
                       bt.algos.Rebalance()])
sOPAMarkowitzMean = bt.Strategy('sOPAMarkowitzMean', [bt.algos.RunAfterDate(startDateForStaticStrategy),bt.algos.RunMonthly(),
                        bt.algos.SelectHasData(),
                        OPAMarkowitz(rf=RISK_FREE_RATE),
                        bt.algos.Rebalance()])
sOPAMarkowitzMeanOnce = bt.Strategy('sOPAMarkowitzMeanOnce', [bt.algos.RunAfterDate(startDateForStaticStrategy),bt.algos.RunOnce(),
                        bt.algos.SelectHasData(),
                        OPAMarkowitz(rf=RISK_FREE_RATE),
                        bt.algos.Rebalance()])
sOPAMarkowitzEMA = bt.Strategy('sOPAMarkowitzEMA', [bt.algos.RunAfterDate(startDateForStaticStrategy),bt.algos.RunMonthly(),
                        bt.algos.SelectHasData(),
                        OPAMarkowitz(er_estimation_method=ER_ESTIMATION_METHOD.EMA,rf=RISK_FREE_RATE),
                        bt.algos.Rebalance()])
sOPAMarkowitzEMAOnce = bt.Strategy('sOPAMarkowitzEMAOnce', [bt.algos.RunAfterDate(startDateForStaticStrategy),bt.algos.RunOnce(),
                        bt.algos.SelectHasData(),
                        OPAMarkowitz(er_estimation_method=ER_ESTIMATION_METHOD.EMA,rf=RISK_FREE_RATE),
                        bt.algos.Rebalance()])
sOPAMarkowitzCAPM = bt.Strategy('sOPAMarkowitzCAPM', [bt.algos.RunAfterDate(startDateForStaticStrategy),bt.algos.RunMonthly(),
                        bt.algos.SelectHasData(),
                        OPAMarkowitz(er_estimation_method=ER_ESTIMATION_METHOD.CAPM,benchmark_ticker=BENCHMARK_TICKER,rf=RISK_FREE_RATE),
                        bt.algos.Rebalance()])
sOPAMarkowitzCAPMOnce = bt.Strategy('sOPAMarkowitzCAPMOnce', [bt.algos.RunAfterDate(startDateForStaticStrategy),bt.algos.RunOnce(),
                        bt.algos.SelectHasData(),                                    
                        OPAMarkowitz(er_estimation_method=ER_ESTIMATION_METHOD.CAPM,benchmark_ticker=BENCHMARK_TICKER,rf=RISK_FREE_RATE),
                        bt.algos.Rebalance()])

sOPAMarkowitzML = bt.Strategy('sOPAMarkowitzML', [bt.algos.RunAfterDate(startDateForStaticStrategy),bt.algos.RunMonthly(),
                        bt.algos.SelectHasData(),
                        OPAMarkowitz(er_estimation_method=ER_ESTIMATION_METHOD.PROPHET,rf=RISK_FREE_RATE),
                        bt.algos.Rebalance()])
sOPAMarkowitzMLOnce = bt.Strategy('sOPAMarkowitzMLOnce', [bt.algos.RunAfterDate(startDateForStaticStrategy),bt.algos.RunOnce(),
                        bt.algos.SelectHasData(),
                        OPAMarkowitz(er_estimation_method=ER_ESTIMATION_METHOD.PROPHET,rf=RISK_FREE_RATE),
                        bt.algos.Rebalance()])
#Risk Parity
sRiskParity3M = bt.Strategy('sRiskParity3M',[bt.algos.RunAfterDate(startDateForStaticStrategy),
                        bt.algos.RunMonthly(),
                        bt.algos.SelectAll(),
                        bt.algos.WeighERC(lookback=pd.DateOffset(months=3)),
                        bt.algos.Rebalance()])
sRiskParity3MOnce = bt.Strategy('sRiskParity3MOnce',[bt.algos.RunAfterDate(startDateForStaticStrategy),
                        bt.algos.RunOnce(),
                        bt.algos.SelectAll(),
                        bt.algos.WeighERC(lookback=pd.DateOffset(months=3)),
                        bt.algos.Rebalance()])
sMomentum3M = bt.Strategy('sMomentum3M',[bt.algos.RunAfterDate(startDateForStaticStrategy),
                        bt.algos.RunMonthly(),
                        bt.algos.SelectAll(),
                        bt.algos.SelectMomentum(n=10,lookback=pd.DateOffset(months=3)),
                        bt.algos.WeighEqually(),
                        bt.algos.Rebalance()])
sMomentum3MOnce = bt.Strategy('sMomentum3MOnce',[bt.algos.RunAfterDate(startDateForStaticStrategy),
                        bt.algos.RunOnce(),
                        bt.algos.SelectAll(),
                        bt.algos.SelectMomentum(n=10,lookback=pd.DateOffset(months=3)),
                        bt.algos.WeighEqually(),
                        bt.algos.Rebalance()])

# # Lancement BackTesting


testBenchmarkSP500 = bt.Backtest(sBenchmarkSP500, benchmark)
testOPAMarkowitzMean= bt.Backtest(sOPAMarkowitzMean, data)
testOPAMarkowitzMeanOnce= bt.Backtest(sOPAMarkowitzMeanOnce, data)

testOPAMarkowitzEMA= bt.Backtest(sOPAMarkowitzEMA, data)
testOPAMarkowitzEMAOnce= bt.Backtest(sOPAMarkowitzEMAOnce, data)

dataCAPM = data.join(benchmark)

testOPAMarkowitzCAPM= bt.Backtest(sOPAMarkowitzCAPM, dataCAPM)
testOPAMarkowitzCAPMOnce= bt.Backtest(sOPAMarkowitzCAPMOnce, dataCAPM)

testOPAMarkowitzML= bt.Backtest(sOPAMarkowitzML, data)
testOPAMarkowitzMLOnce= bt.Backtest(sOPAMarkowitzMLOnce, data)

testRiskParity3M= bt.Backtest(sRiskParity3M, data)
testRiskParity3MOnce= bt.Backtest(sRiskParity3MOnce, data)

testMomentum3M= bt.Backtest(sMomentum3M, data)
testMomentum3MOnce= bt.Backtest(sMomentum3MOnce, data)

from datetime import datetime
start_time = datetime.now()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    resBenchmark = bt.run(testBenchmarkSP500)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

from datetime import datetime
start_time = datetime.now()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    resMean = bt.run(testBenchmarkSP500,testOPAMarkowitzMeanOnce,testOPAMarkowitzMean)
    resEMA = bt.run(testBenchmarkSP500,testOPAMarkowitzEMAOnce,testOPAMarkowitzEMA)
    resCAPM = bt.run(testBenchmarkSP500,testOPAMarkowitzCAPMOnce,testOPAMarkowitzCAPM)
    resML = bt.run(testBenchmarkSP500,testOPAMarkowitzMLOnce,testOPAMarkowitzML)
    resRiskParity3M = bt.run(testBenchmarkSP500,testRiskParity3MOnce,testRiskParity3M)
    resMomentum3M = bt.run(testBenchmarkSP500,testMomentum3MOnce,testMomentum3M)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

resBenchmark.plot()
df_benchmark=resBenchmark.prices
df_benchmark.to_csv(PATH_DATA+'benchmark.csv')
#plt.savefig(PATH_DATA+"benchmark.png", dpi=200)
plt.xlim(['1-1-2021','8-4-2022'])
fig=plt.gcf()
fig.savefig(PATH_DATA+f"benchmark.png",format='png',bbox_inches='tight')

resMean.plot()
df_mean=resMean.prices
df_mean.to_csv(PATH_DATA+'markowitz_mean.csv')
#plt.savefig(PATH_DATA+"markowitz_mean.png", dpi=200)
plt.xlim(['1-1-2021','8-4-2022'])
fig=plt.gcf()
fig.savefig(PATH_DATA+f"markowitz_mean.png",format='png',bbox_inches='tight')

resCAPM.plot()
df_capm=resCAPM.prices
df_capm.to_csv(PATH_DATA+'markowitz_capm.csv')
#plt.savefig(PATH_DATA+"markowitz_capm.png", dpi=200)
plt.xlim(['1-1-2021','8-4-2022'])
fig=plt.gcf()
fig.savefig(PATH_DATA+f"markowitz_capm.png",format='png',bbox_inches='tight')

resEMA.plot()
df_ema=resEMA.prices
df_ema.to_csv(PATH_DATA+'markowitz_ema.csv')
#plt.savefig(PATH_DATA+"markowitz_ema.png", dpi=200)
plt.xlim(['1-1-2021','8-4-2022'])
fig=plt.gcf()
fig.savefig(PATH_DATA+f"markowitz_ema.png",format='png',bbox_inches='tight')

resML.plot()
df_ml=resML.prices
df_ml.to_csv(PATH_DATA+'markowitz_ml.csv')
#plt.savefig(PATH_DATA+"markowitz_ml.png", dpi=200)
plt.xlim(['1-1-2021','8-4-2022'])
fig=plt.gcf()
fig.savefig(PATH_DATA+f"markowitz_ml.png",format='png',bbox_inches='tight')

resRiskParity3M.plot()
df_rp3M=resRiskParity3M.prices
df_rp3M.to_csv(PATH_DATA+'risk_parity_3M.csv')
#plt.savefig(PATH_DATA+"risk_parity_3M.png", dpi=200)
plt.xlim(['1-1-2021','8-4-2022'])
fig=plt.gcf()
fig.savefig(PATH_DATA+f"risk_parity_3M.png",format='png',bbox_inches='tight')

resMomentum3M.plot()
df_m3M=resRiskParity3M.prices
df_m3M.to_csv(PATH_DATA+'momentum_3M.csv')
#plt.savefig(PATH_DATA+"momentum_3M.png", dpi=200)
plt.xlim(['1-1-2021','8-4-2022'])
fig=plt.gcf()
fig.savefig(PATH_DATA+f"momentum_3M.png",format='png',bbox_inches='tight')

from datetime import datetime
start_time = datetime.now()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    res = bt.run(testBenchmarkSP500,testOPAMarkowitzMean,testOPAMarkowitzEMA,testOPAMarkowitzCAPM,
     testOPAMarkowitzML,testRiskParity3M,testMomentum3M)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

# # Affichage du résultat


res.plot()
df_res=res.prices
df_res.to_csv(PATH_DATA+'opa_all.csv')
#plt.savefig(PATH_DATA+"opa_all.png", dpi=200)
plt.xlim(['1-1-2021','8-4-2022'])
fig=plt.gcf()
fig.savefig(PATH_DATA+f"opa_all.png",format='png',bbox_inches='tight')

res.set_date_range(start='2021-01-01',end='2022-04-08')
res.set_riskfree_rate(RISK_FREE_RATE)
res.display()

res.to_csv(path=PATH_DATA+"opa_all_stats.csv")

strategies = ["sOPAMarkowitzMean","sOPAMarkowitzEMA","sOPAMarkowitzCAPM",
              "sOPAMarkowitzML",
              "sRiskParity3M","sMomentum3M"]
for strategy in strategies:
    res.get_security_weights(strategy).to_csv(strategy+"_weights.csv")

plt.figure(figsize=(10,8))
sharpe_data = res.stats.loc['daily_sharpe'].sort_values()
sharpe_data.plot(kind='barh')
for index, value in enumerate(sharpe_data): 
    plt.text(value, index, round(value,2),color='k',fontweight = 'bold',fontsize=20)
ax =plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.xticks([])
plt.yticks(fontsize=20)
plt.title("SHARPE RATIO BY STRATEGY",fontsize=20, color='g', fontweight='bold')
fig=plt.gcf()
fig.savefig(PATH_DATA+f"opa_sharpe.png",format='png',bbox_inches='tight')

strategies = ["sBenchmarkSP500","sOPAMarkowitzMean","sOPAMarkowitzEMA","sOPAMarkowitzCAPM","sOPAMarkowitzML","sRiskParity3M","sMomentum3M"]

for strategy in strategies:
    res.plot_histogram(strategy)

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     testOPAMarkowitzML= bt.Backtest(sOPAMarkowitzML, data)
#     start_time = datetime.now()
#     res2 = bt.run(testOPAMarkowitzML)
#     end_time = datetime.now()
#     print('Duration: {}'.format(end_time - start_time))
#     res2.plot()

