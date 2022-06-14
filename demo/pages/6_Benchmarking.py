import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.write("# Benchmarking method")

st.write("## Benchmark")
st.write("We use S&P 500 as the Reference to compare performances of the different strategies implemented in the project.")
st.write("The S&P 500 is the most representative index of the American stock market because it is made up of \
    a greater number of companies and its value takes into account of the market capitalization of the companies included in the index.")

from PIL import Image
image = Image.open('./img/benchmark.PNG')
st.image(image)

st.write("# Prophet Model Training")
st.write("We decided to train and save Prophet Model on each stock for about the first 2 years of the dataset, until December 31, 2020.")
st.write("Then, the forecasting period matches the backtesting period.")

st.write("# Backtesting period")
st.write("The backtesting period goes from January 01, 2021 to April 08, 2022.")
st.write("For dynamic allocation, each strategy is run monthly.")

st.write("# Backtesting tools")
st.write("We use BT library as the framework to run the different strategies, and PyPortfolioOpt mainly for CAPM and EMA calculations.")

code = '''\

#Custom Algo
class OPAMarkowitz(bt.Algo):
    def __init__(self, ...)
    def __call__(self, target):
        selected = target.temp["selected"]
        ...
        target.temp["weights"] = tw.dropna()
        return True    

#Strategy definition
sOPAMarkowitzML = bt.Strategy('sOPAMarkowitzML', 
    [
    bt.algos.RunAfterDate(startDateForStaticStrategy),
    bt.algos.RunMonthly(),
    bt.algos.SelectHasData(),
    OPAMarkowitz(er_estimation_method=ER_ESTIMATION_METHOD.PROPHET,
                 rf=RISK_FREE_RATE),
    bt.algos.Rebalance()
    ]
    )
#Data feeding
testOPAMarkowitzML= bt.Backtest(sOPAMarkowitzML, data)

#Run
resML = bt.run(#testBenchmarkSP500,
               #testOPAMarkowitzMLOnce,
               testOPAMarkowitzML)
#View result
resML.plot()

#View stats
resML.display()
'''
st.code(code)
image = Image.open('./img/res_display.PNG')
st.image(image)
st.write('# Benchmarking Results')
st.sidebar.write('# Results')
select_mode = st.sidebar.selectbox('Select Mode :',  ('Static vs Dynamic','All Dynamic'))
if select_mode == 'Static vs Dynamic':
    select_strategy = st.sidebar.selectbox('Select Strategy :',  ('Markowitz','Risk Parity', 'Momentum'))
    if select_strategy == 'Markowitz':
        select_estimation_method = st.sidebar.selectbox('Select E(R) Estimation method:',  ('Historical Mean','EMA', 'CAPM',  'Forecast'))
        if select_estimation_method == 'Historical Mean':
            image = Image.open('./img/markowitz_mean.png')
            st.image(image)
        elif select_estimation_method == 'EMA':
            image = Image.open('./img/markowitz_ema.png')
            st.image(image)
        elif select_estimation_method == 'CAPM':
            image = Image.open('./img/markowitz_capm.png')
            st.image(image)
        elif select_estimation_method == 'Forecast':
            image = Image.open('./img/markowitz_ml.png')
            st.image(image)
    elif select_strategy == 'Risk Parity':
        image = Image.open('./img/risk_parity_3M.png')
        st.image(image)
    elif select_strategy == 'Momentum':
        image = Image.open('./img/momentum_3M.png')
        st.image(image)


if select_mode == 'All Dynamic':
    image = Image.open('./img/opa_all.png')
    st.image(image)
    st.write('#### Sharpe Ratio Indicator')
    image = Image.open('./img/opa_sharpe.png')
    st.image(image)
