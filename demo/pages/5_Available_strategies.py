import streamlit as st
from prophet import Prophet
from prophet.plot import plot_cross_validation_metric 
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
import pandas as pd
import matplotlib.pyplot as plt
#warnings
import warnings
warnings.filterwarnings("ignore")

data100=pd.read_csv('./csv/data100_09042022.csv',index_col=0,header=0,parse_dates=True)

st.write('# Toward Modern Portfolio Strategies')

from PIL import Image
image = Image.open('./img/9_OPA_Building_Strategies.PNG')
st.image(image)

add_select_detail_strategy = st.sidebar.radio(
    '',
    ('Allocation Method', 'Allocation Strategy', 'E(R) Estimation Method')
)

if add_select_detail_strategy == 'Allocation Method':
    image = Image.open('./img/10_balance.jpeg')
    st.write('## Allocation method : Rebalance or Not ?')
    st.image(image,width=600)
    st.write('The aim of our project is to adopt a **dynamic allocation strategy** with the hope to be better than with a **static strategy**.')
    st.write('With dynamic method, the weights of the assets in the portfolio are rebalanced Monthly, according to the allocation strategy.')
    st.write('With static method, we buy the assets and hold that position until the end.')
elif add_select_detail_strategy == 'Allocation Strategy':
    st.write('## Allocation strategy : Markowitz, Risk Parity, Momentum')
    st.write("### Risk Parity")
    st.write('Risk parity is a portfolio allocation strategy that uses risk to determine allocations across various components of an investment portfolio. \
        The risk parity strategy modifies the modern portfolio theory (MPT) approach to investing through the use of leverage.')
    st.write("### Momentum")
    st.write('Momentum investing is a system of buying stocks or other securities that have had high returns over the past three to twelve months,\
         and selling those that have had poor returns over the same period.')
    st.write("### Harry Markowitz and the Modern Portfolio Theory")
    st.write("**Deal : Find the Portfolio on the Efficient Frontier that maximizes the Sharpe Ratio**")
    image1 = Image.open('./img/12_sharpe_ratio_formula.PNG')
    st.image(image1)
    image2 = Image.open('./img/13_ef_scatter.PNG')
    st.image(image2)
elif add_select_detail_strategy == 'E(R) Estimation Method':
    st.write('## E(R) Estimation Method')
    st.write("### Historical Mean")
    st.write('The mean historical return of an asset is its past rate of return and performance.')
    image = Image.open('./img/14_historical_mean_formula.PNG')
    st.image(image)
    st.write("### EMA : Exponential Moving Average")  
    st.write('The EMA is a kind of moving average that gives more importance to recent data.')
    image = Image.open('./img/15_ema_formula.PNG')
    st.image(image)
    st.write("### CAPM : Capital Asset Pricing Model")
    image = Image.open('./img/capm_formula.PNG')
    st.image(image)
    st.write("### Future Prediction : Facebook Prophet Model")
    image = Image.open('./img/prophet_formula.PNG')
    st.image(image)
    st.write("#### Sample output")
    ticker = st.selectbox('Select a share :',  (data100.columns.tolist()))

    ticker_list_for_forecast = [ticker]
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
            st.pyplot(fig)

            st.write('#### Prophet forecast componnets')
            ticker_models[ticker].plot_components(ticker_forecast)
            st.write(plt.gcf())
            if(ticker == 'googl'):
                ticker_cross_validations[ticker] = cross_validation(ticker_models[ticker], period='30 days', horizon = '30 days')
                st.write('#### Prophet cross validation and metrics')
                plot_cross_validation_metric(ticker_cross_validations[ticker], metric='rmse')
                st.write(plt.gcf())