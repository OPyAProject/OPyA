import streamlit as st
import pandas as pd

st.write('# Dataset')

st.write('The dataset is made of the 100 largest capitalizations in the NYSE stock market at the time of the project')
st.write('These values are easy to buy and sell and can be used in a dynamic portfolio allocation strategy.')


#import du csv avec 500 valeurs du SP500
#colonnes (ticker,name,sector,subsector)
sectors=pd.read_csv('./csv/sp500_sectors.csv',encoding='cp1252')
sectors['ticker']=sectors['ticker'].map(lambda x:x.lower())
sectors.index = sectors.index.map(str)
#on filtre sur les 100 actions qui nous intéressent et on met le ticker comme index
data100=pd.read_csv('./csv/data100_09042022.csv',index_col=0,header=0,parse_dates=True)

sectors_100=sectors[sectors['ticker'].isin(data100.columns.to_list())]
sectors_100=sectors_100.set_index('ticker')

st.dataframe(sectors_100)

st.write('The considered value is the Adjusted Close, which was taken between the 07th of December 2018 and the 08th of April 2022.')
st.write('Below are the 10 first lines of the dataset.')

st.dataframe(data100.head(10))