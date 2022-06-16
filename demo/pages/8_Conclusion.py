import streamlit as st

st.write('# Conclusion ')

st.write('It is very difficult to predict markets.')
st.write('This study shows however that several dynamic allocation strategies can outperform the market on our dataset.')
st.write('')
st.write('Regarding the Sharpe ratio indicator, the strategy of Markowitz with historical mean has the best risk reward.')
st.write('Moreover the performance exceeds that of the investment at the risk-free rate.')
st.write('')
st.write('We have not performed a strategy using the RSI but this metrics seems to give interesting results.')
st.write('Another lead would be to use deep learning models instead of Prophet. We made several trials on the Google Temporal Fusion Transformer (TFT), but due to computing limitations we were not able to get exploitable results.')
st.write('')
st.write('To confirm the generalization of the strategies, other studies should be performed on other datasets with other trends.')
st.write('Our models do not include transaction costs which will influence returns depending on the strategies and the actualization frequencies.')
st.write('Our models focus on returns, but another approach could be to better estimate the risk other than with covariance matrix.')
