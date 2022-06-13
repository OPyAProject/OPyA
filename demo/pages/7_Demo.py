import streamlit as st

st.write('# Demo')

st.write('Choose the strategies you want to use')

st.write('### Benchmark SP500')
sBenchmarkSP500= st.checkbox('sBenchmarkSP500')

st.write('### Markowitz')
sOPAMarkowitzMeanOnce_checkbox= st.checkbox('Buy_and_Hold_Mean')
sOPAMarkowitzMean_checkbox= st.checkbox('Dynamic_Mean')
sOPAMarkowitzEMAOnce_checkbox= st.checkbox('Buy_and_Hold_Exponential')
sOPAMarkowitzEMA_checkbox= st.checkbox('Dynamic_Exponential')
sOPAMarkowitzCAPMOnce_checkbox= st.checkbox('Buy_and_Hold_CAPM')
sOPAMarkowitzCAPM_checkbox= st.checkbox('Dynamic_CAPM')
sOPAMarkowitzMLOnce_checkbox= st.checkbox('Buy_and_Hold_Machine_Learning_Prophet')
sOPAMarkowitzML_checkbox= st.checkbox('Dynamic_Machine_Learning_Prophet')

st.write('### Other')
sRiskParity3MOnce_checkbox= st.checkbox('Buy_and_Hold_Risk_Parity')
sRiskParity3M_checkbox= st.checkbox('Dynamic_Risk_Parity')
sMomentum3MOnce_checkbox= st.checkbox('Buy_and_Hold_Momentum')
sMomentum3M_checkbox= st.checkbox('Dynamic_Momentum')

