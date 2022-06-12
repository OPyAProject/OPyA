import streamlit as st

st.write('# Data exploration')

st.write('### Market Global Tendency')

st.write('Over the studied time period, a global uptrend can be observed.')

from PIL import Image
image = Image.open('./img/1_all_stocks_subplots.png')
st.image(image)

st.write('This is confirmed by simulating a naive portfolio with equal weights.')

image2 = Image.open('./img/2_naive_portolio.png')
st.image(image2)

