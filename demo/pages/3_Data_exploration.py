import streamlit as st

st.write('# Data exploration')

st.write('### Market Global Tendency')

from PIL import Image
image = Image.open('./img/1_all_stocks_subplots.png')
st.image(image)

from PIL import Image
image2 = Image.open('./img/2_naive_portolio.png')
st.image(image2)

