import streamlit as st
from PIL import Image

image = Image.open('./img/data_scientest_logo.png')
st.image(image)

st.write('# OPyA project')
st.write('### Online Portfolio Allocation benchmarking with Python',)
st.write(' *Data Scientist promo Sep21 - Sandra CHARLERY, Maxime VANPEENE*')


image = Image.open('./img/0_intro_image_finance.jpeg')
st.image(image)