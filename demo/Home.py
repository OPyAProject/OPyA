import streamlit as st

st.write('# OPyA project')
st.write('### Online Portfolio Allocation benchmarking with Python',)
st.write(' *Data Scientist promo Sep21 - Sandra CHARLERY, Maxime VANPEENE*')

from PIL import Image
image = Image.open('./img/intro_image_finance.jpeg')
st.image(image)