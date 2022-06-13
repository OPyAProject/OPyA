import streamlit as st

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
    st.write("You selected E(R) Estimation Method.")
