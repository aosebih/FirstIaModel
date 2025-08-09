import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor


st.set_page_config(
    page_title="Housing Price Estimator",
    page_icon="🏠",
    layout="centered"
)


np.random.seed(42)
n = 1000
data = pd.DataFrame({
    'LotArea': np.random.randint(2000, 15000, size=n),
    'BedroomAbvGr': np.random.randint(1, 6, size=n),
    'OverallQual': np.random.randint(1, 11, size=n),
    'YearBuilt': np.random.randint(1950, 2023, size=n),
})
data['SalePrice'] = (
    data['LotArea'] * np.random.uniform(2, 4, size=n) +
    data['BedroomAbvGr'] * 10000 +
    data['OverallQual'] * 15000 +
    (2023 - data['YearBuilt']) * -500 +
    np.random.normal(0, 10000, size=n)
).astype(int)


features = ['LotArea', 'BedroomAbvGr', 'OverallQual', 'YearBuilt']
X = data[features]
y = data['SalePrice']
model = DecisionTreeRegressor(random_state=10)
model.fit(X, y)


st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h1 style='color: #2E8B57;'>🏡 Housing Price Estimator</h1>
        <p style='font-size: 18px;'>Enter your house details below to get an instant price prediction.</p>
    </div>
""", unsafe_allow_html=True)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        lot_area = st.slider("Lot Area (sq ft)", 2000, 15000, 5000)
        bedrooms = st.selectbox("Bedrooms Above Ground", [1, 2, 3, 4, 5])
    with col2:
        quality = st.slider("Overall Quality (1–10)", 1, 10, 5)
        year_built = st.slider("Year Built", 1950, 2023, 2000)

    submitted = st.form_submit_button("💰 Predict Sale Price")


if submitted:
    input_data = pd.DataFrame([{
        'LotArea': lot_area,
        'BedroomAbvGr': bedrooms,
        'OverallQual': quality,
        'YearBuilt': year_built
    }])
    prediction = model.predict(input_data)[0]
    st.markdown(f"""
        <div style='text-align: center; padding: 1rem; background-color: #f0f8ff; border-radius: 10px;'>
            <h2 style='color: #2E8B57;'>Estimated Sale Price:</h2>
            <h1 style='color: #1E90FF;'>${prediction:,.0f}</h1>
        </div>
    """, unsafe_allow_html=True)


with st.expander("📊 View Sample Data"):
    st.dataframe(data.head(10))


st.markdown("""
    <hr style='margin-top: 2rem;'>
    <div style='text-align: center; font-size: 14px; color: gray;'>
        Built with by SEBIH using Streamlit & Scikit-Learn
    </div>
""", unsafe_allow_html=True)

