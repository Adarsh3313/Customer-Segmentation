import streamlit as st
import pandas as pd
import pickle

with open("clustering_model.pkl", "rb") as f:
    model = pickle.load(f)

feature_columns = [
    'Income',
    'MntWines',
    'MntMeatProducts',
    'MntGoldProds',
    'NumWebPurchases',
    'NumStorePurchases',
    'Recency',
    'Total_Spent'
]

st.set_page_config(page_title="Customer Segmentation App", layout="wide")
st.title("📊 Customer Segmentation with PCA & Clustering")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(data.head())

    spend_columns = [
        'MntWines',
        'MntMeatProducts',
        'MntGoldProds',
        'MntFishProducts',
        'MntFruits',
        'MntSweetProducts'
    ]
    data['Total_Spent'] = data[spend_columns].sum(axis=1)

    X = data[feature_columns]
    clusters = model.predict(X)
    data['Cluster'] = clusters

    st.subheader("Clustered Data")
    st.dataframe(data.head())

    csv_data = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Clustered Data",
        data=csv_data,
        file_name="clustered_data.csv",
        mime="text/csv"
    )
