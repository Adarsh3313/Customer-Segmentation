import streamlit as st
import pandas as pd
import pickle

# Load the trained clustering model (PCA + KMeans pipeline)
with open("clustering_model.pkl", "rb") as f:
    model = pickle.load(f)

# Columns used for training
feature_columns = [
    'Income',
    'MntWines',
    'MntMeatProducts',
    'MntGoldP',
    'NumWebPurchases',
    'NumStorePurchases',
    'Recency',
    'Total_Spent'
]

st.set_page_config(page_title="Customer Segmentation App", layout="wide")
st.title("📊 Customer Segmentation with PCA & Clustering")
st.write(
    "Upload customer data to generate cluster predictions using the trained model."
)

# File uploader
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(data.head())

    # Select features used during training
    X = data[feature_columns]

    # Predict clusters
    clusters = model.predict(X)
    data["Cluster"] = clusters

    st.subheader("Clustered Data")
    st.dataframe(data.head())

    # Convert to CSV for download
    csv_data = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Clustered Data",
        data=csv_data,
        file_name="clustered_data.csv",
        mime="text/csv",
    )
