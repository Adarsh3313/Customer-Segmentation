import streamlit as st
import pandas as pd
import pickle

# Load trained clustering model (PCA + KMeans pipeline with preprocessing)
with open("clustering_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Customer Segmentation App", layout="wide")
st.title("📊 Customer Segmentation with PCA & Clustering")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(data.head())

    # Recreate Total_Spent (as you did in the notebook)
    spend_columns = [
        'MntWines',
        'MntMeatProducts',
        'MntGoldProds',
        'MntFruits',
        'MntFishProducts',
        'MntSweetProducts'
    ]
    data['Total_Spent'] = data[spend_columns].sum(axis=1)

    # ⚠️ Important: pass the full raw dataframe to the pipeline
    clusters = model.predict(data)

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
