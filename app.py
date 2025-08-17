{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "32cb89cb-be11-4c61-8c00-48c83f443273",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-08-17 17:09:11.846 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\adars\\anaconda\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import pickle\n",
    "\n",
    "# Load model\n",
    "with open(\"clustering_model.pkl\", \"rb\") as f:\n",
    "    model = pickle.load(f)\n",
    "\n",
    "st.set_page_config(page_title=\"Customer Segmentation App\", layout=\"wide\")\n",
    "\n",
    "st.title(\"Customer Segmentation with PCA & Clustering\")\n",
    "\n",
    "st.write(\"Upload customer data to get cluster predictions.\")\n",
    "\n",
    "# Upload CSV file\n",
    "uploaded_file = st.file_uploader(\"Upload CSV\", type=[\"csv\", \"xlsx\"])\n",
    "\n",
    "if uploaded_file is not None:\n",
    "    if uploaded_file.name.endswith(\".csv\"):\n",
    "        data = pd.read_csv(uploaded_file)\n",
    "    else:\n",
    "        data = pd.read_excel(uploaded_file)\n",
    "\n",
    "    st.subheader(\"Uploaded Data\")\n",
    "    st.dataframe(data.head())\n",
    "\n",
    "    # Predict clusters\n",
    "    clusters = model.predict(data)\n",
    "    data[\"Cluster\"] = clusters\n",
    "\n",
    "    st.subheader(\"Clustered Data\")\n",
    "    st.dataframe(data.head())\n",
    "\n",
    "    # Download results\n",
    "    csv = data.to_csv(index=False).encode(\"utf-8\")\n",
    "    st.download_button(\"Download Clustered Data\", csv, \"clustered_customers.csv\", \"text/csv\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "18e621da-4f2b-442d-8565-9972a9a14492",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
