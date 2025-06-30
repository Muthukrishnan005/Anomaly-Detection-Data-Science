import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

st.set_page_config(page_title="Amazon Anomaly Detector", layout="wide")
st.title("📦 Amazon Product Anomaly Detector")

uploaded_file = st.file_uploader("Upload your Amazon dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Raw Data Preview")
    st.dataframe(df.head())

    try:
        df['discounted_price'] = df['discounted_price'].astype(str).str.replace('₹', '', regex=False).str.replace(',', '', regex=False).astype(float)
        df['actual_price'] = df['actual_price'].astype(str).str.replace('₹', '', regex=False).str.replace(',', '', regex=False).astype(float)
        df['discount_percentage'] = df['discount_percentage'].astype(str).str.replace('%', '', regex=False).astype(float)
        df['rating'] = df['rating'].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)
        df['rating_count'] = pd.to_numeric(df['rating_count'].astype(str).str.replace(',', '', regex=False), errors='coerce')
    except Exception as e:
        st.error(f"Data cleaning error: {e}")
        st.stop()

    df['price_diff'] = df['actual_price'] - df['discounted_price']

    df['anomaly_invalid_price'] = df['actual_price'] < df['discounted_price']
    df['anomaly_high_discount'] = df['discount_percentage'] > 90
    df['anomaly_low_rating_high_votes'] = (df['rating'] < 2) & (df['rating_count'] > 1000)
    df['anomaly_negative_discount'] = df['discount_percentage'] < 0
    df['rule_based_anomaly'] = df[[
        'anomaly_invalid_price',
        'anomaly_high_discount',
        'anomaly_low_rating_high_votes',
        'anomaly_negative_discount'
    ]].any(axis=1)

    features = ['actual_price', 'discounted_price', 'discount_percentage', 'rating', 'rating_count']
    X = df[features].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    model = IsolationForest(contamination=0.01, random_state=42)
    df['iforest_anomaly'] = model.fit_predict(X_scaled)
    df['iforest_anomaly'] = df['iforest_anomaly'] == -1

    svm_model = OneClassSVM(kernel='rbf', nu=0.01, gamma='auto')
    df['svm_anomaly'] = svm_model.fit_predict(X_scaled)
    df['svm_anomaly'] = df['svm_anomaly'] == -1

    df['final_anomaly'] = df['rule_based_anomaly'] | df['iforest_anomaly'] | df['svm_anomaly']

    st.subheader("🚨 Rule-Based Anomalies")
    st.dataframe(df[df['rule_based_anomaly']][['product_name', 'actual_price', 'discounted_price', 'discount_percentage', 'rating', 'rating_count']])

    st.subheader("🤖 Isolation Forest Anomalies")
    st.dataframe(df[df['iforest_anomaly']][['product_name', 'actual_price', 'discounted_price', 'discount_percentage', 'rating', 'rating_count']])

    st.subheader("🔵 One-Class SVM Anomalies")
    st.dataframe(df[df['svm_anomaly']][['product_name', 'actual_price', 'discounted_price', 'discount_percentage', 'rating', 'rating_count']])

    st.subheader("📦 All Detected Anomalies (Any Method)")
    st.dataframe(df[df['final_anomaly']][['product_name', 'actual_price', 'discounted_price', 'discount_percentage', 'rating', 'rating_count']])

    st.success(f"Rule-Based: {df['rule_based_anomaly'].sum()} | Isolation Forest: {df['iforest_anomaly'].sum()} | One-Class SVM: {df['svm_anomaly'].sum()}")
    
    iforest_csv = df[df['iforest_anomaly']].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Isolation Forest Anomalies", iforest_csv, "isolation_forest_anomalies.csv", "text/csv")

    svm_csv = df[df['svm_anomaly']].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download One-Class SVM Anomalies", svm_csv, "svm_anomalies.csv", "text/csv")

    final_csv = df[df['final_anomaly']].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download All Detected Anomalies", final_csv, "all_anomalies.csv", "text/csv")

else:
    st.info("👆 Upload the AMAZON CSV file to get started.") 
