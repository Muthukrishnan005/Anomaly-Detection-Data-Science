import pickle
import pandas as pd
import numpy as np

def load_models():
    """Load the saved models and scaler"""
    with open('models/isolation_forest_model.pkl', 'rb') as f:
        iforest = pickle.load(f)
    
    with open('models/oneclass_svm_model.pkl', 'rb') as f:
        svm = pickle.load(f)
    
    with open('models/standard_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    return iforest, svm, scaler

def preprocess_data(df):
    """Preprocess new data in the same way as training data"""
    # Clean price columns
    df['discounted_price'] = df['discounted_price'].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)
    df['actual_price'] = df['actual_price'].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)
    df['discount_percentage'] = df['discount_percentage'].astype(str).str.replace('%', '').astype(float)
    df['rating'] = df['rating'].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)
    df['rating_count'] = pd.to_numeric(df['rating_count'].astype(str).str.replace(',', ''), errors='coerce')
    
    return df

def detect_anomalies(df):
    """Detect anomalies in new data using saved models"""
    # Load models
    iforest, svm, scaler = load_models()
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Extract features
    features = ['actual_price', 'discounted_price', 'discount_percentage', 'rating', 'rating_count']
    X = df[features].fillna(0)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Rule-based anomalies
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
    
    # Model predictions
    df['iforest_anomaly'] = iforest.predict(X_scaled) == -1
    df['svm_anomaly'] = svm.predict(X_scaled) == -1
    
    # Combined anomaly score
    df['final_anomaly'] = df['rule_based_anomaly'] | df['iforest_anomaly'] | df['svm_anomaly']
    
    return df

if __name__ == "__main__":
    # Example usage
    print("To use this script, import and use the detect_anomalies function:")
    print("from predict import detect_anomalies")
    print("results = detect_anomalies(your_dataframe)") 