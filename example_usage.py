import pandas as pd
from predict import detect_anomalies

def main():
    # Load your new data
    try:
        # Load sample data (you can replace this with your new data)
        df = pd.read_csv('amazon_sales_dataset.csv')
        print("Data loaded successfully!")
        
        # Detect anomalies
        results = detect_anomalies(df)
        
        # Print summary
        print("\nAnomaly Detection Results:")
        print("-" * 30)
        print(f"Total products analyzed: {len(results)}")
        print(f"Rule-based anomalies: {results['rule_based_anomaly'].sum()}")
        print(f"Isolation Forest anomalies: {results['iforest_anomaly'].sum()}")
        print(f"One-Class SVM anomalies: {results['svm_anomaly'].sum()}")
        print(f"Total anomalies (combined): {results['final_anomaly'].sum()}")
        
        # Save results
        results[results['final_anomaly']].to_csv('detected_anomalies.csv', index=False)
        print("\nResults saved to 'detected_anomalies.csv'")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 