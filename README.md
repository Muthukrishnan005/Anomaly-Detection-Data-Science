# Amazon Sales Anomaly Detection

## Overview
This project implements a comprehensive anomaly detection system for Amazon product listings, combining rule-based and machine learning approaches to identify suspicious patterns in pricing, discounts, and customer reviews.

## Features
- Rule-based anomaly detection
- Machine Learning models (Isolation Forest & One-Class SVM)
- Hyperparameter tuning
- Feature importance analysis
- Interactive visualizations
- Category-wise analysis
- Price range analysis
- Saved models for easy reuse

## Setup
1. Clone the repository
```bash
git clone https://github.com/Muthukrishnan005/Anomaly-Detection-Data-Science.git
cd Anomaly-Detection-Data-Science
```

2. Install required packages
```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook
```bash
jupyter notebook amazon.ipynb
```

## Project Structure
- `amazon.ipynb`: Main Jupyter notebook containing the analysis
- `amazon_sales_dataset.csv`: Dataset file
- `requirements.txt`: Required Python packages
- `predict.py`: Utility functions for using saved models
- `example_usage.py`: Example script showing how to use saved models
- `models/`: Directory containing saved models
  - `isolation_forest_model.pkl`: Trained Isolation Forest model
  - `oneclass_svm_model.pkl`: Trained One-Class SVM model
  - `standard_scaler.pkl`: Fitted StandardScaler
- `README.md`: Project documentation

## Using Saved Models
To use the trained models on new data:

```python
from predict import detect_anomalies
import pandas as pd

# Load your new data
df = pd.read_csv('your_new_data.csv')

# Detect anomalies
results = detect_anomalies(df)

# Save anomalies to CSV
results[results['final_anomaly']].to_csv('detected_anomalies.csv', index=False)
```

## Results
The project successfully identifies:
- Pricing anomalies
- Suspicious discount patterns
- Unusual review patterns
- Category-specific anomalies

## Future Improvements
- Real-time anomaly detection
- API integration
- Deep learning implementation
- Automated reporting system

## License
MIT

## Author
Muthukrishnan R