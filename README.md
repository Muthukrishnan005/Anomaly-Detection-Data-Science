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

## Setup
1. Clone the repository
```bash
git clone https://github.com/[your-username]/Amazon_Sales_Anomaly_Detection.git
cd Amazon_Sales_Anomaly_Detection
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
- `README.md`: Project documentation

## Required Packages
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- plotly
- jupyter

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