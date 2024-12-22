# Credit Risk Model Stability Project

This project develops a predictive model for credit risk assessment, specifically designed to evaluate loan applicants with limited credit history. The model balances predictive performance with temporal stability, making it suitable for real-world financial applications.

## Overview

The project addresses the challenge of financial inclusion by:
- Developing robust credit risk assessment models for individuals with limited credit history
- Maintaining model stability over time to reduce retraining costs
- Implementing advanced feature engineering and machine learning techniques
- Providing an interactive interface for real-time predictions

## Features

- Advanced data preprocessing using Polars for efficient data handling
- Comprehensive feature engineering pipeline including temporal and statistical aggregation
- Implementation of state-of-the-art machine learning models (LightGBM and CatBoost)
- Interactive visualization dashboard built with Streamlit
- Model stability monitoring using Population Stability Index (PSI)

## Technical Architecture

- **Data Preprocessing**: Polars for high-performance data manipulation
- **Model Training**: LightGBM/CatBoost for gradient boosting
- **Interface**: Streamlit for interactive user interface
- **Visualization**: Plotly for dynamic data visualization
- **Model Serialization**: Joblib for efficient model storage

## Performance

- Achieved AUC score of 0.86 in validation
- Maintained model stability with only 0.9% AUC decay over time
- Superior performance compared to traditional logistic regression methods
- Effective handling of imbalanced datasets
## Usage

### Data Preprocessing and Model Training
```bash
python wrangling_all_1202.py
```

### Running the Demo Interface
```bash
python streamlit_final.py
```


## Results

- Successfully handles high-dimensional, temporal, and hierarchical data
- Implements robust feature engineering and selection
- Provides interpretable predictions through SHAP values
- Maintains model stability for long-term deployment

## Future Improvements

- Implementation of dynamic updates via online learning
- Enhanced explainability through SHAP-based visualizations
- Expansion to new regions and demographics
- Integration of real-time data streams

## Contributors

- Qianran Wu - Electrical Engineering and Computer Sciences, Columbia University
- Pengyu Tao - Statistics, Columbia University

## License

This project is under no license.

## Acknowledgments

Based on the Kaggle competition "Home Credit - Credit Risk Model Stability"
