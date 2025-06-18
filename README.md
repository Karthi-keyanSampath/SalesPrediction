# Store Sales Prediction

This project implements a machine learning model to predict store sales based on various features like item details and store characteristics.

## Project Structure

- `sales-prediction-and-analysis.ipynb`: Jupyter notebook containing data analysis, model training, and evaluation
- `app.py`: Streamlit web application for sales prediction
- `XGBoost_GPU_best_model.pkl`: Trained XGBoost model
- `cleaned_data.csv`: Preprocessed dataset
- Other model files: Various trained models for comparison

## Features

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Model training using XGBoost with GPU acceleration
- Interactive web interface for predictions
- Comprehensive error handling and input validation

## Setup and Installation

1. Clone the repository
2. Install required packages:
   ```bash
   pip install streamlit pandas numpy scikit-learn xgboost
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Model Features

The model considers various features for prediction:
- Item characteristics (weight, visibility, type, MRP)
- Store details (location, size, type)
- Historical data patterns

## Usage

1. Open the Streamlit app
2. Enter the required details in the form
3. Click "Predict Sales" to get the sales prediction

## Model Performance

The XGBoost model achieves:
- High R² score on validation data
- Robust performance across different store types
- Effective handling of categorical and numerical features
