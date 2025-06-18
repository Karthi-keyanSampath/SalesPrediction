import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline

# Define the same columns as in training
num_columns = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year']
nominal_columns = ['Item_Type', 'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Identifier']
ordinal_columns = ['Item_Fat_Content', 'Outlet_Size']

# Define the valid categories for each categorical feature
VALID_CATEGORIES = {
    'Item_Type': ['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household', 'Baking Goods', 
                  'Snack Foods', 'Frozen Foods', 'Breakfast', 'Health and Hygiene', 'Hard Drinks', 
                  'Canned', 'Breads', 'Starchy Foods', 'Others', 'Seafood'],
    'Item_Fat_Content': ['Low Fat', 'Regular'],
    'Outlet_Size': ['Small', 'Medium', 'High'],
    'Outlet_Location_Type': ['Tier 1', 'Tier 2', 'Tier 3'],
    'Outlet_Type': ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3']
}

def create_feature_vector(input_data):
    """Convert input data to DataFrame with proper structure"""
    df = pd.DataFrame([input_data])
    required_columns = num_columns + ordinal_columns + nominal_columns
    df = df[required_columns]
    return df

def load_model():
    """Load the model with proper GPU configuration"""
    try:
        model = pickle.load(open("XGBoost_GPU_best_model.pkl", "rb"))
        # Update model parameters for current XGBoost version
        if hasattr(model, 'get_params'):
            params = model.get_params()
            if 'tree_method' in params and params['tree_method'] == 'gpu_hist':
                model.set_params(tree_method='hist', device='cuda')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_sales(features):
    """Make predictions with proper error handling and preprocessing"""
    try:
        model = load_model()
        if model is None:
            return None
            
        # Create feature vector
        feature_df = create_feature_vector(features)
        
        # Make prediction (preprocessing is handled by the pipeline)
        prediction = model.predict(feature_df)
        
        # Convert from log scale back to original scale
        return np.exp(prediction)[0]
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
        return None

def main():
    st.title('Store Sales Prediction')
    st.write('Enter the details below to predict store sales')

    # Create two columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Product Details')
        
        item_weight = st.number_input('Item Weight', min_value=0.0, max_value=100.0, value=12.0)
        
        item_fat_content = st.selectbox(
            'Item Fat Content',
            options=VALID_CATEGORIES['Item_Fat_Content']
        )
        
        item_visibility = st.number_input(
            'Item Visibility', 
            min_value=0.0, 
            max_value=1.0, 
            value=0.1
        )
        
        item_type = st.selectbox(
            'Item Type',
            options=VALID_CATEGORIES['Item_Type']
        )
        
        item_mrp = st.number_input('Item MRP', min_value=0.0, value=100.0)

    with col2:
        st.subheader('Store Details')
        
        outlet_identifier = st.selectbox(
            'Outlet Identifier',
            options=['OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019', 'OUT027', 'OUT035', 'OUT045', 'OUT046', 'OUT049']
        )
        
        outlet_establishment_year = st.number_input(
            'Outlet Establishment Year',
            min_value=1900,
            max_value=2025,
            value=2000
        )
        
        outlet_size = st.selectbox(
            'Outlet Size',
            options=VALID_CATEGORIES['Outlet_Size']
        )
        
        outlet_location_type = st.selectbox(
            'Outlet Location Type',
            options=VALID_CATEGORIES['Outlet_Location_Type']
        )
        
        outlet_type = st.selectbox(
            'Outlet Type',
            options=VALID_CATEGORIES['Outlet_Type']
        )

    # Create input dictionary
    input_data = {
        'Item_Weight': item_weight,
        'Item_Fat_Content': item_fat_content,
        'Item_Visibility': item_visibility,
        'Item_Type': item_type,
        'Item_MRP': item_mrp,
        'Outlet_Identifier': outlet_identifier,
        'Outlet_Establishment_Year': outlet_establishment_year,
        'Outlet_Size': outlet_size,
        'Outlet_Location_Type': outlet_location_type,
        'Outlet_Type': outlet_type
    }

    if st.button('Predict Sales'):
        # Make prediction
        sales = predict_sales(input_data)
        
        if sales is not None:
            st.success(f'Predicted Sales: ₹{sales:,.2f}')
            
            # Show feature importance or additional insights
            st.info("""
            Note: This prediction is based on historical sales data and considers:
            - Product characteristics (weight, visibility, type, etc.)
            - Store features (location, size, type)
            - Market factors (MRP)
            """)

if __name__ == '__main__':
    main()
