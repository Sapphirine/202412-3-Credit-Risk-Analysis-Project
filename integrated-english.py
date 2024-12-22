import streamlit as st
import pandas as pd
import numpy as np
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
import gc
from datetime import datetime, timezone

# Page configuration
st.set_page_config(page_title="Credit Default Risk Analysis & Prediction", layout="wide")
st.title("Credit Default Risk Analysis & Prediction System")

# Set data path
DATA_PATH = Path("parquet_files/train")

@st.cache_data
def load_base_data():
    """Load base data"""
    try:
        df_base = pl.read_parquet(DATA_PATH / "train_base.parquet")
        return df_base.to_pandas()
    except Exception as e:
        st.error(f"Failed to load base data: {str(e)}")
        return None

@st.cache_data
def load_static_data():
    """Load static data"""
    try:
        df_static_cb = pl.read_parquet(DATA_PATH / "train_static_cb_0.parquet")
        df_static_0 = pl.concat([
            pl.read_parquet(DATA_PATH / "train_static_0_0.parquet"),
            pl.read_parquet(DATA_PATH / "train_static_0_1.parquet")
        ])
        return df_static_cb.to_pandas(), df_static_0.to_pandas()
    except Exception as e:
        st.error(f"Failed to load static data: {str(e)}")
        return None, None

@st.cache_resource
def load_model_and_types():
    """Load model and data types"""
    try:
        model = joblib.load('lgbm_best_model.pkl')
        column_types = pd.read_csv('column_dtypes.csv', index_col=0).squeeze()
        # Get categorical feature names
        categorical_columns = column_types[column_types == 'object'].index.tolist()
        
        # Validate loaded data
        if model is None:
            raise ValueError("Failed to load model")
        if column_types.empty:
            raise ValueError("Column types data is empty")
            
        return model, column_types, categorical_columns
    except Exception as e:
        st.error(f"Failed to load model and configuration: {str(e)}")
        return None, None, None

def create_basic_analysis(df):
    """Create basic data analysis"""
    try:
        if df is None:
            return []
            
        figures = []
        
        # TARGET distribution
        if 'target' in df.columns:
            target_counts = df['target'].value_counts()
            fig1 = px.pie(values=target_counts.values,
                         names=['Not Default', 'Default'],
                         title='Default Distribution')
            figures.append(fig1)
        
        # Monthly distribution
        if 'month_decision' in df.columns:
            monthly_counts = df['month_decision'].value_counts().sort_index()
            fig2 = px.line(x=monthly_counts.index, 
                          y=monthly_counts.values,
                          title='Monthly Application Distribution',
                          labels={'x': 'Month', 'y': 'Application Count'})
            figures.append(fig2)
            
        return figures
    except Exception as e:
        st.error(f"Failed to create analysis: {str(e)}")
        return []

def create_prediction_inputs():
    """Create prediction input interface"""
    input_data = {}
    
    st.sidebar.header("Application Information Input")
    
    # Basic information
    st.sidebar.subheader("Basic Information")
    input_data['credamount_770A'] = st.sidebar.number_input('Loan Amount', 
                                                           min_value=10000.0,
                                                           max_value=1000000.0,
                                                           value=100000.0, 
                                                           step=10000.0)
    input_data['month_decision'] = int(st.sidebar.slider('Application Month', 1, 12, 6))
    input_data['weekday_decision'] = int(st.sidebar.slider('Application Weekday', 0, 6, 3))
    
    # Customer information
    st.sidebar.subheader("Customer Information")
    input_data['homephncnt_628L'] = int(st.sidebar.number_input('Number of Home Phones', 
                                                               value=1, 
                                                               min_value=0,
                                                               max_value=10))
    input_data['mobilephncnt_593L'] = int(st.sidebar.number_input('Number of Mobile Phones', 
                                                                 value=1, 
                                                                 min_value=0,
                                                                 max_value=10))
    input_data['numactivecreds_622L'] = int(st.sidebar.number_input('Number of Active Loans', 
                                                                   value=0, 
                                                                   min_value=0,
                                                                   max_value=20))
    
    return input_data

def prepare_prediction_data(input_data, column_types, categorical_columns):
    """Prepare prediction data"""
    try:
        # Create default values dictionary
        default_values = {}
        for col in column_types.index:
            if col not in input_data:
                if col in categorical_columns:
                    default_values[col] = '0'
                elif 'float' in str(column_types[col]):
                    default_values[col] = 0.0
                elif 'int' in str(column_types[col]):
                    default_values[col] = 0
                else:
                    default_values[col] = '0'
        
        # Update input data
        input_data.update(default_values)
        
        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # Convert data types
        for col, dtype in column_types.items():
            if col in df.columns:
                try:
                    if col in categorical_columns:
                        df[col] = df[col].astype(str)
                    elif 'datetime' in str(dtype):
                        df[col] = pd.to_datetime(df[col])
                    else:
                        df[col] = df[col].astype(dtype)
                except Exception as e:
                    st.error(f"Failed to convert column {col}: {str(e)}")
        
        return df
    except Exception as e:
        st.error(f"Failed to prepare data: {str(e)}")
        return None

def show_prediction_results(pred_proba, input_data):
    """Show prediction results"""
    try:
        st.subheader("Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred_proba * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Default Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            st.plotly_chart(fig)
        
        with col2:
            st.write("### Risk Assessment")
            if pred_proba < 0.3:
                st.success("Low Risk")
                recommendation = "Recommend Approval"
            elif pred_proba < 0.7:
                st.warning("Medium Risk")
                recommendation = "Further Evaluation Needed"
            else:
                st.error("High Risk")
                recommendation = "Recommend Rejection"
                
            st.write(f"""
            #### Assessment Details
            - Default Probability: {pred_proba:.2%}
            - Recommended Action: {recommendation}
            - Loan Amount: {input_data['credamount_770A']:,.2f}
            - Number of Active Loans: {input_data['numactivecreds_622L']}
            """)
    except Exception as e:
        st.error(f"Failed to display results: {str(e)}")

def main():
    try:
        # Load model and configuration
        model, column_types, categorical_columns = load_model_and_types()
        
        if model is None or column_types is None or categorical_columns is None:
            st.error("System initialization failed: Failed to load model or configuration files")
            return
            
        # Create main tabs
        tab1, tab2, tab3 = st.tabs(["Data Analysis", "Prediction System", "Model Description"])
        
        with tab1:
            st.markdown("### Data Analysis")
            df_base = load_base_data()
            
            if df_base is not None:
                figures = create_basic_analysis(df_base)
                
                if figures:
                    for fig in figures:
                        st.plotly_chart(fig, use_container_width=True)
                
                del df_base
                gc.collect()
        
        with tab2:
            st.markdown("### Default Prediction System")
            
            # Create input interface
            input_data = create_prediction_inputs()
            
            # Add prediction button
            if st.sidebar.button('Predict'):
                # Prepare prediction data
                df_pred = prepare_prediction_data(input_data, column_types, categorical_columns)
                
                if df_pred is not None:
                    try:
                        # Ensure column order matches model
                        df_pred = df_pred[model.feature_name_]
                        # Make prediction
                        pred_proba = model.predict_proba(df_pred)[0][1]
                        # Show prediction results
                        show_prediction_results(pred_proba, input_data)
                    except Exception as e:
                        st.error(f"Prediction process error: {str(e)}")
        
        with tab3:
            st.markdown("""
            ### Model Description
            
            #### Model Used
            - Model Type: LightGBM
            - Evaluation Metric: AUC
            - Validation Method: 5-fold Cross Validation
            
            #### Key Features
            1. Basic Information
               - Loan Amount
               - Application Time
            2. Customer Features
               - Contact Information
               - Existing Loans
            3. Behavioral Features
               - Historical Defaults
               - Repayment Performance
            
            #### Instructions
            1. Fill in the application information in the input bar on the left
            2. Click "Predict" to get the result
            3. View prediction results and risk assessment
            4. Make decisions based on recommendations
            """)
            
    except Exception as e:
        st.error(f"System error: {str(e)}")

if __name__ == "__main__":
    main()