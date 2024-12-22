import streamlit as st
import pandas as pd
import numpy as np
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
from datetime import datetime, timezone
import os

# Page configuration
st.set_page_config(page_title="Credit Default Prediction Model Demo", layout="wide")
st.title("Credit Default Prediction Model Demo")

@st.cache_data
def compute_and_save_stats():
    """Pre-compute and save statistical data"""
    try:
        # Use Polars for efficient statistical computation
        df = pl.scan_csv('processed_data.csv')
        
        # Calculate credit amount distribution
        credit_stats = (df.select('credamount_770A')
                         .with_columns([
                             pl.when(pl.col('credamount_770A') <= 50000).then(pl.lit('0-50k'))
                             .when(pl.col('credamount_770A') <= 100000).then(pl.lit('50k-100k'))
                             .when(pl.col('credamount_770A') <= 200000).then(pl.lit('100k-200k'))
                             .when(pl.col('credamount_770A') <= 500000).then(pl.lit('200k-500k'))
                             .otherwise(pl.lit('500k+'))
                             .alias('range')
                         ])
                         .collect()
                         .group_by('range')
                         .count()
                         .sort('range')
                         .to_pandas())
        
        # Calculate monthly application statistics
        monthly_stats = (df.select('month_decision')
                          .collect()
                          .group_by('month_decision')
                          .count()
                          .sort('month_decision')
                          .to_pandas())
        
        # Calculate active credit count distribution
        credit_count_stats = (df.select('numactivecreds_622L')
                              .collect()
                              .group_by('numactivecreds_622L')
                              .count()
                              .sort('numactivecreds_622L')
                              .to_pandas())
        
        # Calculate contact information statistics
        contact_stats = {
            'mobile': df.select(pl.col('mobilephncnt_593L')).collect().mean().item(),
            'home': df.select(pl.col('homephncnt_628L')).collect().mean().item()
        }
        
        # Save statistical results
        stats = {
            'credit_stats': credit_stats,
            'monthly_stats': monthly_stats,
            'credit_count_stats': credit_count_stats,
            'contact_stats': contact_stats
        }
        joblib.dump(stats, 'visualization_stats.joblib')
        return stats
        
    except Exception as e:
        st.error(f"Failed to compute statistics: {str(e)}")
        st.write("Error Details:", type(e).__name__)
        st.write("Complete Error Message:", str(e))
        return None

@st.cache_resource
def load_stats():
    """Load pre-computed statistical data"""
    try:
        return joblib.load('visualization_stats.joblib')
    except FileNotFoundError:
        return compute_and_save_stats()
    except Exception as e:
        st.error(f"Failed to load statistical data: {str(e)}")
        return None

def create_credit_amount_dist(stats):
    """Create a credit amount distribution chart"""
    credit_stats = stats['credit_stats']
    # Convert string names to integers for sorting
    order_map = {
        '0-50k': 0,
        '500k+': 4,
        '50k-100k': 1,
        '100k-200k': 2,
        '200k-500k': 3
    }
    credit_stats['sort_order'] = credit_stats['range'].map(order_map)
    credit_stats = credit_stats.sort_values('sort_order').drop('sort_order', axis=1)

    fig = px.bar(credit_stats,
                 x='range',
                 y='count',
                 title="Credit Amount Distribution",
                 labels={'range': 'Credit Amount Range', 'count': 'Application Count'})
    fig.update_layout(xaxis_title="Credit Amount Range", yaxis_title="Application Count")
    return fig

def create_monthly_trend(stats):
    """Create a monthly application trend chart"""
    monthly_stats = stats['monthly_stats']
    fig = px.line(monthly_stats,
                  x='month_decision',
                  y='count',
                  title="Monthly Application Trend",
                  labels={'month_decision': 'Month', 'count': 'Application Count'})
    fig.update_layout(xaxis_title="Month", yaxis_title="Application Count")
    return fig

def create_active_credits_dist(stats):
    """Create an active credit count distribution chart"""
    credit_count_stats = stats['credit_count_stats']
    # Limit display of credit count to avoid visualization impact from outliers
    credit_count_stats = credit_count_stats[credit_count_stats['numactivecreds_622L'] <= 10]
    
    fig = px.bar(credit_count_stats,
                 x='numactivecreds_622L',
                 y='count',
                 title="Active Credit Count Distribution",
                 labels={'numactivecreds_622L': 'Active Credit Count', 'count': 'Customer Count'})
    fig.update_layout(xaxis_title="Active Credit Count", yaxis_title="Customer Count")
    return fig

def create_contact_info_dist(stats):
    """Create a contact information distribution chart"""
    contact_stats = stats['contact_stats']
    contact_df = pd.DataFrame({
        'Type': ['Mobile Number', 'Home Phone'],
        'Average Count': [contact_stats['mobile'], contact_stats['home']]
    })
    
    fig = px.bar(contact_df,
                 x='Type',
                 y='Average Count',
                 title="Average Contact Information Count",
                 labels={'Type': 'Contact Type', 'Average Count': 'Average Count'})
    fig.update_layout(xaxis_title="Contact Type", yaxis_title="Average Count")
    return fig

# Load pre-computed statistical data
stats = load_stats()

if stats is not None:
    tab1, tab2 = st.tabs(["Model Description", "Data Analysis"])
    
    with tab1:
        with st.expander("Model Overview"):
            st.markdown("""
            ### Model Overview
            - Model Used: LightGBM
            - Evaluation Metric: AUC
            - Validation Method: 5-Fold Cross-Validation
            - Model Features: Suitable for credit default risk prediction with multidimensional features.

            ### Main Feature Categories
            1. Basic Information
                - Credit Amount
                - Application Date
            2. Customer Attributes
                - Contact Information Count
                - Active Credit Count
            3. Historical Behavior
                - Default History
                - Payment Behavior
            4. Credit Assessment
                - Credit Limit
                - Debt Situation
            """)
        
    with tab2:
        st.subheader("Data Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Credit Amount Distribution
            fig1 = create_credit_amount_dist(stats)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Active Credit Count Distribution
            fig2 = create_active_credits_dist(stats)
            st.plotly_chart(fig2, use_container_width=True)
            
        with col2:
            # Monthly Application Trend
            fig3 = create_monthly_trend(stats)
            st.plotly_chart(fig3, use_container_width=True)
            
            # Contact Information Distribution
            fig4 = create_contact_info_dist(stats)
            st.plotly_chart(fig4, use_container_width=True)

@st.cache_resource
def load_model_and_types():
    """Load the trained model and column types"""
    model = joblib.load('lgbm_best_model.pkl')
    column_types = pd.read_csv('column_dtypes.csv', index_col=0).squeeze()
    
    # Get all categorical feature column names
    categorical_columns = column_types[column_types == 'object'].index.tolist()
    
    return model, column_types, categorical_columns

try:
    model, column_types, categorical_columns = load_model_and_types()
    st.success("Model and configuration files loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

def prepare_default_values(column_types, categorical_columns):
    """Prepare default values for input fields based on feature types"""
    defaults = {}
    for col, dtype in column_types.items():
        if col in categorical_columns:
            defaults[col] = None  # Use None for categorical features
        elif 'float' in str(dtype):
            defaults[col] = None
        elif 'int' in str(dtype):
            defaults[col] = None
        elif 'datetime' in str(dtype):
            defaults[col] = pd.NaT  # Use NaT for missing datetime values
        else:
            defaults[col] = None
    return defaults

def create_numeric_input(label, default_value, key=None):
    """Create a numeric input field with a checkbox for missing values"""
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        value = st.number_input(label, value=default_value, key=key)
    with col2:
        is_missing = st.checkbox("Missing", key=f"{key}_missing" if key else None)
    return None if is_missing else value




import json
from pathlib import Path

# 预定义的示例数据
EXAMPLE_1_HIGH = {
    "credamount_770A": 200000.0,
    "month_decision": 9,
    "weekday_decision": 7,
    "price_1097A": 90000.0,
    "days30_165L": 3,
    "days180_256L": 0,
    "mean_maxdpdtolerance_577P": 15.0,
    "mean_pmts_overdue_1140A": 300.0,
    "mean_pmts_dpd_1073P": 15.0,
    "homephncnt_628L": 3,
    "mobilephncnt_593L": 3,
    "numactivecreds_622L": 3,
    "pmtnum_254L": 12,
    "max_sex_738L": "Male",
    "max_incometype_1044T": "Type B",
    "mean_outstandingdebt_522A": 10000.0,
    "max_totalamount_6A": 150000.0,
    "max_empl_employedfrom_271D": 1.0
}
EXAMPLE_2_MED = {
    "credamount_770A": 200000.0,
    "month_decision": 7,
    "weekday_decision": 5,
    "price_1097A": 30000.0,
    "days30_165L": 5,
    "days180_256L": 4,
    "mean_maxdpdtolerance_577P": 30.0,
    "mean_pmts_overdue_1140A": 100.0,
    "mean_pmts_dpd_1073P": 5.0,
    "homephncnt_628L": 1,
    "mobilephncnt_593L": 2,
    "numactivecreds_622L": 3,
    "pmtnum_254L": 12,
    "max_sex_738L": "MISSING",
    "max_incometype_1044T": "MISSING",
    "mean_outstandingdebt_522A": 15000.0,
    "max_totalamount_6A": 100000.0,
    "max_empl_employedfrom_271D": 3.0
}
EXAMPLE_3_LOW = {
    "credamount_770A": 200000.0,
    "month_decision": 8,
    "weekday_decision": 5,
    "price_1097A": 30000.0,
    "days30_165L": 1,
    "days180_256L": 2,
    "mean_maxdpdtolerance_577P": 0.0,
    "mean_pmts_overdue_1140A": 0.0,
    "mean_pmts_dpd_1073P": 0.0,
    "homephncnt_628L": 1,
    "mobilephncnt_593L": 4,
    "numactivecreds_622L": 2,
    "pmtnum_254L": 6,
    "max_sex_738L": "Female",
    "max_incometype_1044T": "Type A",
    "mean_outstandingdebt_522A": 15000.0,
    "max_totalamount_6A": 250000.0,
    "max_empl_employedfrom_271D": 1.0
}

# 参数网格搜索范围
param_grid = {
    'credamount_770A': [10000.0, 50000.0, 100000.0, 150000.0, 200000.0],
    'month_decision': list(range(1, 13)),
    'weekday_decision': list(range(1, 8)),
    'price_1097A': [10000.0, 30000.0, 50000.0, 70000.0, 90000.0],
    'days30_165L': [0, 1, 2, 3, 4, 5],
    'days180_256L': [0, 1, 2, 3, 4, 5],
    'mean_maxdpdtolerance_577P': [0.0, 5.0, 15.0, 30.0],
    'mean_pmts_overdue_1140A': [0.0, 100.0, 200.0, 300.0],
    'mean_pmts_dpd_1073P': [0.0, 5.0, 10.0, 15.0],
    'homephncnt_628L': [0, 1, 2, 3],
    'mobilephncnt_593L': [1, 2, 3, 4, 5],
    'numactivecreds_622L': [1, 2, 3, 4, 5, 6, 7],
    'pmtnum_254L': [6, 12, 24, 36],
    'max_sex_738L': ['Male', 'Female', 'MISSING'],
    'max_incometype_1044T': ['Type A', 'Type B', 'Type C', 'MISSING'],
    'mean_outstandingdebt_522A': [1000.0, 5000.0, 10000.0, 15000.0, 20000.0],
    'max_totalamount_6A': [5000.0, 100000.0, 150000.0, 200000.0, 250000.0],
    'max_empl_employedfrom_271D': [0.0, 1.0, 3.0, 5.0, 10.0]
}


def save_default_values(values):
    """保存默认值到JSON文件"""
    try:
        with open('default_values.json', 'w') as f:
            json.dump(values, f, indent=4)
        return True
    except Exception as e:
        st.error(f"Failed to save default values: {str(e)}")
        return False

import random

# 在示例按钮部分添加页面刷新功能和随机按钮
st.sidebar.markdown("### Quick Examples")

# 改为2行2列布局
row1_col1, row1_col2 = st.sidebar.columns(2)
row2_col1, row2_col2 = st.sidebar.columns(2)

# JavaScript刷新代码
refresh_js = """
<script>
    window.location.reload();
</script>
"""

# 第一行第一列
with row1_col1:
    if st.button("Example 1 (High Risk)", help="High risk example (Expected: ~0.168)"):
        if save_default_values(EXAMPLE_1_HIGH):
            st.success("Filled Successfully")
            st.markdown(refresh_js, unsafe_allow_html=True)

# 第一行第二列
with row1_col2:
    if st.button("Example 2 (Med Risk)", help="Medium risk example (Expected: ~0.125)"):
        if save_default_values(EXAMPLE_2_MED):
            st.success("Filled Successfully")
            st.markdown(refresh_js, unsafe_allow_html=True)

# 第二行第一列
with row2_col1:
    if st.button("Example 3 (Low Risk)", help="Low risk example (Expected: ~0.063)"):
        if save_default_values(EXAMPLE_3_LOW):
            st.success("Filled Successfully")
            st.markdown(refresh_js, unsafe_allow_html=True)

# 第二行第二列
with row2_col2:
    if st.button("Random Fill", help="Generate random example"):
        random_values = {
            key: random.choice(values) for key, values in param_grid.items()
        }
        if save_default_values(random_values):
            st.success("Filled Successfully")
            st.markdown(refresh_js, unsafe_allow_html=True)



def get_default_value(field_name, original_default):
    """从JSON文件获取默认值，如果不存在则返回原始默认值"""
    try:
        if Path('default_values.json').exists():
            with open('default_values.json', 'r') as f:
                defaults = json.load(f)
                return defaults.get(field_name, original_default)
    except Exception:
        pass
    return original_default


def create_input_fields():
    """Create input fields for user input in the sidebar"""
    input_data = {}
    
    # 获取所需特征
    required_features = set(model.feature_name_)
    
    # Basic Information
    st.sidebar.subheader("Basic Information")
    if 'credamount_770A' in required_features:
        input_data['credamount_770A'] = create_numeric_input('Credit Amount', 
            get_default_value('credamount_770A', 100000.0), 'credamount')
            
    if 'month_decision' in required_features:
        month_val = create_numeric_input('Decision Month', 
            get_default_value('month_decision', 3), 'month')
        input_data['month_decision'] = None if month_val is None else int(month_val)
        
    if 'weekday_decision' in required_features:
        weekday_val = create_numeric_input('Decision Day of Week', 
            get_default_value('weekday_decision', 4), 'weekday')
        input_data['weekday_decision'] = None if weekday_val is None else int(weekday_val)
        
    if 'price_1097A' in required_features:
        input_data['price_1097A'] = create_numeric_input('Transaction Price', 
            get_default_value('price_1097A', 50000.0), 'price')
            
    if 'days30_165L' in required_features:
        days30_val = create_numeric_input('Events in Last 30 Days', 
            get_default_value('days30_165L', 1), 'days30')
        input_data['days30_165L'] = None if days30_val is None else int(days30_val)
        
    if 'days180_256L' in required_features:
        days180_val = create_numeric_input('Events in Last 180 Days', 
            get_default_value('days180_256L', 2), 'days180')
        input_data['days180_256L'] = None if days180_val is None else int(days180_val)
        
    if 'mean_maxdpdtolerance_577P' in required_features:
        input_data['mean_maxdpdtolerance_577P'] = create_numeric_input('Avg. Max Overdue Tolerance', 
            get_default_value('mean_maxdpdtolerance_577P', 15.0), 'maxdpd')
            
    if 'mean_pmts_overdue_1140A' in required_features:
        input_data['mean_pmts_overdue_1140A'] = create_numeric_input('Avg. Overdue Payment Amount', 
            get_default_value('mean_pmts_overdue_1140A', 200.0), 'pmts_overdue')
            
    if 'mean_pmts_dpd_1073P' in required_features:
        input_data['mean_pmts_dpd_1073P'] = create_numeric_input('Avg. Overdue Payment Days', 
            get_default_value('mean_pmts_dpd_1073P', 5.0), 'pmts_dpd')

    # Customer Information
    st.sidebar.subheader("Customer Information")
    if 'homephncnt_628L' in required_features:
        home_val = create_numeric_input('Home Phone Count', 
            get_default_value('homephncnt_628L', 2), 'home')
        input_data['homephncnt_628L'] = None if home_val is None else int(home_val)
        
    if 'mobilephncnt_593L' in required_features:
        mobile_val = create_numeric_input('Mobile Phone Count', 
            get_default_value('mobilephncnt_593L', 5), 'mobile')
        input_data['mobilephncnt_593L'] = None if mobile_val is None else int(mobile_val)
        
    if 'numactivecreds_622L' in required_features:
        creds_val = create_numeric_input('Active Credit Count', 
            get_default_value('numactivecreds_622L', 4), 'creds')
        input_data['numactivecreds_622L'] = None if creds_val is None else int(creds_val)
        
    if 'pmtnum_254L' in required_features:
        pmt_val = create_numeric_input('Payment Count', 
            get_default_value('pmtnum_254L', 10), 'pmt')
        input_data['pmtnum_254L'] = None if pmt_val is None else int(pmt_val)
        
    if 'max_sex_738L' in required_features:
        default_sex = get_default_value('max_sex_738L', 'MISSING')
        input_data['max_sex_738L'] = st.sidebar.selectbox('Gender Statistic', 
            options=['MISSING', 'Male', 'Female'],
            index=['MISSING', 'Male', 'Female'].index(default_sex))
            
    if 'max_incometype_1044T' in required_features:
        default_income = get_default_value('max_incometype_1044T', 'MISSING')
        input_data['max_incometype_1044T'] = st.sidebar.selectbox('Highest Income Type', 
            options=['MISSING', 'Type A', 'Type B', 'Type C'],
            index=['MISSING', 'Type A', 'Type B', 'Type C'].index(default_income))
    
    # Historical Data
    st.sidebar.subheader("Historical Data")
    if 'mean_outstandingdebt_522A' in required_features:
        input_data['mean_outstandingdebt_522A'] = create_numeric_input('Avg. Outstanding Debt', 
            get_default_value('mean_outstandingdebt_522A', 10000.0), 'debt')
            
    if 'max_totalamount_6A' in required_features:
        input_data['max_totalamount_6A'] = create_numeric_input('Max Total Amount', 
            get_default_value('max_totalamount_6A', 200000.0), 'total')
            
    if 'max_empl_employedfrom_271D' in required_features:
        input_data['max_empl_employedfrom_271D'] = create_numeric_input('Longest Employment Duration (years)', 
            get_default_value('max_empl_employedfrom_271D', 5.0), 'empl')

    return input_data


# def create_input_fields():
#     """Create input fields for user input in the sidebar"""
#     input_data = {}
    
#     # Get required features from model
#     required_features = set(model.feature_name_)
    
#     # Basic Information
#     st.sidebar.subheader("Basic Information")
#     if 'credamount_770A' in required_features:
#         input_data['credamount_770A'] = create_numeric_input('Credit Amount', 100000.0, 'credamount')
#     if 'month_decision' in required_features:
#         month_val = create_numeric_input('Decision Month', 3, 'month')
#         input_data['month_decision'] = None if month_val is None else int(month_val)
#     if 'weekday_decision' in required_features:
#         weekday_val = create_numeric_input('Decision Day of Week', 4, 'weekday')
#         input_data['weekday_decision'] = None if weekday_val is None else int(weekday_val)
#     if 'price_1097A' in required_features:
#         input_data['price_1097A'] = create_numeric_input('Transaction Price', 50000.0, 'price')
#     if 'days30_165L' in required_features:
#         days30_val = create_numeric_input('Events in Last 30 Days', 1, 'days30')
#         input_data['days30_165L'] = None if days30_val is None else int(days30_val)
#     if 'days180_256L' in required_features:
#         days180_val = create_numeric_input('Events in Last 180 Days', 2, 'days180')
#         input_data['days180_256L'] = None if days180_val is None else int(days180_val)
#     if 'mean_maxdpdtolerance_577P' in required_features:
#         input_data['mean_maxdpdtolerance_577P'] = create_numeric_input('Avg. Max Overdue Tolerance', 15.0, 'maxdpd')
#     if 'mean_pmts_overdue_1140A' in required_features:
#         input_data['mean_pmts_overdue_1140A'] = create_numeric_input('Avg. Overdue Payment Amount', 200.0, 'pmts_overdue')
#     if 'mean_pmts_dpd_1073P' in required_features:
#         input_data['mean_pmts_dpd_1073P'] = create_numeric_input('Avg. Overdue Payment Days', 5.0, 'pmts_dpd')

#     # Customer Information
#     st.sidebar.subheader("Customer Information")
#     if 'homephncnt_628L' in required_features:
#         home_val = create_numeric_input('Home Phone Count', 2, 'home')
#         input_data['homephncnt_628L'] = None if home_val is None else int(home_val)
#     if 'mobilephncnt_593L' in required_features:
#         mobile_val = create_numeric_input('Mobile Phone Count', 5, 'mobile')
#         input_data['mobilephncnt_593L'] = None if mobile_val is None else int(mobile_val)
#     if 'numactivecreds_622L' in required_features:
#         creds_val = create_numeric_input('Active Credit Count', 4, 'creds')
#         input_data['numactivecreds_622L'] = None if creds_val is None else int(creds_val)
#     if 'pmtnum_254L' in required_features:
#         pmt_val = create_numeric_input('Payment Count', 10, 'pmt')
#         input_data['pmtnum_254L'] = None if pmt_val is None else int(pmt_val)
#     if 'max_sex_738L' in required_features:
#         input_data['max_sex_738L'] = st.sidebar.selectbox('Gender Statistic', 
#                                                          options=['MISSING', 'Male', 'Female'], 
#                                                          index=0)
#     if 'max_incometype_1044T' in required_features:
#         input_data['max_incometype_1044T'] = st.sidebar.selectbox('Highest Income Type', 
#                                                                  options=['MISSING', 'Type A', 'Type B', 'Type C'], 
#                                                                  index=0)
    
#     # Historical Data
#     st.sidebar.subheader("Historical Data")
#     if 'mean_outstandingdebt_522A' in required_features:
#         input_data['mean_outstandingdebt_522A'] = create_numeric_input('Avg. Outstanding Debt', 10000.0, 'debt')
#     if 'max_totalamount_6A' in required_features:
#         input_data['max_totalamount_6A'] = create_numeric_input('Max Total Amount', 200000.0, 'total')
#     if 'max_empl_employedfrom_271D' in required_features:
#         input_data['max_empl_employedfrom_271D'] = create_numeric_input('Longest Employment Duration (years)', 5.0, 'empl')
        

#     # FIXME: Debug
#     # print(input_data)
#     # breakpoint()
    
#     return input_data


def prepare_data_for_model(input_data):
    """Prepare input data for the model"""
    # Create a DataFrame
    df = pd.DataFrame([input_data])
    
    # Get required features from model
    required_features = set(model.feature_name_)
    
    # Handle missing columns and null values
    for col in required_features:
        if col not in df.columns or df[col].isnull().any():
            if col in categorical_columns:
                df[col] = 'MISSING'
            else:
                # For numeric columns, use median or 0 depending on your preference
                # You might want to compute these statistics from your training data
                df[col] = 0
    
    # Convert data types with proper error handling
    for col in df.columns:
        if col in column_types:
            try:
                if col in categorical_columns:
                    df[col] = df[col].fillna('MISSING').astype(str)
                elif 'datetime' in str(column_types[col]):
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    df[col] = df[col].fillna(pd.NaT)
                else:
                    # For numeric columns, convert to float first to handle missing values
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(0).astype(column_types[col])
            except Exception as e:
                st.warning(f"Error converting column {col}: {str(e)}. Using default value.")
                if col in categorical_columns:
                    df[col] = 'MISSING'
                else:
                    df[col] = 0
    
    # Ensure all required features are present and in correct order
    final_df = pd.DataFrame(columns=model.feature_name_)
    for col in model.feature_name_:
        if col in df.columns:
            final_df[col] = df[col]
        else:
            if col in categorical_columns:
                final_df[col] = 'MISSING'
            else:
                final_df[col] = 0
    
    return final_df



# 在全局范围定义示例数据
EXAMPLE_DATA = {
    "high_risk": {
        "credamount_770A": 200000.0,
        "month_decision": 9,
        "weekday_decision": 7,
        "price_1097A": 90000.0,
        "days30_165L": 3,
        "days180_256L": 0,
        "mean_maxdpdtolerance_577P": 15.0,
        "mean_pmts_overdue_1140A": 300.0,
        "mean_pmts_dpd_1073P": 15.0,
        "homephncnt_628L": 3,
        "mobilephncnt_593L": 3,
        "numactivecreds_622L": 3,
        "pmtnum_254L": 12,
        "max_sex_738L": "Male",
        "max_incometype_1044T": "Type B",
        "mean_outstandingdebt_522A": 10000.0,
        "max_totalamount_6A": 150000.0,
        "max_empl_employedfrom_271D": 1.0
    },
    "medium_risk": {
        "credamount_770A": 200000.0,
        "month_decision": 7,
        "weekday_decision": 5,
        "price_1097A": 30000.0,
        "days30_165L": 5,
        "days180_256L": 4,
        "mean_maxdpdtolerance_577P": 30.0,
        "mean_pmts_overdue_1140A": 100.0,
        "mean_pmts_dpd_1073P": 5.0,
        "homephncnt_628L": 1,
        "mobilephncnt_593L": 2,
        "numactivecreds_622L": 3,
        "pmtnum_254L": 12,
        "max_sex_738L": "MISSING",
        "max_incometype_1044T": "MISSING",
        "mean_outstandingdebt_522A": 15000.0,
        "max_totalamount_6A": 100000.0,
        "max_empl_employedfrom_271D": 3.0
    },
    "low_risk": {
        "credamount_770A": 200000.0,
        "month_decision": 8,
        "weekday_decision": 5,
        "price_1097A": 30000.0,
        "days30_165L": 1,
        "days180_256L": 2,
        "mean_maxdpdtolerance_577P": 0.0,
        "mean_pmts_overdue_1140A": 0.0,
        "mean_pmts_dpd_1073P": 0.0,
        "homephncnt_628L": 1,
        "mobilephncnt_593L": 4,
        "numactivecreds_622L": 2,
        "pmtnum_254L": 6,
        "max_sex_738L": "Female",
        "max_incometype_1044T": "Type A",
        "mean_outstandingdebt_522A": 15000.0,
        "max_totalamount_6A": 250000.0,
        "max_empl_employedfrom_271D": 1.0
    }
}



# Main content area
input_data = create_input_fields()

def show_feature_importance(model):
    """Display feature importance analysis"""
    st.subheader("Feature Importance Analysis")
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': model.feature_name_,
        'importance': model.feature_importances_
    })
    
    
    # 构建 feature_dict，忽略键中的最后一部分
    feature_dict = {
        "mean_outstandingdebt": "Average outstanding debt",
        "max_empl_employedfrom": "Longest employment duration",
        "mean_numberofoverdueinstlmax": "Average max overdue installment count",
        "isbidproduct": "Whether it's a bid product",
        "max_numberofoverdueinstlmaxdat": "Max overdue installment count date range",
        "mean_dateofcredstart": "Average credit start date",
        "mean_dpdmaxdateyear": "Average year of max overdue days",
        "validfrom": "Record validity start date",
        "max_totalamount": "Max total amount",
        "max_birth": "Max birth-related offset",
        "pmtsum": "Total payments sum",
        "pmtssum": "Total payments sum",
        "days30": "Events within 30 days",
        "days180": "Events within 180 days",
        "mean_maxdpdtolerance": "Average max overdue tolerance",
        "mean_pmts_overdue": "Average overdue payment amount",
        "max_incometype": "Max income type",
        "pmtnum": "Number of payments",
        "max_sex": "Gender-related statistic",
        "price": "Transaction price",
        "mean_pmts_dpd": "Average overdue payment days",
        "default": "Feature not defined"  # Default explanation for unknown features
    }

    # 将特征名称统一为去掉最后一部分的形式
    def preprocess_feature_name(feature_name):
        return feature_name.rsplit("_", 1)[0]  # 去掉最后的 `_` 和后续部分

    # 对 feature_dict 的映射进行处理
    importance_df['description'] = (
        importance_df['feature']
        .map(lambda x: preprocess_feature_name(x))  # 预处理特征名称
        .map(feature_dict)  # 映射到解释
        .fillna('Feature not defined')  # 如果映射不到，使用默认值
    )

    # 合成完整描述
    # importance_df['description'] = importance_df['feature'] + " (" + importance_df['description'] + ")"
    # importance_df['description'] = importance_df['feature'] + " (" + importance_df['description'] + ")"

    importance_df = importance_df.sort_values('importance', ascending=False).head(20)

    # Create bar chart for feature importance
    fig = px.bar(importance_df, 
                 x='importance', 
                 y='description',
                 orientation='h',
                 title='Top 20 Most Important Features',
                 labels={'importance': 'Importance Score', 'feature': 'Feature Name'},
                 color='importance',
                 color_continuous_scale='Viridis')
    fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
    # fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)


        
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import streamlit as st

def get_risk_percentile(pred_proba, mu=0.12, sigma=0.03):
    """
    将预测概率转换为基于正态分布的百分位数
    参数基于观察到的 N(0.12, 0.03) 分布
    """
    z_score = (pred_proba - mu) / sigma
    percentile = norm.cdf(z_score) * 100
    return np.clip(percentile, 0, 100)

def get_risk_level(percentile):
    """获取风险等级文本"""
    if percentile < 30:
        return "LOW RISK"
    elif percentile < 70:
        return "MEDIUM RISK"
    else:
        return "HIGH RISK"

def show_risk_distribution(pred_proba):
    """Display risk evaluation details"""
    percentile_score = get_risk_percentile(pred_proba)
    risk_level = get_risk_level(percentile_score)
    
    st.subheader("Risk Evaluation Details")
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk Gauge Chart - 改进的显示方式
        fig = go.Figure(go.Indicator(
            mode="gauge+number",  # 移除delta模式
            value=percentile_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            number={
                'suffix': "%",  # 添加百分号
                'font': {'size': 50},
            },
            title={
                'text': f"<b>{risk_level}</b>",  # 在数字下方显示风险等级
                'font': {'size': 24},
            },
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'thickness': 0.75,
                    'value': percentile_score
                }
            }
        ))
        
        # 调整图表布局
        fig.update_layout(
            height=400,
            margin=dict(l=30, r=30, t=50, b=30),
            font={'size': 16}
        )
        
        st.plotly_chart(fig)
    
    with col2:
        st.write("### Risk Level Classification")
        if percentile_score < 30:
            st.success("Low Risk (Bottom 30%)")
        elif percentile_score < 70:
            st.warning("Medium Risk (30%-70%)")
        else:
            st.error("High Risk (Top 30%)")
            
        st.write(f"""
        #### Risk Evaluation Result
        - Original Score: {pred_proba:.2%}
        - Risk Percentile: {percentile_score:.1f}%
        - Interpretation: This applicant's risk level is higher than {percentile_score:.1f}% of the population
        - Recommendation: {'Approve Loan' if percentile_score < 70 else 'Evaluate Cautiously'}
        """)
        
        
import numpy as np
import pandas as pd
import random
import time

def random_parameter_combination():
    """随机生成一组参数组合"""
    return {key: random.choice(values) for key, values in param_grid.items()}

def stream_predictions(model, base_input_data, num_samples=100):
    """流式执行随机采样预测并实时保存结果"""
    
    # 创建进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 检查CSV文件是否存在，决定是否需要写入表头
    file_exists = os.path.exists('save_cuipy.csv')
    
    # 记录成功的预测次数
    successful_predictions = 0
    
    for i in range(num_samples):
        # 随机生成参数组合
        current_input = base_input_data.copy()
        random_params = random_parameter_combination()
        current_input.update(random_params)
        
        try:
            # 准备数据并进行预测
            df_pred = prepare_data_for_model(current_input)
            pred_proba = model.predict_proba(df_pred)[0][1]
            
            # 准备结果字典
            result_dict = current_input.copy()
            result_dict['prediction_probability'] = pred_proba
            
            # 转换为DataFrame并追加到CSV
            result_df = pd.DataFrame([result_dict])
            result_df.to_csv('save_cuipy.csv', 
                           mode='a', 
                           header=not file_exists and i == 0,
                           index=False)
            
            successful_predictions += 1
            file_exists = True
            
            # 更新进度
            progress_bar.progress((i + 1) / num_samples)
            status_text.text(f'完成 {i+1}/{num_samples} 预测 (成功: {successful_predictions})')
                
        except Exception as e:
            print(f"Error with combination {random_params}: {str(e)}")
            continue
            
    return successful_predictions



# Update the prediction section in the main code
if st.sidebar.button('Make Prediction'):
    try:
        
        # # FIXME: begin debug
        # # 执行流式预测
        # successful_count = stream_predictions(model, input_data, num_samples=100)
        
        # # 显示完成信息
        # st.write("### Grid Search Complete")
        # st.write(f"Successfully completed {successful_count} predictions")
        # st.write("Results saved to 'save_cuipy.csv'")
        
        # # 读取并显示最新结果的统计信息
        # if os.path.exists('save_cuipy.csv'):
        #     results_df = pd.read_csv('save_cuipy.csv')
        #     st.write("### Results Summary")
        #     st.write("Prediction probability statistics:")
        #     st.write(results_df['prediction_probability'].describe())
        #     # 创建可视化
        #     fig = px.histogram(results_df, x='prediction_probability',
        #                      title='Distribution of Prediction Probabilities')
        #     st.plotly_chart(fig)
        # # FIXME: end debug
        
        
        df_pred = prepare_data_for_model(input_data)
        # print(input_data)
        # breakpoint()
        # {'credamount_770A': 100000.0, 'month_decision': 3, 'weekday_decision': 4, 'price_1097A': 50000.0, 'days30_165L': 1, 'days180_256L': 2, 'mean_maxdpdtolerance_577P': 15.0, 'mean_pmts_overdue_1140A': 200.0, 'mean_pmts_dpd_1073P': 5.0, 'homephncnt_628L': 2, 'mobilephncnt_593L': 5, 'numactivecreds_622L': 4, 'pmtnum_254L': 10, 'max_sex_738L': 'MISSING', 'max_incometype_1044T': 'MISSING', 'mean_outstandingdebt_522A': 10000.0, 'max_totalamount_6A': 200000.0, 'max_empl_employedfrom_271D': 5.0}
        
        # # Log the prepared data for debugging
        # st.write("### Debug Information")
        # st.write("Required features:", len(model.feature_name_))
        # st.write("Provided features:", len(df_pred.columns))
        # st.write("Missing features:", set(model.feature_name_) - set(df_pred.columns))
        
        # Make prediction
        pred_proba = model.predict_proba(df_pred)[0][1]
        
        # Create tabs for organizing content
        tab1, tab2, tab3 = st.tabs(["Prediction Result", "Feature Analysis", "Risk Details"])
        
        with tab1:
            show_risk_distribution(pred_proba)
            
        with tab2:
            show_feature_importance(model)
            
        with tab3:
            st.subheader("Detailed Input Data")
            st.write("Data Types:")
            st.write(df_pred.dtypes)
            st.write("Input Data Example:")
            st.write(df_pred)
            
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.write("Error Type:", type(e).__name__)
        st.write("Error Details:", str(e))
        # Add traceback for better debugging
        import traceback
        st.write("Complete Error Details:", traceback.format_exc())

# Footer Section
st.markdown("""
---
### Instructions
1. Fill in customer information in the left sidebar.
2. Click "Make Prediction" to get prediction results.
3. Check analysis results in different tabs.
4. Feature Importance Chart displays the most important features according to the model.
5. Risk Details page provides a more detailed risk assessment.
""")
