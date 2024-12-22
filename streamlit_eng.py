import streamlit as st
import pandas as pd
import numpy as np
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
from datetime import datetime, timezone

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
    """Prepare default values for input fields"""
    defaults = {}
    for col, dtype in column_types.items():
        if col in categorical_columns:
            defaults[col] = '0'  # Default value for categorical features
        elif 'float' in str(dtype):
            defaults[col] = 0.0
        elif 'int' in str(dtype):
            defaults[col] = 0
        elif 'datetime' in str(dtype):
            defaults[col] = datetime.now().strftime('%Y-%m-%d')  # Default for datetime
        else:
            defaults[col] = '0'
    return defaults

def create_input_fields():
    """Create input fields for user input in the sidebar"""
    input_data = {}
    
    # Basic Information
    st.sidebar.subheader("Basic Information")
    input_data['credamount_770A'] = st.sidebar.number_input('Credit Amount', value=100000.0)
    input_data['month_decision'] = int(st.sidebar.slider('Decision Month', 1, 12, 3))
    input_data['weekday_decision'] = int(st.sidebar.slider('Decision Day of Week', 0, 6, 4))
    input_data['price_1097A'] = st.sidebar.number_input('Transaction Price', value=50000.0)
    input_data['days30_165L'] = int(st.sidebar.number_input('Events in Last 30 Days', value=1))
    input_data['days180_256L'] = int(st.sidebar.number_input('Events in Last 180 Days', value=2))
    input_data['mean_maxdpdtolerance_577P'] = st.sidebar.number_input('Avg. Max Overdue Tolerance', value=15.0)
    input_data['mean_pmts_overdue_1140A'] = st.sidebar.number_input('Avg. Overdue Payment Amount', value=200.0)
    input_data['mean_pmts_dpd_1073P'] = st.sidebar.number_input('Avg. Overdue Payment Days', value=5.0)

    
    # Customer Information
    st.sidebar.subheader("Customer Information")
    input_data['homephncnt_628L'] = int(st.sidebar.number_input('Home Phone Count', value=2))
    input_data['mobilephncnt_593L'] = int(st.sidebar.number_input('Mobile Phone Count', value=5))
    input_data['numactivecreds_622L'] = int(st.sidebar.number_input('Active Credit Count', value=4))
    input_data['pmtnum_254L'] = int(st.sidebar.number_input('Payment Count', value=10))
    input_data['max_sex_738L'] = st.sidebar.selectbox('Gender Statistic', options=['Male', 'Female'], index=0)
    input_data['max_incometype_1044T'] = st.sidebar.selectbox('Highest Income Type', options=['Type A', 'Type B', 'Type C'], index=0)

      
    st.sidebar.subheader("Historical Data")
    input_data['mean_outstandingdebt_522A'] = st.sidebar.number_input('Avg. Outstanding Debt', value=10000.0)
    input_data['max_totalamount_6A'] = st.sidebar.number_input('Max Total Amount', value=200000.0)
    input_data['max_empl_employedfrom_271D'] = st.sidebar.number_input('Longest Employment Duration (years)', value=5.0)

    
    # Add default values for all required features
    defaults = prepare_default_values(column_types, categorical_columns)
    for col, default_value in defaults.items():
        if col not in input_data:
            input_data[col] = default_value
    
    return input_data

def prepare_data_for_model(input_data):
    """Prepare input data for the model"""
    # Create a DataFrame
    df = pd.DataFrame([input_data])
    
    # Ensure all categorical columns exist
    for col in categorical_columns:
        if col not in df.columns:
            df[col] = '0'
    
    # Convert columns to appropriate data types
    for col, dtype in column_types.items():
        if col in df.columns:
            try:
                if col in categorical_columns:
                    df[col] = df[col].astype(str)  # Ensure categorical features are strings
                elif 'datetime' in str(dtype):
                    df[col] = pd.to_datetime(df[col])
                else:
                    df[col] = df[col].astype(dtype)
            except Exception as e:
                st.error(f"Error converting column {col} to {dtype}: {str(e)}")
    
    # Ensure column order matches the model
    df = df[model.feature_name_]
    
    return df

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
    importance_df['description'] = importance_df['feature'] + " (" + importance_df['description'] + ")"


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
    # fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

def show_risk_distribution(pred_proba):
    """Display risk evaluation details"""
    st.subheader("Risk Evaluation Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred_proba * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Default Probability (%)"},
            delta={'reference': 50},
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
        # Risk Classification Details
        st.write("### Risk Level Classification")
        if pred_proba < 0.3:
            st.success("Low Risk (< 30%)")
        elif pred_proba < 0.7:
            st.warning("Medium Risk (30%-70%)")
        else:
            st.error("High Risk (> 70%)")
        
        st.write(f"""
        #### Risk Evaluation Result
        - Default Probability: {pred_proba:.2%}
        - Risk Level: {'Low' if pred_proba < 0.3 else 'Medium' if pred_proba < 0.7 else 'High'}
        - Recommendation: {'Approve Loan' if pred_proba < 0.5 else 'Evaluate Cautiously'}
        """)

# Prediction and Analysis Section
if st.sidebar.button('Make Prediction'):
    try:
        df_pred = prepare_data_for_model(input_data)
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
