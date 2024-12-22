import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os

def load_data_and_model():
    """加载所有必要的数据和模型"""
    try:
        # 加载原始数据
        data = pd.read_csv('/Users/wuqianran/Desktop/bigdata_finalproject/final/processed_data.csv')
        
        # 加载模型
        model = joblib.load('/Users/wuqianran/Desktop/bigdata_finalproject/final/lgbm_best_model.pkl')
        
        # 加载分类特征列表
        categorical_features = joblib.load('/Users/wuqianran/Desktop/bigdata_finalproject/final/categorical_features.pkl')
        
        # 确保分类特征的类型正确
        for col in categorical_features:
            if col in data.columns:
                data[col] = data[col].astype('category')
        
        return data, model, categorical_features
    except Exception as e:
        st.error(f"Error loading data and model: {str(e)}")
        return None, None, None

def prepare_input_features(data):
    """准备输入特征"""
    input_values = {}
    
    # 月份和星期几的输入
    st.sidebar.subheader("Decision Features")
    input_values['month_decision'] = st.sidebar.slider('Month Decision', 1, 12, 1)
    input_values['weekday_decision'] = st.sidebar.slider('Weekday Decision', 1, 7, 1)
    
    # 数值特征
    st.sidebar.subheader("Numeric Features")
    numeric_features = [
        'credamount_770A', 'applicationcnt_361L', 'applications30d_658L',
        'applicationscnt_1086L', 'applicationscnt_464L', 'applicationscnt_867L',
        'clientscnt_1022L', 'clientscnt_100L'
    ]
    
    for feature in numeric_features:
        if feature in data.columns:
            min_val = float(data[feature].min())
            max_val = float(data[feature].max())
            default_val = float(data[feature].median())
            input_values[feature] = st.sidebar.slider(
                f"{feature.replace('_', ' ').title()}",
                min_value=min_val,
                max_value=max_val,
                value=default_val
            )
    
    return input_values

def create_input_df(input_values, data, categorical_features):
    """创建输入DataFrame并确保分类特征正确"""
    # 创建包含所有列的DataFrame
    input_df = pd.DataFrame(columns=data.columns)
    input_df.loc[0] = data.iloc[0].copy()  # 使用第一行作为模板
    
    # 填充用户输入的值
    for col, val in input_values.items():
        input_df.loc[0, col] = val
    
    # 确保所有分类特征的类型正确
    for col in categorical_features:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype('category')
            # 确保类别与原始数据相同
            if col in data.columns:
                input_df[col] = input_df[col].cat.set_categories(data[col].cat.categories)
    
    return input_df

def main():
    st.title("Loan Approval Simulator")
    st.markdown("Simulate loan approval decisions based on user inputs.")
    
    # 加载数据和模型
    data, model, categorical_features = load_data_and_model()
    if data is None or model is None or categorical_features is None:
        return
    
    st.write("Data shape:", data.shape)
    
    # 获取用户输入
    input_values = prepare_input_features(data)
    
    # 添加预测按钮
    if st.button("Predict"):
        try:
            # 创建预测用的DataFrame
            input_df = create_input_df(input_values, data, categorical_features)
            
            # 调试信息
            with st.expander("Debug Information"):
                st.write("Input Data Types:")
                st.write(input_df.dtypes)
                st.write("\nCategorical Features:")
                st.write(categorical_features)
                st.write("\nInput DataFrame Sample:")
                st.write(input_df)
            
            # 进行预测
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]
            
            # 显示结果
            st.subheader("Prediction Result")
            if prediction == 1:
                st.success("Loan Approved ✅")
            else:
                st.error("Loan Rejected ❌")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Approval Probability", f"{prediction_proba[1]:.2%}")
            with col2:
                st.metric("Rejection Probability", f"{prediction_proba[0]:.2%}")
            
        except Exception as e:
            st.error("Prediction Error")
            st.error(str(e))
            st.write("Full error trace:", e)

if __name__ == "__main__":
    main()