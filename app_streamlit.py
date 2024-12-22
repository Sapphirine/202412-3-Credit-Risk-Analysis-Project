# 导入必要的库
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import joblib

# 加载处理后的数据
def load_data():
    data = pd.read_csv('/Users/wuqianran/Desktop/bigdata_finalproject/final/processed_data.csv')
    return data

# 加载保存好的模型
def load_model():
    lgbm_model = joblib.load('/Users/wuqianran/Desktop/bigdata_finalproject/final/lgbm_model.pkl')
    catboost_model = joblib.load('/Users/wuqianran/Desktop/bigdata_finalproject/final/catboost_model.pkl')
    return lgbm_model, catboost_model

# 加载CatBoost和LGBMClassifier的训练结果
def load_results():
    catboost_results = pd.read_csv('/Users/wuqianran/Desktop/bigdata_finalproject/final/catboost_results.csv')
    lgbm_results = pd.read_csv('/Users/wuqianran/Desktop/bigdata_finalproject/final/lgbm_results.csv')
    return catboost_results, lgbm_results

# Streamlit应用程序
def main():
    st.title("Loan Approval Simulator")
    st.markdown("Simulate loan approval decisions based on user inputs.")

    # 加载数据和模型
    data = load_data()
    lgbm_model, catboost_model = load_model()

    # 加载训练结果
    catboost_results, lgbm_results = load_results()

    # 用户输入
    st.sidebar.header("Input Features")
    # 根据实际数据的特征进行调整
    income = st.sidebar.slider("Income ($)", 1000, 10000, 5000)
    age = st.sidebar.slider("Age", 18, 60, 30)
    transaction_count = st.sidebar.slider("Transaction Count", 1, 50, 10)

    # 预测
    input_data = pd.DataFrame({
        "Income": [income],
        "Age": [age],
        "Transaction_Count": [transaction_count]
    })
    prediction = lgbm_model.predict(input_data)[0]
    prediction_proba = lgbm_model.predict_proba(input_data)[0]

    # 展示预测结果
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"Loan Approved ✅")
    else:
        st.error(f"Loan Rejected ❌")

    st.write("Probability of Approval:", f"{prediction_proba[1]:.2f}")
    st.write("Probability of Rejection:", f"{prediction_proba[0]::.2f}")

    # 数据可视化
    st.subheader("Feature Distribution in Training Data")
    st.bar_chart(data.describe().T)

    st.subheader("Training Data Preview")
    st.write(data.head())

    # 展示CatBoost和LGBMClassifier的训练结果
    st.subheader("CatBoost Training Results")
    st.write(catboost_results)

    st.subheader("LGBMClassifier Training Results")
    st.write(lgbm_results)

if __name__ == "__main__":
    main()
