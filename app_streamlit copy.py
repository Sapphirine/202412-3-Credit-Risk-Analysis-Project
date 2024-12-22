# 导入必要的库
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 生成模拟数据
def generate_data():
    np.random.seed(42)
    data = pd.DataFrame({
        "Income": np.random.uniform(1000, 10000, 500),
        "Age": np.random.randint(18, 60, 500),
        "Transaction_Count": np.random.randint(1, 50, 500),
        "Loan_Approved": np.random.choice([0, 1], 500, p=[0.4, 0.6])  # 0: Rejected, 1: Approved
    })
    return data

# 构建随机森林模型
def train_model(data):
    X = data[["Income", "Age", "Transaction_Count"]]
    y = data["Loan_Approved"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    return model

# Streamlit应用程序
def main():
    st.title("Loan Approval Simulator")
    st.markdown("Simulate loan approval decisions based on user inputs.")

    # 加载数据和模型
    data = generate_data()
    model = train_model(data)

    # 用户输入
    st.sidebar.header("Input Features")
    income = st.sidebar.slider("Income ($)", 1000, 10000, 5000)
    age = st.sidebar.slider("Age", 18, 60, 30)
    transaction_count = st.sidebar.slider("Transaction Count", 1, 50, 10)

    # 预测
    input_data = pd.DataFrame({
        "Income": [income],
        "Age": [age],
        "Transaction_Count": [transaction_count]
    })
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    # 展示预测结果
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"Loan Approved ✅")
    else:
        st.error(f"Loan Rejected ❌")

    st.write("Probability of Approval:", f"{prediction_proba[1]:.2f}")
    st.write("Probability of Rejection:", f"{prediction_proba[0]:.2f}")

    # 数据可视化
    st.subheader("Feature Distribution in Training Data")
    st.bar_chart(data[["Income", "Age", "Transaction_Count"]].describe().T)

    st.subheader("Training Data Preview")
    st.write(data.head())

if __name__ == "__main__":
    main()
