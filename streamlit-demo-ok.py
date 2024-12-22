import streamlit as st
import pandas as pd
import numpy as np
import polars as pl
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timezone

# 页面配置
st.set_page_config(page_title="信贷违约预测模型Demo", layout="wide")
st.title("信贷违约预测模型Demo")

def get_default_datetime():
    return pd.Timestamp('1970-01-01').to_datetime64()

@st.cache_resource
def load_model_and_types():
    model = joblib.load('lgbm_best_model.pkl')
    column_types = pd.read_csv('column_dtypes.csv', index_col=0).squeeze()
    
    # 获取所有分类特征列名
    categorical_columns = column_types[column_types == 'object'].index.tolist()
    
    return model, column_types, categorical_columns

try:
    model, column_types, categorical_columns = load_model_and_types()
    st.success("模型和配置文件加载成功!")
except Exception as e:
    st.error(f"模型加载失败: {str(e)}")
    st.stop()

def prepare_default_values(column_types, categorical_columns):
    defaults = {}
    for col, dtype in column_types.items():
        if col in categorical_columns:
            defaults[col] = '0'  # 分类特征默认值为'0'
        elif 'float' in str(dtype):
            defaults[col] = 0.0
        elif 'int' in str(dtype):
            defaults[col] = 0
        elif 'datetime' in str(dtype):
            defaults[col] = get_default_datetime()
        else:
            defaults[col] = '0'
    return defaults

def create_input_fields():
    input_data = {}
    
    # 基础信息
    st.sidebar.subheader("基础信息")
    input_data['credamount_770A'] = st.sidebar.number_input('信贷金额', value=100000.0)
    input_data['month_decision'] = int(st.sidebar.slider('决策月份', 1, 12, 3))
    input_data['weekday_decision'] = int(st.sidebar.slider('决策星期', 0, 6, 4))
    
    # 客户信息
    st.sidebar.subheader("客户信息")
    input_data['homephncnt_628L'] = int(st.sidebar.number_input('家庭电话数量', value=2))
    input_data['mobilephncnt_593L'] = int(st.sidebar.number_input('手机号码数量', value=5))
    input_data['numactivecreds_622L'] = int(st.sidebar.number_input('当前活跃信贷数量', value=4))
    
    # 添加所有必需的特征默认值
    defaults = prepare_default_values(column_types, categorical_columns)
    for col, default_value in defaults.items():
        if col not in input_data:
            input_data[col] = default_value
    
    return input_data

def prepare_data_for_model(input_data):
    # 创建DataFrame
    df = pd.DataFrame([input_data])
    
    # 确保所有分类特征列存在
    for col in categorical_columns:
        if col not in df.columns:
            df[col] = '0'
    
    # 根据列类型进行转换
    for col, dtype in column_types.items():
        if col in df.columns:
            try:
                if col in categorical_columns:
                    df[col] = df[col].astype(str)  # 确保分类特征是字符串类型
                elif 'datetime' in str(dtype):
                    df[col] = pd.to_datetime(df[col])
                else:
                    df[col] = df[col].astype(dtype)
            except Exception as e:
                st.error(f"列 {col} 转换为 {dtype} 类型时出错: {str(e)}")
    
    # 确保列顺序与模型一致
    df = df[model.feature_name_]
    
    return df

# 主要内容区域
input_data = create_input_fields()

# 预测按钮
if st.sidebar.button('进行预测'):
    try:
        # 准备数据
        df_pred = prepare_data_for_model(input_data)
        
        # 显示数据类型信息（调试用）
        st.write("数据类型信息：")
        st.write(df_pred.dtypes)
        
        # 显示数据示例（调试用）
        st.write("输入数据示例：")
        st.write(df_pred)
        
        # 确认所有必需的列都存在
        missing_cols = set(model.feature_name_) - set(df_pred.columns)
        if missing_cols:
            st.error(f"缺少以下列: {missing_cols}")
        
        # 进行预测
        pred_proba = model.predict_proba(df_pred)[0][1]
        
        # 显示预测结果
        st.header("预测结果")
        col1, col2 = st.columns(2)
        
        # 使用Gauge图显示违约概率
        with col1:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = pred_proba * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "违约概率 (%)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    },
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "salmon"}
                    ]
                }
            ))
            st.plotly_chart(fig)
        
        # 风险等级评估
        with col2:
            st.subheader("风险评估")
            if pred_proba < 0.3:
                st.success("低风险客户")
            elif pred_proba < 0.7:
                st.warning("中等风险客户")
            else:
                st.error("高风险客户")
            
            st.write("风险评估指标:")
            st.write(f"- 违约概率: {pred_proba:.2%}")
            st.write(f"- 建议操作: {'可以发放贷款' if pred_proba < 0.5 else '谨慎发放贷款'}")
            
    except Exception as e:
        st.error(f"预测过程中发生错误: {str(e)}")
        st.write("错误类型:", type(e).__name__)
        st.write("错误详情:", str(e))