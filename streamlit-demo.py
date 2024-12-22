import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# 页面配置
st.set_page_config(page_title="信贷违约预测模型Demo", layout="wide")
st.title("信贷违约预测模型Demo")

# 加载模型和必要文件
@st.cache_resource
def load_model_and_files():
    model = joblib.load('lgbm_best_model.pkl')
    categorical_features = joblib.load('categorical_features.pkl')
    column_dtypes = joblib.load('column_dtypes.pkl')
    return model, categorical_features, column_dtypes

try:
    model, categorical_features, column_dtypes = load_model_and_files()
    st.success("模型和配置文件加载成功!")
except Exception as e:
    st.error(f"模型加载失败: {str(e)}")
    st.stop()

# 侧边栏 - 数据输入区域
st.sidebar.header("输入预测数据")

def create_input_fields():
    # 创建主要输入字段
    input_data = {}
    
    # 基础信息
    st.sidebar.subheader("基础信息")
    input_data['credamount_770A'] = st.sidebar.number_input('信贷金额', value=100000.0)
    input_data['month_decision'] = st.sidebar.slider('决策月份', 1, 12, 6)
    input_data['weekday_decision'] = st.sidebar.slider('决策星期', 0, 6, 3)
    
    # 客户信息
    st.sidebar.subheader("客户信息")
    input_data['numactivecreds_622L'] = st.sidebar.number_input('当前活跃信贷数量', value=1)
    input_data['homephncnt_628L'] = st.sidebar.number_input('家庭电话数量', value=1)
    input_data['mobilephncnt_593L'] = st.sidebar.number_input('手机号码数量', value=1)
    
    return input_data

# 主要内容区域
input_data = create_input_fields()

# 预测按钮
if st.sidebar.button('进行预测'):
    # 准备数据
    df_pred = pd.DataFrame([input_data])
    
    # 补充缺失列
    missing_cols = [col for col in model.feature_name_ if col not in df_pred.columns]
    for col in missing_cols:
        df_pred[col] = 0
    
    # 确保列顺序与训练时一致
    df_pred = df_pred[model.feature_name_]
    
    # 进行预测
    try:
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
                        {'range': [70, 100], 'color': "lightred"}
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

# 添加特征重要性可视化
if st.checkbox("显示特征重要性"):
    st.subheader("模型特征重要性")
    
    importance_df = pd.DataFrame({
        'feature': model.feature_name_,
        'importance': model.feature_importances_
    })
    importance_df = importance_df.sort_values('importance', ascending=False).head(20)
    
    fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                 title='Top 20 最重要特征')
    st.plotly_chart(fig)

# 添加模型说明
with st.expander("查看模型说明"):
    st.markdown("""
    ### 模型信息
    - 使用的模型: LightGBM
    - 模型评估指标: AUC
    - 模型验证方法: 5折交叉验证
    
    ### 使用说明
    1. 在左侧边栏输入客户信息
    2. 点击"进行预测"按钮获取预测结果
    3. 查看预测结果和风险评估
    4. 可选择显示特征重要性分析
    
    ### 注意事项
    - 所有输入字段都需要填写
    - 预测结果仅供参考，请结合实际情况判断
    """)
