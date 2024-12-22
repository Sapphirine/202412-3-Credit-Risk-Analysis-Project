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

# 页面配置
st.set_page_config(page_title="Credit Default Risk Analysis & Prediction", layout="wide")
st.title("Credit Default Risk Analysis & Prediction System")

# 设置数据路径
DATA_PATH = Path("parquet_files/train")

@st.cache_data
def load_base_data():
    """加载基础数据"""
    try:
        df_base = pl.read_parquet(DATA_PATH / "train_base.parquet")
        return df_base.to_pandas()
    except Exception as e:
        st.error(f"基础数据加载失败: {str(e)}")
        return None

@st.cache_data
def load_static_data():
    """加载static数据"""
    try:
        df_static_cb = pl.read_parquet(DATA_PATH / "train_static_cb_0.parquet")
        df_static_0 = pl.concat([
            pl.read_parquet(DATA_PATH / "train_static_0_0.parquet"),
            pl.read_parquet(DATA_PATH / "train_static_0_1.parquet")
        ])
        return df_static_cb.to_pandas(), df_static_0.to_pandas()
    except Exception as e:
        st.error(f"Static数据加载失败: {str(e)}")
        return None, None

@st.cache_resource
def load_model_and_types():
    """加载模型和数据类型"""
    try:
        model = joblib.load('lgbm_best_model.pkl')
        column_types = pd.read_csv('column_dtypes.csv', index_col=0).squeeze()
        # 获取分类特征列名
        categorical_columns = column_types[column_types == 'object'].index.tolist()
        
        # 验证加载的数据
        if model is None:
            raise ValueError("模型加载失败")
        if column_types.empty:
            raise ValueError("列类型数据为空")
            
        return model, column_types, categorical_columns
    except Exception as e:
        st.error(f"模型和配置加载失败: {str(e)}")
        return None, None, None

def create_basic_analysis(df):
    """创建基础数据分析"""
    try:
        if df is None:
            return []
            
        figures = []
        
        # TARGET分布
        if 'target' in df.columns:
            target_counts = df['target'].value_counts()
            fig1 = px.pie(values=target_counts.values,
                         names=['Not Default', 'Default'],
                         title='违约分布')
            figures.append(fig1)
        
        # 月度分布
        if 'month_decision' in df.columns:
            monthly_counts = df['month_decision'].value_counts().sort_index()
            fig2 = px.line(x=monthly_counts.index, 
                          y=monthly_counts.values,
                          title='月度申请分布',
                          labels={'x': '月份', 'y': '申请数量'})
            figures.append(fig2)
            
        return figures
    except Exception as e:
        st.error(f"分析创建失败: {str(e)}")
        return []

def create_prediction_inputs():
    """创建预测输入界面"""
    input_data = {}
    
    st.sidebar.header("申请信息输入")
    
    # 基础信息
    st.sidebar.subheader("基础信息")
    input_data['credamount_770A'] = st.sidebar.number_input('贷款金额', 
                                                           min_value=10000.0,
                                                           max_value=1000000.0,
                                                           value=100000.0, 
                                                           step=10000.0)
    input_data['month_decision'] = int(st.sidebar.slider('申请月份', 1, 12, 6))
    input_data['weekday_decision'] = int(st.sidebar.slider('申请星期', 0, 6, 3))
    
    # 客户信息
    st.sidebar.subheader("客户信息")
    input_data['homephncnt_628L'] = int(st.sidebar.number_input('家庭电话数量', 
                                                               value=1, 
                                                               min_value=0,
                                                               max_value=10))
    input_data['mobilephncnt_593L'] = int(st.sidebar.number_input('手机号码数量', 
                                                                 value=1, 
                                                                 min_value=0,
                                                                 max_value=10))
    input_data['numactivecreds_622L'] = int(st.sidebar.number_input('当前活跃贷款数', 
                                                                   value=0, 
                                                                   min_value=0,
                                                                   max_value=20))
    
    return input_data

def prepare_prediction_data(input_data, column_types, categorical_columns):
    """准备预测数据"""
    try:
        # 加载列的平均值
        column_means = pd.read_csv('column_means.csv', index_col=0).squeeze().to_dict()
        
        # 创建默认值字典
        default_values = {}
        for col in column_types.index:
            if col not in input_data:
                if col in categorical_columns:
                    default_values[col] = '0'
                elif 'float' in str(column_types[col]):
                    default_values[col] = column_means.get(col, 0)
                elif 'int' in str(column_types[col]):
                    default_values[col] = column_means.get(col, 0)
                else:
                    default_values[col] = '0'
        
        # 更新输入数据
        input_data.update(default_values)
        
        # 创建DataFrame
        df = pd.DataFrame([input_data])
        
        # 转换数据类型
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
                    st.error(f"列 {col} 转换失败: {str(e)}")
        
        return df
    except Exception as e:
        st.error(f"数据准备失败: {str(e)}")
        return None

def show_prediction_results(pred_proba, input_data):
    """显示预测结果"""
    try:
        st.subheader("预测结果")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 风险仪表盘
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred_proba * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "违约概率 (%)"},
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
            st.write("### 风险评估")
            if pred_proba < 0.3:
                st.success("低风险")
                recommendation = "建议批准"
            elif pred_proba < 0.7:
                st.warning("中等风险")
                recommendation = "需进一步评估"
            else:
                st.error("高风险")
                recommendation = "建议拒绝"
                
            st.write(f"""
            #### 评估详情
            - 违约概率: {pred_proba:.2%}
            - 建议操作: {recommendation}
            - 申请金额: {input_data['credamount_770A']:,.2f}
            - 活跃贷款数: {input_data['numactivecreds_622L']}
            """)
    except Exception as e:
        st.error(f"结果显示失败: {str(e)}")

def main():
    try:
        # 加载模型和配置
        model, column_types, categorical_columns = load_model_and_types()
        
        if model is None or column_types is None or categorical_columns is None:
            st.error("系统初始化失败：模型或配置文件加载失败")
            return
            
        # 创建主要标签页
        tab1, tab2, tab3 = st.tabs(["数据分析", "预测系统", "模型说明"])
        
        with tab1:
            st.markdown("### 数据分析")
            df_base = load_base_data()
            
            
            # 计算并保存每列的平均值
            column_means = df_base.mean().to_frame('mean')
            column_means.to_csv('column_means.csv')
            
            if df_base is not None:
                figures = create_basic_analysis(df_base)
                
                if figures:
                    for fig in figures:
                        st.plotly_chart(fig, use_container_width=True)
                
                del df_base
                gc.collect()
        
        with tab2:
            st.markdown("### 违约预测系统")
            
            # 创建输入界面
            input_data = create_prediction_inputs()
            
            # 添加预测按钮
            if st.sidebar.button('进行预测'):
                # 准备预测数据
                df_pred = prepare_prediction_data(input_data, column_types, categorical_columns)
                
                if df_pred is not None:
                    try:
                        # 确保列顺序与模型一致
                        df_pred = df_pred[model.feature_name_]
                        # 进行预测
                        pred_proba = model.predict_proba(df_pred)[0][1]
                        # 显示预测结果
                        show_prediction_results(pred_proba, input_data)
                    except Exception as e:
                        st.error(f"预测过程出错: {str(e)}")
        
        with tab3:
            st.markdown("""
            ### 模型说明
            
            #### 使用的模型
            - 模型类型: LightGBM
            - 评估指标: AUC
            - 验证方法: 5折交叉验证
            
            #### 主要特征
            1. 基础信息
               - 贷款金额
               - 申请时间
            2. 客户特征
               - 联系方式
               - 现有贷款
            3. 行为特征
               - 历史违约
               - 还款表现
            
            #### 使用说明
            1. 在左侧输入栏填写申请信息
            2. 点击"进行预测"获取结果
            3. 查看预测结果和风险评估
            4. 参考建议进行决策
            """)
            
    except Exception as e:
        st.error(f"系统运行出错: {str(e)}")

if __name__ == "__main__":
    main()