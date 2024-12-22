import streamlit as st
import pandas as pd
import numpy as np
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
from datetime import datetime, timezone

# 页面配置
st.set_page_config(page_title="信贷违约预测模型Demo", layout="wide")
st.title("信贷违约预测模型Demo")

@st.cache_data
def compute_and_save_stats():
    """预计算统计数据并保存"""
    try:
        # 使用 polars 进行高效的统计计算
        df = pl.scan_csv('processed_data.csv')
        
        # 计算信贷金额分布
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
        
        # 计算月度申请统计
        monthly_stats = (df.select('month_decision')
                          .collect()
                          .group_by('month_decision')
                          .count()
                          .sort('month_decision')
                          .to_pandas())
        
        # 计算活跃信贷数量分布
        credit_count_stats = (df.select('numactivecreds_622L')
                              .collect()
                              .group_by('numactivecreds_622L')
                              .count()
                              .sort('numactivecreds_622L')
                              .to_pandas())
        
        # 计算联系方式统计
        contact_stats = {
            'mobile': df.select(pl.col('mobilephncnt_593L')).collect().mean().item(),
            'home': df.select(pl.col('homephncnt_628L')).collect().mean().item()
        }
        
        # 保存统计结果
        stats = {
            'credit_stats': credit_stats,
            'monthly_stats': monthly_stats,
            'credit_count_stats': credit_count_stats,
            'contact_stats': contact_stats
        }
        joblib.dump(stats, 'visualization_stats.joblib')
        return stats
        
    except Exception as e:
        st.error(f"统计计算失败: {str(e)}")
        st.write("错误详情:", type(e).__name__)
        st.write("完整错误信息:", str(e))
        return None

@st.cache_resource
def load_stats():
    """加载预计算的统计数据"""
    try:
        return joblib.load('visualization_stats.joblib')
    except FileNotFoundError:
        return compute_and_save_stats()
    except Exception as e:
        st.error(f"统计数据加载失败: {str(e)}")
        return None

def create_credit_amount_dist(stats):
    """创建信贷金额分布图"""
    credit_stats = stats['credit_stats']
    fig = px.bar(credit_stats,
                 x='range',
                 y='count',
                 title="信贷金额分布",
                 labels={'range': '信贷金额范围', 'count': '申请数量'})
    fig.update_layout(xaxis_title="信贷金额范围", yaxis_title="申请数量")
    return fig

def create_monthly_trend(stats):
    """创建月度申请趋势图"""
    monthly_stats = stats['monthly_stats']
    fig = px.line(monthly_stats,
                  x='month_decision',
                  y='count',
                  title="月度申请趋势",
                  labels={'month_decision': '月份', 'count': '申请数量'})
    fig.update_layout(xaxis_title="月份", yaxis_title="申请数量")
    return fig

def create_active_credits_dist(stats):
    """创建活跃信贷数量分布图"""
    credit_count_stats = stats['credit_count_stats']
    # 限制显示的信贷数量范围，避免极端值影响可视化效果
    credit_count_stats = credit_count_stats[credit_count_stats['numactivecreds_622L'] <= 10]
    
    fig = px.bar(credit_count_stats,
                 x='numactivecreds_622L',
                 y='count',
                 title="活跃信贷数量分布",
                 labels={'numactivecreds_622L': '活跃信贷数量', 'count': '客户数量'})
    fig.update_layout(xaxis_title="活跃信贷数量", yaxis_title="客户数量")
    return fig

def create_contact_info_dist(stats):
    """创建联系方式分布图"""
    contact_stats = stats['contact_stats']
    contact_df = pd.DataFrame({
        '类型': ['手机号码', '家庭电话'],
        '平均数量': [contact_stats['mobile'], contact_stats['home']]
    })
    
    fig = px.bar(contact_df,
                 x='类型',
                 y='平均数量',
                 title="平均联系方式数量",
                 labels={'类型': '联系方式类型', '平均数量': '平均数量'})
    fig.update_layout(xaxis_title="联系方式类型", yaxis_title="平均数量")
    return fig

# 加载预计算的统计数据
stats = load_stats()

if stats is not None:
    tab1, tab2 = st.tabs(["模型说明", "数据分析"])
    
    with tab1:
        with st.expander("模型介绍"):
            st.markdown("""
            ### 模型介绍
            - 使用的模型: LightGBM
            - 评估指标: AUC
            - 验证方法: 5折交叉验证
            - 模型特点: 适用于信贷违约风险预测，考虑多维度特征

            ### 主要特征类别
            1. 基础信息
                - 信贷金额
                - 申请时间
            2. 客户属性
                - 联系方式数量
                - 活跃信贷数量
            3. 历史行为
                - 历史违约情况
                - 还款表现
            4. 信用评估
                - 信用额度
                - 负债情况
            """)
        
    with tab2:
        st.subheader("数据分布分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 信贷金额分布
            fig1 = create_credit_amount_dist(stats)
            st.plotly_chart(fig1, use_container_width=True)
            
            # 活跃信贷数量分布
            fig2 = create_active_credits_dist(stats)
            st.plotly_chart(fig2, use_container_width=True)
            
        with col2:
            # 月度申请趋势
            fig3 = create_monthly_trend(stats)
            st.plotly_chart(fig3, use_container_width=True)
            
            # 联系方式分布
            fig4 = create_contact_info_dist(stats)
            st.plotly_chart(fig4, use_container_width=True)
        
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

def show_feature_importance(model):
    """显示特征重要性分析"""
    st.subheader("特征重要性分析")
    
    # 获取特征重要性
    importance_df = pd.DataFrame({
        'feature': model.feature_name_,
        'importance': model.feature_importances_
    })
    importance_df = importance_df.sort_values('importance', ascending=False).head(20)

    # 创建特征重要性条形图
    fig = px.bar(importance_df, 
                 x='importance', 
                 y='feature',
                 orientation='h',
                 title='Top 20 最重要特征',
                 labels={'importance': '重要性得分', 'feature': '特征名称'},
                 color='importance',
                 color_continuous_scale='Viridis')
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

def show_risk_distribution(pred_proba):
    """显示风险分布分析"""
    st.subheader("风险评估详情")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 风险仪表盘
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = pred_proba * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "违约概率 (%)"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "salmon"}  # 修改这里，使用合法的颜色名称
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
        # 风险分类详情
        st.write("### 风险等级划分")
        if pred_proba < 0.3:
            st.success("低风险 (< 30%)")
            risk_level = "低风险"
            risk_color = "lightgreen"
        elif pred_proba < 0.7:
            st.warning("中等风险 (30% - 70%)")
            risk_level = "中等风险"
            risk_color = "yellow"
        else:
            st.error("高风险 (> 70%)")
            risk_level = "高风险"
            risk_color = "salmon"
        
        st.write(f"""
        #### 风险评估结果
        - 违约概率: {pred_proba:.2%}
        - 风险等级: {risk_level}
        - 建议操作: {'建议批准贷款' if pred_proba < 0.5 else '建议谨慎评估'}
        """)

        # 添加风险分布图
        risk_ranges = ['低风险(0-30%)', '中等风险(30-70%)', '高风险(70-100%)']
        risk_values = [30, 40, 30]  # 示例分布值
        
        fig2 = go.Figure(data=[
            go.Bar(
                x=risk_ranges,
                y=risk_values,
                marker_color=['lightgreen', 'yellow', 'salmon']
            )
        ])
        fig2.update_layout(
            title="风险分布统计",
            xaxis_title="风险等级",
            yaxis_title="占比 (%)"
        )
        st.plotly_chart(fig2)

def show_comparison_analysis(input_values, model_features):
    """显示特征比较分析"""
    st.subheader("重要指标分析")
    
    # 选择要显示的关键指标
    key_metrics = {
        '信贷金额': 'credamount_770A',
        '活跃信贷数': 'numactivecreds_622L',
        '联系方式数量': 'mobilephncnt_593L'
    }
    
    # 创建比较图表
    fig = go.Figure()
    
    for metric_name, metric_key in key_metrics.items():
        if metric_key in input_values:
            value = input_values[metric_key]
            fig.add_trace(go.Bar(
                name=metric_name,
                x=[metric_name],
                y=[value],
                text=[f"{value:,.2f}"],
                textposition='auto',
            ))
    
    fig.update_layout(
        title="关键指标值",
        showlegend=False,
        xaxis_title="指标名称",
        yaxis_title="指标值"
    )
    
    st.plotly_chart(fig)

# 修改预测按钮部分
if st.sidebar.button('进行预测'):
    try:
        df_pred = prepare_data_for_model(input_data)
        pred_proba = model.predict_proba(df_pred)[0][1]

        # 创建三个标签页来组织内容
        tab1, tab2, tab3 = st.tabs(["预测结果", "特征分析", "风险详情"])
        
        with tab1:
            show_risk_distribution(pred_proba)
            
        with tab2:
            show_feature_importance(model)
            show_comparison_analysis(input_data, model.feature_name_)
            
        with tab3:
            st.subheader("详细数据")
            st.write("输入数据类型：")
            st.write(df_pred.dtypes)
            st.write("输入数据示例：")
            st.write(df_pred)
            
    except Exception as e:
        st.error(f"预测过程中发生错误: {str(e)}")
        st.write("错误类型:", type(e).__name__)
        st.write("错误详情:", str(e))

# 添加页脚说明
st.markdown("""
---
### 使用说明
1. 在左侧输入栏填写客户信息
2. 点击"进行预测"按钮获取预测结果
3. 查看不同标签页中的分析结果
4. 特征重要性图表显示了模型认为最重要的特征
5. 风险详情页面提供了更详细的风险评估信息
""")
