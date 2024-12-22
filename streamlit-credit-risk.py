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
st.set_page_config(page_title="Credit Default Risk Analysis", layout="wide")
st.title("Credit Default Risk Analysis Demo")

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

def create_basic_analysis(df):
    """创建基础数据分析"""
    try:
        figures = []
        
        # TARGET分布
        if 'target' in df.columns:
            fig1 = px.pie(values=df['target'].value_counts().values,
                         names=['Not Default', 'Default'],
                         title='Target Variable Distribution')
            figures.append(fig1)
        
        # 月度分布
        if 'month_decision' in df.columns:
            monthly_counts = df['month_decision'].value_counts().sort_index()
            fig2 = px.line(x=monthly_counts.index, 
                          y=monthly_counts.values,
                          title='Monthly Application Distribution',
                          labels={'x': 'Month', 'y': 'Number of Applications'})
            figures.append(fig2)
            
        # 周分布
        if 'WEEK_NUM' in df.columns:
            weekly_counts = df['WEEK_NUM'].value_counts().sort_index()
            fig3 = px.line(x=weekly_counts.index,
                          y=weekly_counts.values,
                          title='Weekly Application Distribution',
                          labels={'x': 'Week', 'y': 'Number of Applications'})
            figures.append(fig3)
        
        return figures
    except Exception as e:
        st.error(f"分析创建失败: {str(e)}")
        st.write("错误类型:", type(e).__name__)
        st.write("完整错误信息:", str(e))
        return []

def create_summary_stats(df):
    """创建数据摘要统计"""
    try:
        summary = pd.DataFrame({
            'Records': len(df),
            'Columns': len(df.columns),
            'Missing Values (%)': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100),
        }, index=['Value']).T
        
        # 添加目标变量统计（如果存在）
        if 'target' in df.columns:
            default_rate = (df['target'].sum() / len(df)) * 100
            summary.loc['Default Rate (%)'] = default_rate
            
        return summary
    except Exception as e:
        st.error(f"统计创建失败: {str(e)}")
        return None

def explore_columns(df):
    """探索数据列的基本信息"""
    try:
        col_info = pd.DataFrame({
            'Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Null Percentage': (df.isnull().sum() / len(df)) * 100
        })
        return col_info
    except Exception as e:
        st.error(f"列信息统计失败: {str(e)}")
        return None

def main():
    # 创建标签页
    tab1, tab2 = st.tabs(["Basic Analysis", "Detailed Analysis"])
    
    with tab1:
        st.markdown("""
        ### Basic Data Analysis
        
        This tab shows basic statistics and distributions from the base data file.
        """)
        
        # 加载基础数据
        df_base = load_base_data()
        
        if df_base is not None:
            # 显示基本统计信息
            st.subheader("Data Summary")
            summary = create_summary_stats(df_base)
            if summary is not None:
                st.write(summary)
            
            # 显示基础分析图表
            figures = create_basic_analysis(df_base)
            
            # 使用两列布局显示图表
            if figures:
                col1, col2 = st.columns(2)
                cols = [col1, col2]
                for idx, fig in enumerate(figures):
                    with cols[idx % 2]:
                        st.plotly_chart(fig, use_container_width=True)
            
            # 显示列信息
            st.subheader("Column Information")
            col_info = explore_columns(df_base)
            if col_info is not None:
                st.write(col_info)
            
            # 清理内存
            del df_base
            gc.collect()
    
    with tab2:
        st.markdown("""
        ### Detailed Analysis
        
        Select which aspects of the data you want to analyze:
        """)
        
        analysis_type = st.selectbox(
            "Choose analysis type",
            ["Static Data Analysis", "Credit Bureau Analysis", "Application Analysis"]
        )
        
        if analysis_type == "Static Data Analysis":
            df_static_cb, df_static_0 = load_static_data()
            
            if df_static_cb is not None and df_static_0 is not None:
                st.subheader("Static Data Summary")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Static CB Data Summary")
                    st.write(create_summary_stats(df_static_cb))
                    st.write("Column Information")
                    st.write(explore_columns(df_static_cb))
                
                with col2:
                    st.write("Static 0 Data Summary")
                    st.write(create_summary_stats(df_static_0))
                    st.write("Column Information")
                    st.write(explore_columns(df_static_0))
                
                # 清理内存
                del df_static_cb, df_static_0
                gc.collect()

# 运行应用
if __name__ == "__main__":
    main()