import sys
from pathlib import Path
import os
import gc
import datetime
from glob import glob
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import StratifiedGroupKFold, TimeSeriesSplit
from sklearn.metrics import (roc_auc_score, precision_recall_curve, 
                           roc_curve, classification_report)
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer

import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('ggplot')
sns.set_palette("husl")

ROOT = '/Users/wuqianran/Desktop/bigdata_finalproject/final'

class Pipeline:
    @staticmethod
    def set_table_dtypes(df):
        for col in df.columns:
            if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.Int64))
            elif col in ["date_decision"]:
                df = df.with_columns(pl.col(col).cast(pl.Date))
            elif col[-1] in ("P", "A"):
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            elif col[-1] in ("M",):
                df = df.with_columns(pl.col(col).cast(pl.String))
            elif col[-1] in ("D",):
                df = df.with_columns(pl.col(col).cast(pl.Date))
        return df

    @staticmethod
    def handle_dates(df):
        for col in df.columns:
            if col[-1] in ("D",):
                df = df.with_columns(pl.col(col) - pl.col("date_decision"))
                df = df.with_columns(pl.col(col).dt.total_days())
        df = df.drop("date_decision", "MONTH")
        return df

    @staticmethod
    def filter_cols(df):
        # 过滤掉缺失值过多的列
        for col in df.columns:
            if col not in ["target", "case_id", "WEEK_NUM"]:
                isnull = df[col].is_null().mean()
                if isnull > 0.95:
                    df = df.drop(col)
        
        # 过滤掉基数过高的分类列
        for col in df.columns:
            if (col not in ["target", "case_id", "WEEK_NUM"]) & (df[col].dtype == pl.String):
                freq = df[col].n_unique()
                if (freq == 1) | (freq > 200):
                    df = df.drop(col)
        
        return df
    
class Aggregator:
    @staticmethod
    def num_expr(df):
        cols = [col for col in df.columns if col[-1] in ("P", "A")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in cols]
        return expr_max + expr_mean

    @staticmethod
    def date_expr(df):
        cols = [col for col in df.columns if col[-1] in ("D")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in cols]
        return expr_max + expr_mean

    @staticmethod
    def str_expr(df):
        cols = [col for col in df.columns if col[-1] in ("M",)]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in cols]
        return expr_max + expr_mean

    @staticmethod
    def other_expr(df):
        cols = [col for col in df.columns if col[-1] in ("T", "L")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in cols]
        return expr_max + expr_mean

    @staticmethod
    def count_expr(df):
        cols = [col for col in df.columns if "num_group" in col]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in cols]
        return expr_max + expr_mean

    @staticmethod
    def get_exprs(df):
        exprs = (Aggregator.num_expr(df) + 
                Aggregator.date_expr(df) + 
                Aggregator.str_expr(df) + 
                Aggregator.other_expr(df) + 
                Aggregator.count_expr(df))
        return exprs
    
class ExploratoryAnalysis:
    def __init__(self, df):
        self.df = df
        # self.numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        # self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        self.numeric_cols = df.select(pl.col(pl.Float64, pl.Int64)).columns
        self.categorical_cols = df.select(pl.col(pl.Utf8, pl.Categorical)).columns

        
    def plot_target_distribution(self):
        if 'target' in self.df.columns:
            plt.figure(figsize=(10, 6))
            self.df['target'].value_counts(normalize=True).plot(kind='bar')
            plt.title('Target Distribution')
            plt.xlabel('Target Value')
            plt.ylabel('Percentage')
            plt.tight_layout()
        return plt.gcf()
    
    def plot_numeric_distributions(self, cols=None):
        if cols is None:
            # 只选择前6个数值型特征
            cols = self.numeric_cols[:6]

        plt.figure(figsize=(15, 10))
        for i, col in enumerate(cols, 1):
            plt.subplot(2, 3, i)
            plt.hist(self.df[col].dropna(), bins=50)
            plt.title(f'Distribution of {col}')
        plt.tight_layout()
        return plt.gcf()
        
    def plot_correlation_matrix(self, top_n=10):
        corr_matrix = self.df[self.numeric_cols].corr()
        target_corrs = corr_matrix['target'].abs().sort_values(ascending=False)
        top_features = target_corrs[:top_n].index

        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix.loc[top_features, top_features],
                    annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Top Features')
        return plt.gcf()

    def plot_correlations(self, min_corr=0.1):
        numeric_df = self.df[self.numeric_cols].fillna(0)
        if 'target' in self.df.columns:
            numeric_df['target'] = self.df['target']

        correlations = numeric_df.corr()['target']
        correlations = correlations[abs(correlations) >= min_corr]
        correlations = correlations.sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        correlations.plot(kind='bar')
        plt.title('Features Correlation with Target')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return plt.gcf()

    def plot_missing_values(self):
        """绘制缺失值分布"""
        missing = (self.df.isnull().sum() / len(self.df)).sort_values(ascending=False)
        missing = missing[missing > 0]

        if len(missing) > 0:
            plt.figure(figsize=(10, 6))
            missing.plot(kind='barh')
            plt.title('Missing Values Percentage')
            plt.xlabel('Missing Ratio')
            plt.tight_layout()
        return plt.gcf()
    
class ModelEvaluator:
    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    def plot_roc_curve(self):
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        auc_score = roc_auc_score(self.y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        return plt.gcf()
        
    def get_classification_report(self):
        y_pred = self.model.predict(self.X_test)
        return classification_report(self.y_test, y_pred)

    def plot_feature_importance(self, top_n=20):
        importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=importance.head(top_n), x='importance', y='feature')
        plt.title(f'Top {top_n} Most Important Features')
        plt.xlabel('Feature Importance')
        return plt.gcf()


# 修改数据加载部分，只取部分数据
def read_file(path, depth=None, sample_size=10000):  # 添加sample_size参数
    df = pl.read_parquet(path)
    # 添加采样
    if sample_size:
        df = df.sample(n=min(sample_size, df.height))
    df = df.pipe(Pipeline.set_table_dtypes)
    if depth in [1,2]:
        df = df.group_by("case_id").agg(Aggregator.get_exprs(df))
    return df

def read_files(regex_path, depth=None, sample_size=10000):  # 添加sample_size参数
    chunks = []
    for path in glob(str(regex_path)):
        df = pl.read_parquet(path)
        # 添加采样
        if sample_size:
            df = df.sample(n=min(sample_size, df.height))
        df = df.pipe(Pipeline.set_table_dtypes)
        if depth in [1, 2]:
            df = df.group_by("case_id").agg(Aggregator.get_exprs(df))
        chunks.append(df)
    
    df = pl.concat(chunks, how="vertical_relaxed")
    df = df.unique(subset=["case_id"])
    return df

def feature_eng(df_base, depth_0, depth_1, depth_2):
    df_base = (
        df_base
        .with_columns(
            month_decision = pl.col("date_decision").dt.month(),
            weekday_decision = pl.col("date_decision").dt.weekday(),
        )
    )
    for i, df in enumerate(depth_0 + depth_1 + depth_2):
        df_base = df_base.join(df, how="left", on="case_id", suffix=f"_{i}")
    df_base = df_base.pipe(Pipeline.handle_dates)
    return df_base

def to_pandas(df_data, cat_cols=None):
    df_data = df_data.to_pandas()
    if cat_cols is None:
        cat_cols = list(df_data.select_dtypes("object").columns)
    df_data[cat_cols] = df_data[cat_cols].astype("category")
    return df_data, cat_cols


def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type)=="category":
            continue
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def analyze_feature_importance(model, feature_names, top_n=20):
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop {} Important Features:".format(top_n))
    print(importance.head(top_n))

    # 计算累积重要性
    importance['cumulative_importance'] = importance['importance'].cumsum()
    print("\nFeatures needed for 90% importance:",
          len(importance[importance['cumulative_importance'] <= 0.9]))

def main_full():
    # 接上文的数据加载和预处理部分
    
    # 特征编码
    cnt_encoding_cols = df_train.select(pl.selectors.by_dtype([pl.String, pl.Boolean, pl.Categorical])).columns
    
    mappings = {}
    for col in cnt_encoding_cols:
        mappings[col] = df_train.group_by(col).len()

    df_train_lazy = df_train.select(mappings.keys()).lazy()
    
    # 计数编码
    for col, mapping in mappings.items():
        remapping = {category: count for category, count in mapping.rows()}
        remapping[None] = -2
        expr = pl.col(col).replace(remapping, default=-1)
        df_train_lazy = df_train_lazy.with_columns(expr.alias(col + '_cnt'))
        del col, mapping, remapping
        gc.collect()

    del mappings
    transformed_train = df_train_lazy.collect()
    df_train = pl.concat([df_train, transformed_train.select("^*cnt$")], how='horizontal')
    del transformed_train, cnt_encoding_cols
    gc.collect()

    # 转换为pandas并优化内存
    df_train, cat_cols = to_pandas(df_train)
    df_train = reduce_mem_usage(df_train)
    print("Processed train data shape:\t", df_train.shape)

    # 准备训练数据
    y = df_train["target"]
    weeks = df_train["WEEK_NUM"]
    df_train = df_train.drop(columns=["target", "case_id", "WEEK_NUM"])

    # 设置交叉验证
    n_splits = 5
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=False)

    # 保存分类特征信息
    categorical_features = []
    for col in df_train.columns:
        if pd.api.types.is_categorical_dtype(df_train[col]) or df_train[col].dtype == 'object':
            categorical_features.append(col)

    # 保存必要信息
    joblib.dump(categorical_features, 'categorical_features.pkl')
    joblib.dump(df_train.dtypes, 'column_dtypes.pkl')
    df_train.dtypes.to_frame('dtype').to_csv('column_dtypes.csv')
    df_train.to_csv('processed_data.csv', index=False)

    # 模型参数
    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 100,
        "colsample_bytree": 0.6,
        "colsample_bynode": 0.6,
        "verbose": -1,
        "random_state": 42,
        "reg_alpha": 0.1,
        "reg_lambda": 1,
        "extra_trees": True,
        'num_leaves': 8,
        "min_data_in_leaf": 50,
        "device": "cpu",
    }

    # 模型训练
    fitted_models = []
    cv_scores = []
    best_auc = 0
    best_model = None
    fold_predictions = []
    
    print("Starting model training...")
    for fold, (idx_train, idx_valid) in enumerate(cv.split(df_train, y, groups=weeks), 1):
        print(f"\nFold {fold}")
        
        X_train, y_train = df_train.iloc[idx_train], y.iloc[idx_train]
        X_valid, y_valid = df_train.iloc[idx_valid], y.iloc[idx_valid]
        
        # 训练模型
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set = [(X_valid, y_valid)],
            callbacks = [lgb.log_evaluation(200), lgb.early_stopping(100)]
        )
        
        # 评估模型
        y_pred_valid = model.predict_proba(X_valid)[:,1]
        auc_score = roc_auc_score(y_valid, y_pred_valid)
        cv_scores.append(auc_score)
        
        # 创建评估器对象
        evaluator = ModelEvaluator(model, X_train, X_valid, y_train, y_valid)
        
        # 保存评估图表
        roc_curve = evaluator.plot_roc_curve()
        roc_curve.savefig(f'fold_{fold}_roc_curve.png')
        
        feature_importance = evaluator.plot_feature_importance()
        feature_importance.savefig(f'fold_{fold}_feature_importance.png')
        
        # 保存分类报告
        class_report = evaluator.get_classification_report()
        with open(f'fold_{fold}_classification_report.txt', 'w') as f:
            f.write(class_report)
        
        # 保存预测结果
        fold_predictions.append({
            'fold': fold,
            'predictions': y_pred_valid,
            'actual': y_valid,
            'auc_score': auc_score
        })
        
        # 更新最佳模型
        if auc_score > best_auc:
            best_auc = auc_score
            best_model = model

        fitted_models.append(model)

    # 保存最佳模型
    if best_model is not None:
        joblib.dump(best_model, 'lgbm_best_model.pkl')

    # 保存交叉验证结果
    cv_results = pd.DataFrame({
        'fold': range(1, n_splits + 1),
        'auc_score': cv_scores
    })
    cv_results.to_csv('lgbm_results.csv', index=False)

    # 打印最终结果
    print("\nTraining completed!")
    print("CV AUC scores:", cv_scores)
    print("Mean CV AUC score:", np.mean(cv_scores))
    print("Best CV AUC score:", best_auc)

    # 创建结果摘要
    results_summary = {
        'cv_scores': cv_scores,
        'mean_auc': np.mean(cv_scores),
        'best_auc': best_auc,
        'feature_importance': pd.DataFrame({
            'feature': df_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
    }

    # 保存结果摘要
    joblib.dump(results_summary, 'training_results.pkl')

# 修改模型参数
quick_params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "max_depth": 3,  # 减小深度
    "learning_rate": 0.1,  # 增大学习率
    "n_estimators": 10,  # 减少迭代次数
    "colsample_bytree": 0.6,
    "colsample_bynode": 0.6,
    "verbose": -1,
    "random_state": 42,
    "reg_alpha": 0.1,
    "reg_lambda": 1,
    "extra_trees": True,
    'num_leaves': 8,
    "min_data_in_leaf": 20,  # 减小最小叶子节点样本数
    "device": "cpu",
}

# 修改交叉验证折数
n_splits = 2  # 减少交叉验证次数

def main(sample_size=10000):  # 添加sample_size参数
    # 设置路径
    global ROOT, TRAIN_DIR, TEST_DIR
    ROOT = Path(ROOT)
    TRAIN_DIR = ROOT / "parquet_files" / "train"
    TEST_DIR = ROOT / "parquet_files" / "test"

    # 加载数据
    data_store = {
        "df_base": read_file(TRAIN_DIR / "train_base.parquet", sample_size=sample_size),
        "depth_0": [
            read_file(TRAIN_DIR / "train_static_cb_0.parquet", sample_size=sample_size),
            read_files(TRAIN_DIR / "train_static_0_*.parquet", sample_size=sample_size),
        ],
        "depth_1": [
            read_files(TRAIN_DIR / "train_applprev_1_*.parquet", 1, sample_size=sample_size),
            read_file(TRAIN_DIR / "train_tax_registry_a_1.parquet", 1, sample_size=sample_size),
            read_file(TRAIN_DIR / "train_tax_registry_b_1.parquet", 1, sample_size=sample_size),
            read_file(TRAIN_DIR / "train_tax_registry_c_1.parquet", 1, sample_size=sample_size),
            read_files(TRAIN_DIR / "train_credit_bureau_a_1_*.parquet", 1, sample_size=sample_size),
            read_file(TRAIN_DIR / "train_credit_bureau_b_1.parquet", 1, sample_size=sample_size),
            read_file(TRAIN_DIR / "train_other_1.parquet", 1, sample_size=sample_size),
            read_file(TRAIN_DIR / "train_person_1.parquet", 1, sample_size=sample_size),
            read_file(TRAIN_DIR / "train_deposit_1.parquet", 1, sample_size=sample_size),
            read_file(TRAIN_DIR / "train_debitcard_1.parquet", 1, sample_size=sample_size),
        ],
        "depth_2": [
            read_file(TRAIN_DIR / "train_credit_bureau_b_2.parquet", 2, sample_size=sample_size),
            read_files(TRAIN_DIR / "train_credit_bureau_a_2_*.parquet", 2, sample_size=sample_size),
        ]
    }

    print("Data loaded, starting feature engineering...")

    # 特征工程
    df_train = feature_eng(**data_store)
    print("Initial train data shape:\t", df_train.shape)
    
    del data_store
    df_train = df_train.pipe(Pipeline.filter_cols)
    gc.collect()

    # 数据分析
    print("Starting exploratory data analysis...")
    try:
        eda = ExploratoryAnalysis(df_train)
        plt.figure(figsize=(10, 6))
        eda.plot_target_distribution()
        plt.savefig('target_distribution.png')

        eda.plot_numeric_distributions()
        plt.savefig('numeric_distributions.png')

        print("\nGenerating additional analysis...")
        eda.plot_correlations()
        plt.savefig('feature_correlations.png')

        eda.plot_missing_values()
        plt.savefig('missing_values.png')
        plt.close('all')

        print("Exploratory analysis completed and saved.")
    except Exception as e:
        print(f"Error in exploratory analysis: {str(e)}")

    # 特征编码
    print("Starting feature encoding...")
    cnt_encoding_cols = df_train.select(pl.selectors.by_dtype([pl.String, pl.Boolean, pl.Categorical])).columns
    
    mappings = {}
    for col in cnt_encoding_cols:
        mappings[col] = df_train.group_by(col).len()

    df_train_lazy = df_train.select(mappings.keys()).lazy()
    
    for col, mapping in mappings.items():
        remapping = {category: count for category, count in mapping.rows()}
        remapping[None] = -2
        expr = pl.col(col).replace(remapping, default=-1)
        df_train_lazy = df_train_lazy.with_columns(expr.alias(col + '_cnt'))
        del col, mapping, remapping
        gc.collect()

    del mappings
    transformed_train = df_train_lazy.collect()
    df_train = pl.concat([df_train, transformed_train.select("^*cnt$")], how='horizontal')
    del transformed_train, cnt_encoding_cols
    gc.collect()

    # 转换为pandas并优化内存
    df_train, cat_cols = to_pandas(df_train)
    df_train = reduce_mem_usage(df_train)
    print("Processed train data shape:\t", df_train.shape)

    # 准备训练数据
    y = df_train["target"]
    weeks = df_train["WEEK_NUM"]
    df_train = df_train.drop(columns=["target", "case_id", "WEEK_NUM"])

    # 设置交叉验证
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=False)

    # 开始训练
    print("Starting quick model training...")
    fitted_models = []
    cv_scores = []
    all_feature_importance = []  # 添加一个列表存储所有fold的特征重要性

    for fold, (idx_train, idx_valid) in enumerate(cv.split(df_train, y, groups=weeks), 1):
        print(f"\nFold {fold}/{n_splits}")
        X_train, y_train = df_train.iloc[idx_train], y.iloc[idx_train]
        X_valid, y_valid = df_train.iloc[idx_valid], y.iloc[idx_valid]

        model = lgb.LGBMClassifier(**quick_params)
        model.fit(X_train, y_train,
                 eval_set=[(X_valid, y_valid)],
                 callbacks=[lgb.log_evaluation(5)])
        fitted_models.append(model)

        importance = pd.DataFrame({
            'feature': df_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        all_feature_importance.append(importance)

        y_pred = model.predict_proba(X_valid)[:,1]
        auc_score = roc_auc_score(y_valid, y_pred)
        cv_scores.append(auc_score)
        
        print(f"Fold {fold} AUC: {auc_score:.4f}")
        
        # 保存特征重要性图
        evaluator = ModelEvaluator(model, X_train, X_valid, y_train, y_valid)
        evaluator.plot_feature_importance(top_n=10)  # 只展示前10个特征
        plt.savefig(f'fold_{fold}_feature_importance.png')
        plt.close()

    print("\nQuick training completed!")
    print("CV scores:", cv_scores)
    print("Mean CV score:", np.mean(cv_scores))

    for fold, model in enumerate(fitted_models, 1):
        print(f"\nFeature Importance Analysis for Fold {fold}:")
        analyze_feature_importance(model, df_train.columns)

    # 计算平均特征重要性
    mean_importance = pd.concat(all_feature_importance).groupby('feature')['importance'].mean()
    mean_importance = mean_importance.sort_values(ascending=False)

    # 创建结果摘要字典
    results_summary = {
        'data_shape': {
            'initial': df_train.shape,
            'processed': df_train.shape
        },
        'cv_scores': cv_scores,
        'mean_auc': np.mean(cv_scores),
        'std_auc': np.std(cv_scores),
        'feature_importance': importance.to_dict(),
        'missing_values': df_train.isnull().sum().to_dict(),
        'correlations': df_train.corr()['target'].to_dict()
    }

    # 保存为JSON格式
    import json
    with open('analysis_results.json', 'w') as f:
        json.dump(results_summary, f, indent=4)

    # 打印重要特征
    print("\nTop 10 Important Features:")
    print(mean_importance.head(10))

    return results_summary


if __name__ == "__main__":
    main(sample_size=10000)  # 使用10000条数据快速预览

# if __name__ == "__main__":
#     main()