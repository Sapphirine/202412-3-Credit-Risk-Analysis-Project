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

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')

# plt.style.use('seaborn')
ROOT = Path('/Users/wuqianran/Desktop/bigdata_finalproject/final')

# 保留原有的Pipeline类
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
        important_features = [
            'credamount_770A', 'homephncnt_628L', 'mobilephncnt_593L', 
            'numactivecreds_622L', 'month_decision', 'weekday_decision',
            # 添加更多重要特征
            'mean_actualdpd_943P', 'mean_currdebt_94A', 
            'mean_maxdpdtolerance_577P', 'max_debtoutstand_525A',
            'max_debtoverdue_47A', 'mean_overdueamount_31A'
        ]
        
        for col in df.columns:
            if col in important_features:
                continue
                
            if col not in ["target", "case_id", "WEEK_NUM"]:
                isnull = df[col].is_null().mean()
                if isnull > 0.95:
                    df = df.drop(col)
        
        for col in df.columns:
            if (col not in ["target", "case_id", "WEEK_NUM"]) & (df[col].dtype == pl.String):
                freq = df[col].n_unique()
                if (freq == 1) | (freq > 200):
                    df = df.drop(col)
        
        return df

# 保留原有的Aggregator类
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

# 添加新的特征工程类
class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        
    def create_domain_features(self, df):
        """改进特征工程"""
        features = df.copy()
        
        try:
            # 基本信用相关特征
            if 'credamount_770A' in features.columns and 'numactivecreds_622L' in features.columns:
                features['CREDIT_PER_ACTIVE_LOAN'] = (
                    features['credamount_770A'] / (features['numactivecreds_622L'] + 1)
                ).replace([np.inf, -np.inf], np.nan)
            
            # 联系方式特征
            if all(col in features.columns for col in ['mobilephncnt_593L', 'homephncnt_628L']):
                features['TOTAL_CONTACTS'] = features['mobilephncnt_593L'] + features['homephncnt_628L']
                features['CONTACTS_RATIO'] = (
                    features['mobilephncnt_593L'] / (features['homephncnt_628L'] + 1)
                ).replace([np.inf, -np.inf], np.nan)
            
            # 逾期相关特征
            dpd_cols = [col for col in features.columns if 'dpd' in col.lower()]
            overdue_cols = [col for col in features.columns if 'overdue' in col.lower()]
            debt_cols = [col for col in features.columns if 'debt' in col.lower()]
            
            if dpd_cols:
                features['DPD_MEAN'] = features[dpd_cols].mean(axis=1)
                features['DPD_MAX'] = features[dpd_cols].max(axis=1)
            
            if overdue_cols:
                features['OVERDUE_MEAN'] = features[overdue_cols].mean(axis=1)
                features['OVERDUE_MAX'] = features[overdue_cols].max(axis=1)
            
            if debt_cols:
                features['TOTAL_DEBT'] = features[debt_cols].sum(axis=1)
                
            # 时间相关特征
            if 'month_decision' in features.columns:
                features['IS_QUARTER_END'] = features['month_decision'].isin([3, 6, 9, 12]).astype(int)
                features['IS_YEAR_END'] = (features['month_decision'] == 12).astype(int)
            
            if 'weekday_decision' in features.columns:
                features['IS_WEEKEND'] = (features['weekday_decision'] >= 5).astype(int)
            
            # 分组统计特征
            for col in features.columns:
                if col not in ['credamount_770A', 'target', 'case_id', 'WEEK_NUM']:
                    # 计算每个特征值的历史违约率
                    temp_dict = (
                        features.groupby(col)['target']
                        .agg(['mean', 'count'])
                        .reset_index()
                    )
                    temp_dict = temp_dict[temp_dict['count'] >= 10]  # 只保留出现次数>=10的值
                    features[f'{col}_TARGET_MEAN'] = (
                        features[col].map(temp_dict.set_index(col)['mean'])
                    )
                    
            print("\nCreated features:", [col for col in features.columns if col not in df.columns])
            
        except Exception as e:
            print(f"Error in feature creation: {str(e)}")
            return features
        
        return features

    def handle_missing_values(self, df):
        """处理缺失值"""
        df_filled = df.copy()
        
        # 数值型特征用中位数填充
        numeric_features = df_filled.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_features:
            df_filled[col] = df_filled[col].fillna(df_filled[col].median())
            
        # 类别型特征用众数填充
        categorical_features = df_filled.select_dtypes(include=['object', 'category']).columns
        for col in categorical_features:
            df_filled[col] = df_filled[col].fillna(df_filled[col].mode().iloc[0])
            
        return df_filled

# 保留原有的数据读取和处理函数
def read_file(path, depth=None, sample_size=None):
    df = pl.read_parquet(path)
    if sample_size:
        df = df.sample(n=min(sample_size, df.height))
    df = df.pipe(Pipeline.set_table_dtypes)
    if depth in [1,2]:
        df = df.group_by("case_id").agg(Aggregator.get_exprs(df))
    return df

def read_files(regex_path, depth=None, sample_size=None):
    chunks = []
    for path in glob(str(regex_path)):
        df = pl.read_parquet(path)
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
    """保留原有的特征工程函数"""
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
    """转换为pandas dataframe"""
    df_data = df_data.to_pandas()
    if cat_cols is None:
        cat_cols = list(df_data.select_dtypes("object").columns)
    df_data[cat_cols] = df_data[cat_cols].astype("category")
    return df_data, cat_cols

def reduce_mem_usage(df):
    """降低内存使用"""
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

def train_model(df_train, y, weeks, params=None):
    """优化后的模型训练函数"""
    if params is None:
        params = {
            "boosting_type": "gbdt",
            "objective": "binary",
            "metric": "auc",
            "max_depth": 6,
            "num_leaves": 40,
            "learning_rate": 0.01,
            "n_estimators": 300,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_weight": 5,
            "min_child_samples": 20,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
            "class_weight": "balanced"
        }

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    scores = []
    feature_importance = pd.DataFrame()
    oof_predictions = np.zeros(len(df_train))

    for fold, (train_idx, val_idx) in enumerate(cv.split(df_train, y, groups=weeks), 1):
        X_train, X_val = df_train.iloc[train_idx], df_train.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 创建验证集
        train_data = lgb.Dataset(X_train, y_train)
        valid_data = lgb.Dataset(X_val, y_val, reference=train_data)
        
        # 训练模型
        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(200)
            ]
        )
        
        # 预测和评估
        y_pred = model.predict(X_val)
        auc_score = roc_auc_score(y_val, y_pred)
        scores.append(auc_score)
        
        # 保存OOF预测
        oof_predictions[val_idx] = y_pred
        
        # 记录特征重要性
        fold_importance = pd.DataFrame({
            'feature': df_train.columns,
            'importance': model.feature_importance(),
            'fold': fold
        })
        feature_importance = pd.concat([feature_importance, fold_importance])
        
        models.append(model)
        print(f"Fold {fold} AUC: {auc_score:.4f}")
    
    print(f"\nMean AUC: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    print(f"Full OOF AUC: {roc_auc_score(y, oof_predictions):.4f}")
    
    return models, scores, feature_importance

def select_features(df, feature_importance, min_score=0.001):
    """优化的特征选择"""
    mean_importance = feature_importance.groupby('feature')['importance'].mean()
    keep_features = ['target', 'case_id', 'WEEK_NUM']
    keep_features.extend(mean_importance[mean_importance > min_score].index)
    
    print("\nSelected features:", keep_features)
    return df[keep_features]

def main(sample_size=None):
    """主函数"""
    try:
        # 设置路径
        # ROOT = Path(ROOT)
        TRAIN_DIR = ROOT / "parquet_files" / "train"
        
        # 加载数据
        print("Loading data...")
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
        
        # 特征工程
        print("\nPerforming feature engineering...")
        df_train = feature_eng(**data_store)
        df_train = df_train.pipe(Pipeline.filter_cols)
        
        # 转换为pandas并创建新特征
        df_train, cat_cols = to_pandas(df_train)
        
        # 打印数据信息用于调试
        print("\nShape after initial processing:", df_train.shape)
        print("\nColumns after initial processing:", df_train.columns.tolist())
        
        fe = FeatureEngineer()
        df_train = fe.create_domain_features(df_train)
        df_train = fe.handle_missing_values(df_train)
        df_train = reduce_mem_usage(df_train)
        
        # 准备训练数据
        y = df_train['target']
        weeks = df_train['WEEK_NUM']
        X = df_train.drop(['target', 'case_id', 'WEEK_NUM'], axis=1)
        
        # 训练初始模型获取特征重要性
        init_models, _, init_importance = train_model(X, y, weeks)
        
        # 特征选择
        X_selected = select_features(X, init_importance)
        print(f"\nSelected {X_selected.shape[1]} features from {X.shape[1]} original features")
        
        # 使用选定特征重新训练
        final_models, scores, feature_importance = train_model(X_selected, y, weeks)
    
        
        # 训练模型
        print("\nTraining model...")
        models, scores, feature_importance = train_model(X, y, weeks)
        
        # 保存结果
        results = {
            'models': models,
            'scores': scores,
            'feature_importance': feature_importance,
        }
        
        joblib.dump(results, ROOT / 'model_results.pkl')
        
        # 输出特征重要性
        print("\nTop 10 most important features:")
        mean_importance = feature_importance.groupby('feature')['importance'].mean()
        print(mean_importance.sort_values(ascending=False).head(10))
        
        return results
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    results = main(sample_size=10000)  # 使用小样本快速测试
    # results = main()  # 使用全量数据