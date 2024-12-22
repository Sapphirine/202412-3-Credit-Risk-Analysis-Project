# %%
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
import joblib

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')

ROOT = '/Users/wuqianran/Desktop/bigdata_finalproject/final'

from sklearn.model_selection import TimeSeriesSplit, GroupKFold, StratifiedGroupKFold
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer

# %%
class Pipeline:

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

    def handle_dates(df):
        for col in df.columns:
            if col[-1] in ("D",):
                df = df.with_columns(pl.col(col) - pl.col("date_decision"))  #!!?
                df = df.with_columns(pl.col(col).dt.total_days()) # t - t-1
        df = df.drop("date_decision", "MONTH")
        return df

    def filter_cols(df):
        for col in df.columns:
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



class Aggregator:
    # Please add or subtract features yourself, be aware that too many features will take up too much space.
    def num_expr(df):
        cols = [col for col in df.columns if col[-1] in ("P", "A")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]

        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        # expr_first = [pl.first(col).alias(f"first_{col}") for col in cols]
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in cols]
        expr_median = [pl.median(col).alias(f"median_{col}") for col in cols]
        expr_var = [pl.var(col).alias(f"var_{col}") for col in cols]

        return expr_max  + expr_mean 

    def date_expr(df):
        cols = [col for col in df.columns if col[-1] in ("D")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        # expr_min = [pl.min(col).alias(f"min_{col}") for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        # expr_first = [pl.first(col).alias(f"first_{col}") for col in cols]
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in cols]
        expr_median = [pl.median(col).alias(f"median_{col}") for col in cols]

        return expr_max  + expr_mean 

    def str_expr(df):
        cols = [col for col in df.columns if col[-1] in ("M",)]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        # expr_min = [pl.min(col).alias(f"min_{col}") for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        # expr_first = [pl.first(col).alias(f"first_{col}") for col in cols]
        # expr_count = [pl.count(col).alias(f"count_{col}") for col in cols]
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in cols]
        return expr_max  + expr_mean
    
    def other_expr(df):
        cols = [col for col in df.columns if col[-1] in ("T", "L")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        # expr_min = [pl.min(col).alias(f"min_{col}") for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        # expr_first = [pl.first(col).alias(f"first_{col}") for col in cols]
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in cols]
        return expr_max  + expr_mean

    def count_expr(df):
        cols = [col for col in df.columns if "num_group" in col]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        # expr_min = [pl.min(col).alias(f"min_{col}") for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        # expr_first = [pl.first(col).alias(f"first_{col}") for col in cols]
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in cols]
        return expr_max + expr_mean

    def get_exprs(df):
        exprs = Aggregator.num_expr(df) + \
                Aggregator.date_expr(df) + \
                Aggregator.str_expr(df) + \
                Aggregator.other_expr(df) + \
                Aggregator.count_expr(df)

        return exprs


# %%
def read_file(path, depth=None):
    df = pl.read_parquet(path)
    df = df.pipe(Pipeline.set_table_dtypes)
    if depth in [1,2]:
        df = df.group_by("case_id").agg(Aggregator.get_exprs(df)) 
    return df


def read_files(regex_path, depth=None):
    chunks = []
    
    for path in glob(str(regex_path)):
        df = pl.read_parquet(path)
        df = df.pipe(Pipeline.set_table_dtypes)
        if depth in [1, 2]:
            df = df.group_by("case_id").agg(Aggregator.get_exprs(df))
        chunks.append(df)
    
    df = pl.concat(chunks, how="vertical_relaxed")
    df = df.unique(subset=["case_id"])
    return df

# %%
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

# %%
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
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
        else:
            continue
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

# %%
%%time

ROOT            = Path(ROOT)

TRAIN_DIR       = ROOT / "parquet_files" / "train"
TEST_DIR        = ROOT / "parquet_files" / "test"

data_store = {
    "df_base": read_file(TRAIN_DIR / "train_base.parquet"),
    "depth_0": [
        read_file(TRAIN_DIR / "train_static_cb_0.parquet"),
        read_files(TRAIN_DIR / "train_static_0_*.parquet"),
    ],
    "depth_1": [
        read_files(TRAIN_DIR / "train_applprev_1_*.parquet", 1),
        read_file(TRAIN_DIR / "train_tax_registry_a_1.parquet", 1),
        read_file(TRAIN_DIR / "train_tax_registry_b_1.parquet", 1),
        read_file(TRAIN_DIR / "train_tax_registry_c_1.parquet", 1),
        read_files(TRAIN_DIR / "train_credit_bureau_a_1_*.parquet", 1),
        read_file(TRAIN_DIR / "train_credit_bureau_b_1.parquet", 1),
        read_file(TRAIN_DIR / "train_other_1.parquet", 1),
        read_file(TRAIN_DIR / "train_person_1.parquet", 1),
        read_file(TRAIN_DIR / "train_deposit_1.parquet", 1),
        read_file(TRAIN_DIR / "train_debitcard_1.parquet", 1),
    ],
    "depth_2": [
        read_file(TRAIN_DIR / "train_credit_bureau_b_2.parquet", 2),
        read_files(TRAIN_DIR / "train_credit_bureau_a_2_*.parquet", 2),
    ]
}

# %%
%%time

df_train = feature_eng(**data_store)
print("train data shape:\t", df_train.shape)
del data_store
df_train = df_train.pipe(Pipeline.filter_cols)
gc.collect()

# %%
!pip install --upgrade polars

# %%
cnt_encoding_cols = df_train.select(pl.selectors.by_dtype([pl.String, pl.Boolean, pl.Categorical])).columns

mappings = {}
for col in cnt_encoding_cols:
    mappings[col] = df_train.group_by(col).len()

df_train_lazy = df_train.select(mappings.keys()).lazy()
# df_train_lazy = pl.LazyFrame(df_train.select('case_id'))

for col, mapping in mappings.items():
    remapping = {category: count for category, count in mapping.rows()}
    remapping[None] = -2
    expr = pl.col(col).replace(
                remapping,
                default=-1,
            )
    df_train_lazy = df_train_lazy.with_columns(expr.alias(col + '_cnt'))
    del col, mapping, remapping
    gc.collect()

del mappings
transformed_train = df_train_lazy.collect()

df_train = pl.concat([df_train, transformed_train.select("^*cnt$")], how='horizontal')
del transformed_train, cnt_encoding_cols

gc.collect()

# %%
df_train, cat_cols = to_pandas(df_train)
df_train = reduce_mem_usage(df_train)
print("train data shape:\t", df_train.shape)
nums=df_train.select_dtypes(exclude='category').columns
from itertools import combinations, permutations
#df_train=df_train[nums]
nans_df = df_train[nums].isna()
nans_groups={}
for col in nums:
    cur_group = nans_df[col].sum()
    try:
        nans_groups[cur_group].append(col)
    except:
        nans_groups[cur_group]=[col]
del nans_df; x=gc.collect()

def reduce_group(grps):
    use = []
    for g in grps:
        mx = 0; vx = g[0]
        for gg in g:
            n = df_train[gg].nunique()
            if n>mx:
                mx = n
                vx = gg
            #print(str(gg)+'-'+str(n),', ',end='')
        use.append(vx)
        #print()
    print('Use these',use)
    return use

def group_columns_by_correlation(matrix, threshold=0.8):
    # 计算列之间的相关性
    correlation_matrix = matrix.corr()

    # 分组列
    groups = []
    remaining_cols = list(matrix.columns)
    while remaining_cols:
        col = remaining_cols.pop(0)
        group = [col]
        correlated_cols = [col]
        for c in remaining_cols:
            if correlation_matrix.loc[col, c] >= threshold:
                group.append(c)
                correlated_cols.append(c)
        groups.append(group)
        remaining_cols = [c for c in remaining_cols if c not in correlated_cols]
    
    return groups

uses=[]
for k,v in nans_groups.items():
    if len(v)>1:
            Vs = nans_groups[k]
            #cross_features=list(combinations(Vs, 2))
            #make_corr(Vs)
            grps= group_columns_by_correlation(df_train[Vs], threshold=0.8)
            use=reduce_group(grps)
            uses=uses+use
            #make_corr(use)
    else:
        uses=uses+v
    print('####### NAN count =',k)
print(uses)
print(len(uses))
uses=uses+list(df_train.select_dtypes(include='category').columns)
print(len(uses))
df_train=df_train[uses]
# df_train.drop(['requesttype_4525192L_cnt','max_empl_employedtotal_800L_cnt', 'max_empl_industry_691L_cnt'], axis=1, inplace=True)

# %%
y = df_train["target"]
weeks = df_train["WEEK_NUM"]
df_train= df_train.drop(columns=["target", "case_id", "WEEK_NUM"])
n_splits=5
cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=False)

# %%
# 在训练模型后，保存分类特征信息
categorical_features = []
for col in df_train.columns:
    if pd.api.types.is_categorical_dtype(df_train[col]) or df_train[col].dtype == 'object':
        categorical_features.append(col)

# 保存分类特征列表
joblib.dump(categorical_features, '/Users/wuqianran/Desktop/bigdata_finalproject/final/categorical_features.pkl')

# 保存数据类型信息
joblib.dump(df_train.dtypes, '/Users/wuqianran/Desktop/bigdata_finalproject/final/column_dtypes.pkl')

# %%
# params = {
#     "boosting_type": "gbdt",
    
#     "objective": "binary",
#     "metric": "auc",
#     "max_depth": 8,  
#     "learning_rate": 0.01,
#     "n_estimators": 10000,  
#     "colsample_bytree": 0.8,
#     "colsample_bynode": 0.8,
#     "verbose": -1,
#     "random_state": 42,
#     "reg_alpha": 0.3,
#     "reg_lambda": 8,
#     "extra_trees":True,
#     'num_leaves':32,
#     "sample_weight":'balanced',
#     # "device": "cpu", 
#     "device": "gpu", 
#     "verbose": -1,
# }
# 保存数据类型信息
df_train.dtypes.to_frame('dtype').to_csv('/Users/wuqianran/Desktop/bigdata_finalproject/final/column_dtypes.csv')

# 保存处理后的数据
df_train.to_csv('/Users/wuqianran/Desktop/bigdata_finalproject/final/processed_data.csv', index=False)

# 计算并保存每列的平均值
column_means = df_train.mean().to_frame('mean')
column_means.to_csv('/Users/wuqianran/Desktop/bigdata_finalproject/final/column_means.csv')

params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "max_depth": 4,  # 降低树的最大深度
    "learning_rate": 0.05,  # 降低学习率
    "n_estimators": 100,  # 增加迭代次数
    "colsample_bytree": 0.6,
    "colsample_bynode": 0.6,
    "verbose": -1,
    "random_state": 42,
    "reg_alpha": 0.1,  # 增加 L1 正则化
    "reg_lambda": 1,  # 增加 L2 正则化
    "extra_trees": True,
    'num_leaves': 8,  # 减少叶子节点的数量
    "min_data_in_leaf": 50,  # 增加每个叶子节点的最小数据量
    "device": "cpu",
    # "device": "gpu",
    "verbose": -1,
}
fitted_models = []
cv_scores = []
best_auc = 0
best_model = None

for idx_train, idx_valid in cv.split(df_train, y, groups=weeks):#   Because it takes a long time to divide the data set, 
    X_train, y_train = df_train.iloc[idx_train], y.iloc[idx_train]# each time the data set is divided, two models are trained to each other twice, which saves time.
    X_valid, y_valid = df_train.iloc[idx_valid], y.iloc[idx_valid]
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set = [(X_valid, y_valid)],
        callbacks = [lgb.log_evaluation(200), lgb.early_stopping(100)] )
    fitted_models.append(model)
    y_pred_valid = model.predict_proba(X_valid)[:,1]
    auc_score = roc_auc_score(y_valid, y_pred_valid)
    cv_scores.append(auc_score)
    
    # 如果当前模型的 AUC 分数是最好的，则保存该模型
    if auc_score > best_auc:
        best_auc = auc_score
        best_model = model
        
if best_model is not None:
    joblib.dump(best_model, '/Users/wuqianran/Desktop/bigdata_finalproject/final/lgbm_best_model.pkl')

lgb_cv_results = pd.DataFrame({
    'fold': range(1, n_splits + 1),
    'auc_score': cv_scores
})
lgb_cv_results.to_csv('/Users/wuqianran/Desktop/bigdata_finalproject/final/lgbm_results.csv', index=False)


print("CV AUC scores: ", cv_scores)
print("Maximum CV AUC score: ", max(cv_scores))

# %%
# %%
# 在训练模型后，保存分类特征信息
categorical_features = []
for col in df_train.columns:
    if pd.api.types.is_categorical_dtype(df_train[col]) or df_train[col].dtype == 'object':
        categorical_features.append(col)

# 保存分类特征列表
joblib.dump(categorical_features, '/Users/wuqianran/Desktop/bigdata_finalproject/final/categorical_features.pkl')

# 保存数据类型信息
joblib.dump(df_train.dtypes, '/Users/wuqianran/Desktop/bigdata_finalproject/final/column_dtypes.pkl')


# %%
y = df_train["target"]
weeks = df_train["WEEK_NUM"]
df_train= df_train.drop(columns=["target", "case_id", "WEEK_NUM"])
df_train[cat_cols] = df_train[cat_cols].astype(str)

# %%
from catboost import CatBoostClassifier, Pool

# params = {
#     "eval_metric": "AUC",  
#     # "depth": 10,  
#     "learning_rate": 0.03,
#     "iterations": 6000,  # 4000
#     # "random_seed": 3107,  
#     # "l2_leaf_reg": 10,  
#     # "border_count": 254,  
#     "verbose": 500,  
#     "task_type": "GPU",
#     "early_stopping_rounds": 100  # 设置早停机制
# }

params = {
    "eval_metric": "AUC",  
    "depth": 6,  # 降低树的最大深度
    "learning_rate": 0.01,  # 降低学习率
    "iterations": 3000,  # 减少迭代次数
    "l2_leaf_reg": 5,  # 增加 L2 正则化
    "verbose": 500,  
    "task_type": "CPU",
    "early_stopping_rounds": 100  # 设置早停机制
}

fitted_models = []
cv_scores = []
best_auc = 0
best_model = None

cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=False)

step = 0
for idx_train, idx_valid in cv.split(df_train, y, groups=weeks):#   Because it takes a long time to divide the data set, 
    step += 1
    print(f'current step: {step}')
    
    X_train, y_train = df_train.iloc[idx_train], y.iloc[idx_train]# each time the data set is divided, two models are trained to each other twice, which saves time.
    X_valid, y_valid = df_train.iloc[idx_valid], y.iloc[idx_valid]

    train_pool = Pool(X_train, y_train,cat_features=cat_cols)
    val_pool = Pool(X_valid, y_valid,cat_features=cat_cols)

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool, verbose=100, early_stopping_rounds=50)

    
    fitted_models.append(model)
    y_pred_valid = model.predict_proba(X_valid)[:,1]
    auc_score = roc_auc_score(y_valid, y_pred_valid)
    cv_scores.append(auc_score)
    
    # 如果当前模型的 AUC 分数是最好的，则保存该模型
    if auc_score > best_auc:
        best_auc = auc_score
        best_model = model

# 保存最好的模型到文件
if best_model is not None:
    joblib.dump(best_model, '/Users/wuqianran/Desktop/bigdata_finalproject/final/catboost_best_model.pkl')

# 保存训练结果到文件
cv_results = pd.DataFrame({
    'fold': range(1, n_splits + 1),
    'auc_score': cv_scores
})
cv_results.to_csv('/Users/wuqianran/Desktop/bigdata_finalproject/final/catboost_results.csv', index=False)

 
print("CV AUC scores: ", cv_scores)
print("AVG CV AUC score: ", np.mean(cv_scores))
print("Maximum CV AUC score: ", max(cv_scores))


