# 在训练模型后，保存分类特征信息
categorical_features = []
for col in df_train.columns:
    if pd.api.types.is_categorical_dtype(df_train[col]) or df_train[col].dtype == 'object':
        categorical_features.append(col)

# 保存分类特征列表
joblib.dump(categorical_features, '/Users/wuqianran/Desktop/bigdata_finalproject/final/categorical_features.pkl')

# 保存数据类型信息
joblib.dump(df_train.dtypes, '/Users/wuqianran/Desktop/bigdata_finalproject/final/column_dtypes.pkl')
