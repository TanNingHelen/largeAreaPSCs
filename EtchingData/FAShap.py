import os
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
import shap
import warnings

# 配置设置
warnings.filterwarnings('ignore')

# 加载数据
df = pd.read_excel("FinalData.xlsx")
X = df.drop('PCE', axis=1)

# 保存原始列名
original_columns = X.columns.tolist()
X.columns = [col.replace(' ', '_') for col in X.columns]

# 加载模型
catboost_model = CatBoostRegressor()
catboost_model.load_model("models/best_catboost_model.cbm")

# 计算SHAP值
X_catboost = X.copy()
X_catboost.columns = original_columns
explainer = shap.TreeExplainer(catboost_model)
shap_values_all = explainer.shap_values(Pool(X_catboost))

# 查找FA特征
fa_columns = []
for col in original_columns:
    if 'FA' in col.upper():
        fa_columns.append(col)

if not fa_columns:
    print("未找到FA特征列")
    exit()

print(f"找到的FA特征: {fa_columns}")

# 分析每个FA特征
for fa_original in fa_columns:
    print(f"\n处理特征: {fa_original}")

    # 获取特征索引
    fa_idx = original_columns.index(fa_original)

    # 获取特征对应的统一列名
    fa_col = fa_original.replace(' ', '_')

    # 提取数据
    fa_shap_values = shap_values_all[:, fa_idx]
    fa_values = X[fa_col].values

    # 创建结果DataFrame
    result_df = pd.DataFrame({
        'FA_Value': fa_values,
        'SHAP_Value': fa_shap_values
    })

    # 按FA值从小到大排序
    result_df = result_df.sort_values('FA_Value')

    # 保存到CSV
    csv_file = f"FA_{fa_original}_SHAP_Values.csv"
    result_df.to_csv(csv_file, index=False)

    print(f"已生成: {csv_file}")
    print(f"数据行数: {len(result_df)}")
    print("前5行数据:")
    print(result_df.head().to_string(index=False))

print("\n所有FA特征的SHAP值分析完成!")