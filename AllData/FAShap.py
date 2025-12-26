import os
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
import shap
import warnings
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

# 配置设置
warnings.filterwarnings('ignore')

# 模型权重配置（基于测试集R²）
model_configs = {
    'rf': {'path': 'models/best_rf_model.pkl', 'r2': 0.6892},
    'xgb': {'path': 'models/best_xgb_model.pkl', 'r2': 0.7630},
    'catboost': {'path': 'models/best_catboost_model.pkl', 'r2': 0.6762},
    'lgbm': {'path': 'models/best_lgbm_model.pkl', 'r2': 0.7446}
}

# 加载数据
df = pd.read_excel("FinalDataAll.xlsx")
X = df.drop('PCE', axis=1)

# 保存原始列名
original_columns = X.columns.tolist()
X.columns = [col.replace(' ', '_') for col in X.columns]

# 加载所有模型
print("加载集成模型...")
models = {}
weights = {}
successful_models = 0

# 计算总R²用于权重归一化
total_r2 = sum(config['r2'] for config in model_configs.values())

for model_name, config in model_configs.items():
    try:
        if model_name == 'catboost':
            # 尝试用joblib加载模型
            try:
                model = joblib.load(config['path'])
                print(f"✅ CatBoost模型从 {config['path']} 加载成功")
            except:
                # 尝试用CatBoost自己的加载方法
                model = CatBoostRegressor()
                model.load_model(config['path'])
                print(f"✅ CatBoost模型从 {config['path']} 加载成功 (使用CatBoost原生格式)")

        elif model_name == 'xgb':
            # 加载XGBoost模型
            model = joblib.load(config['path'])
            print(f"✅ XGBoost模型加载成功")

        elif model_name == 'lgbm':
            # 加载LightGBM模型
            model = joblib.load(config['path'])
            print(f"✅ LightGBM模型加载成功")

        elif model_name == 'rf':
            # 加载RandomForest模型
            model = joblib.load(config['path'])
            print(f"✅ RandomForest模型加载成功")

        models[model_name] = model
        # 计算权重：该模型R²占总R²的比例
        weights[model_name] = config['r2'] / total_r2
        successful_models += 1
        print(f"  {model_name.upper()}权重: {weights[model_name]:.4f}")

    except Exception as e:
        print(f"❌ {model_name.upper()}模型加载失败: {e}")

# 检查是否有模型成功加载
if successful_models == 0:
    print("❌ 所有模型加载失败，无法进行分析")
    exit(1)

print(f"\n✅ 成功加载 {successful_models}/{len(model_configs)} 个模型")
print("模型权重汇总:")
for model_name, weight in weights.items():
    if model_name in models:
        print(f"  {model_name.upper()}: {weight:.4f}")

# 计算集成模型的加权SHAP值
print("\n计算集成模型的加权SHAP值...")
weighted_shap_values = None
total_weight = 0

for model_name, model in models.items():
    try:
        print(f"计算 {model_name.upper()} 的SHAP值...")

        # 准备特征数据
        X_features = X.copy()
        X_features.columns = original_columns

        if model_name == 'catboost':
            # 对于CatBoost，使用Pool
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(Pool(X_features))
        elif model_name == 'xgb':
            # 对于XGBoost
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_features)
        elif model_name == 'lgbm':
            # 对于LightGBM
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_features)
        elif model_name == 'rf':
            # 对于RandomForest
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_features)

        # 加权SHAP值
        model_weight = weights[model_name]
        if weighted_shap_values is None:
            weighted_shap_values = shap_values * model_weight
        else:
            weighted_shap_values += shap_values * model_weight

        total_weight += model_weight
        print(f"  {model_name.upper()} SHAP值计算完成，权重: {model_weight:.4f}")

    except Exception as e:
        print(f"  ❌ {model_name.upper()} SHAP值计算失败: {e}")

# 归一化加权SHAP值
if weighted_shap_values is not None:
    weighted_shap_values /= total_weight
    print("\n✅ 集成模型SHAP值计算完成")
else:
    print("❌ 所有模型的SHAP值计算都失败了")
    exit(1)

# 查找FA特征
print("\n查找FA特征...")
fa_columns = []
for col in original_columns:
    if 'FA' in col.upper():
        fa_columns.append(col)

if not fa_columns:
    print("未找到FA特征列，尝试查找其他可能的名称...")
    # 如果没有找到，尝试其他可能的FA相关列名
    for col in X.columns:
        if 'FA' in col.upper():
            # 找到对应的原始列名
            idx = list(X.columns).index(col)
            if idx < len(original_columns):
                fa_columns.append(original_columns[idx])

if not fa_columns:
    print("仍未找到FA特征列，请检查数据列名")
    exit()

print(f"找到的FA特征: {fa_columns}")

# 创建输出目录
os.makedirs("ensemble_results", exist_ok=True)

# 分析每个FA特征
for fa_original in fa_columns:
    print(f"\n{'=' * 50}")
    print(f"处理特征: {fa_original}")
    print(f"{'=' * 50}")

    # 获取特征索引
    fa_idx = original_columns.index(fa_original)

    # 获取特征对应的统一列名
    fa_col = fa_original.replace(' ', '_')

    # 提取集成SHAP值
    ensemble_shap_values = weighted_shap_values[:, fa_idx]
    fa_values = X[fa_col].values

    # 创建结果DataFrame
    result_df = pd.DataFrame({
        'FA_Value': fa_values,
        'Ensemble_SHAP_Value': ensemble_shap_values,
        'FA_Feature': fa_original
    })

    # 按FA值从小到大排序
    result_df = result_df.sort_values('FA_Value')

    # 计算统计信息
    unique_values = result_df['FA_Value'].nunique()
    shap_mean = result_df['Ensemble_SHAP_Value'].mean()
    shap_std = result_df['Ensemble_SHAP_Value'].std()

    # 保存到CSV
    csv_file = f"ensemble_results/Ensemble_FA_{fa_col}_SHAP_Values.csv"
    result_df.to_csv(csv_file, index=False)

    print(f"📊 统计信息:")
    print(f"  FA唯一值数量: {unique_values}")
    print(f"  平均SHAP值: {shap_mean:.6f}")
    print(f"  SHAP值标准差: {shap_std:.6f}")
    print(f"  SHAP值范围: [{result_df['Ensemble_SHAP_Value'].min():.6f}, {result_df['Ensemble_SHAP_Value'].max():.6f}]")

    print(f"\n💾 已保存到文件: {csv_file}")
    print(f"  数据行数: {len(result_df)}")

    print(f"\n📋 前10行数据:")
    print(result_df.head(10).to_string(index=False))

    print(f"\n📋 后10行数据:")
    print(result_df.tail(10).to_string(index=False))

# 生成汇总报告
print(f"\n{'=' * 60}")
print("ENSEMBLE SHAP分析报告")
print(f"{'=' * 60}")
print(f"✅ 分析的FA特征数量: {len(fa_columns)}")
print(f"✅ 使用的模型数量: {successful_models}")
print(f"✅ 模型权重: {weights}")

# 保存模型权重信息
weights_df = pd.DataFrame([
    {'Model': model_name.upper(),
     'R2_Score': model_configs[model_name]['r2'],
     'Weight': weight}
    for model_name, weight in weights.items()
])
weights_df.to_csv("ensemble_results/model_weights.csv", index=False)
print(f"💾 模型权重信息已保存到: ensemble_results/model_weights.csv")

print(f"\n🎯 集成模型SHAP分析完成！")
print(f"📁 结果保存在: ensemble_results/ 目录")
print(f"📊 每个FA特征生成一个CSV文件，包含FA值和集成SHAP值")