import os
import joblib
import numpy as np
import pandas as pd
# 注意：已移除 CatBoost 导入
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import shap
import matplotlib.pyplot as plt
import warnings
from matplotlib import rcParams

# === 配置设置 ===
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimSong']
plt.rcParams['axes.unicode_minus'] = False
rcParams.update({'font.size': 12})

# 创建输出目录
os.makedirs("picture", exist_ok=True)

# === 加载数据 ===
df = pd.read_excel("FinalData09112Part.xlsx")
y = df['PCE']
X = df.drop('PCE', axis=1)

# 统一列名格式（用于其他模型）
X.columns = [col.replace(' ', '_') for col in X.columns]

# === 定义模型路径 ===
MODEL_PATHS = {
    "RandomForest": "models/best_randomforest_model_part.pkl",
    "XGBoost": "models/best_xgboost_model_part.pkl",
    "LightGBM": "models/best_lgbm_model_part.pkl"
}

# === 加载所有模型 ===
print("\n加载模型...")
models = {}
for name, path in MODEL_PATHS.items():
    # 直接加载 sklearn API 兼容的模型 (RF, XGB, LGBM)
    model = joblib.load(path)
    models[name] = model
    print(f"{name}模型加载成功!")


# === 计算各模型SHAP值并求平均 ===
def calculate_mean_shap_values(X_data):
    """计算每个模型的SHAP值，并进行简单平均"""
    shap_values_list = []

    for name, model in models.items():
        # 对于 Sklearn API 兼容的模型 (RF, XGB, LGBM)
        explainer = shap.TreeExplainer(model)
        # shap_values 返回 numpy array (对于 TreeExplainer 和 sklearn 模型)
        shap_vals = explainer.shap_values(X_data)
        shap_values_list.append(shap_vals)

    # 计算平均SHAP值
    # 假设所有 shap_vals 形状一致 (n_samples, n_features)
    mean_shap_values = np.mean(shap_values_list, axis=0)
    return mean_shap_values


# === 绘制重要性柱状图 ===
def plot_importance_bar(mean_shap_values, X_data, title, filename, color):
    """绘制重要性柱状图（带自定义颜色）"""
    plt.figure(figsize=(12, 8))

    # 计算平均绝对SHAP值
    mean_abs_shap = np.mean(np.abs(mean_shap_values), axis=0)
    features = X_data.columns
    sorted_idx = np.argsort(mean_abs_shap)[-15:]  # 取top15

    # 绘制柱状图（使用指定颜色）
    plt.barh(range(len(sorted_idx)), mean_abs_shap[sorted_idx], color=color)
    plt.yticks(range(len(sorted_idx)), features[sorted_idx])
    plt.xlabel('平均SHAP绝对值 (三模型平均)')
    plt.title(title)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"成功保存: {filename}")

    return mean_abs_shap, features


# === 颜色配置 ===
COLOR_CONFIG = {
    "all": '#1f77b4'  # 默认蓝色
}

# === 分析所有数据 ===
print("\n分析所有数据...")
mean_shap_values_all = calculate_mean_shap_values(X)

# 绘制重要性图并获取重要性值
mean_abs_shap, features = plot_importance_bar(
    mean_shap_values_all,
    X,
    "所有样本 Top15 特征重要性 (三模型平均)",
    "picture/shap_top15_all_ensemble_mean_rf_xgb_lgbm_part.png",
    COLOR_CONFIG["all"]
)

# 创建特征重要性DataFrame并保存
importance_df = pd.DataFrame({
    'feature': features,
    'mean_abs_shap': mean_abs_shap
})

# 按重要性排序
importance_df = importance_df.sort_values('mean_abs_shap', ascending=False)

# 保存到CSV文件
importance_df.to_csv("picture/feature_importance_ranking_mean_rf_xgb_lgbm.csv", index=False, encoding='utf-8-sig')
print("特征重要性排名已保存到: picture/feature_importance_ranking_mean_rf_xgb_lgbm.csv")

# 打印前15个最重要的特征
print("\n=== 前15个最重要的特征 ===")
for i, (feature, importance) in enumerate(zip(importance_df['feature'][:15], importance_df['mean_abs_shap'][:15]), 1):
    print(f"{i}. {feature}: {importance:.6f}")

print("\n=== 分析完成 ===")
print("生成的图片文件: picture/shap_top15_all_ensemble_mean_rf_xgb_lgbm_part.png")
print("生成的数据文件: picture/feature_importance_ranking_mean_rf_xgb_lgbm.csv")