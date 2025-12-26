import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import warnings
from matplotlib import rcParams

# === 配置设置 ===
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimSong']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号
rcParams.update({'font.size': 12})

# 创建输出目录
os.makedirs("picture", exist_ok=True)

# === 加载数据 ===
df = pd.read_excel("FinalData10312All.xlsx")
y = df['PCE']
X = df.drop('PCE', axis=1)

# 统一列名格式（避免空格等问题）
X.columns = [col.replace(' ', '_') for col in X.columns]

# === 加载 XGBoost 模型 ===
print("\n加载 XGBoost 模型...")
try:
    xgb_model = joblib.load("models/best_xgb_model.pkl")  # 确保模型文件存在
    print("XGBoost 模型加载成功!")
except FileNotFoundError:
    raise FileNotFoundError("未找到模型文件 'models/best_xgboost_model.pkl'，请确认路径或先训练模型。")

# 确保模型是 XGBRegressor 类型
if not isinstance(xgb_model, xgb.XGBRegressor):
    raise TypeError("加载的模型不是 XGBRegressor 类型！")


# === SHAP 值计算函数 ===
def calculate_shap_values(X_data):
    """计算 XGBoost 模型的 SHAP 值"""
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_data)
    return shap_values


# === 特征重要性条形图绘制函数 ===
def plot_importance_bar(shap_values, X_data, title, filename, color):
    """绘制重要性柱状图（带自定义颜色）"""
    plt.figure(figsize=(12, 8))

    # 计算平均绝对SHAP值
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    features = X_data.columns
    sorted_idx = np.argsort(mean_abs_shap)[-15:]  # 取 top15

    # 绘制柱状图
    plt.barh(range(len(sorted_idx)), mean_abs_shap[sorted_idx], color=color)
    plt.yticks(range(len(sorted_idx)), features[sorted_idx])
    plt.xlabel('平均SHAP绝对值 (XGBoost)')
    plt.title(title)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"成功保存: {filename}")

    # 返回特征重要性数据
    importance_df = pd.DataFrame({
        'feature': features,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)

    return importance_df


# === 颜色配置 ===
COLOR_CONFIG = {
    "all": '#ff7f0e'  # 默认橙色
}

# === 分析所有数据 ===
print("\n分析所有数据...")
shap_values_all = calculate_shap_values(X)

importance_df = plot_importance_bar(
    shap_values_all,
    X,
    "所有样本 Top15 特征重要性 (XGBoost)",
    "picture/shap_top15_all_xgb.png",
    COLOR_CONFIG["all"]
)

# 保存特征重要性到 CSV
importance_df.to_csv("picture/feature_importance_xgb.csv", index=False)
print("特征重要性数据已保存到: picture/feature_importance_xgb.csv")

# 打印前10个最重要的特征
print("\n=== Top 10 最重要特征 ===")
for i, (feature, importance) in enumerate(zip(importance_df['feature'][:10], importance_df['mean_abs_shap'][:10]), 1):
    print(f"{i}. {feature}: {importance:.6f}")

print("\n=== 分析完成 ===")
print("生成的图片文件: picture/shap_top15_all_xgb.png")
print("生成的数据文件: picture/feature_importance_xgb.csv")



