import os
import joblib
import numpy as np
import pandas as pd
# from catboost import CatBoostRegressor, Pool # 已注释掉 CatBoost 导入
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import shap
import matplotlib.pyplot as plt
import warnings
from matplotlib import rcParams

# 配置设置
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimSong']
plt.rcParams['axes.unicode_minus'] = False
rcParams.update({'font.size': 12})

# 创建输出目录
os.makedirs("picture", exist_ok=True)

# 加载数据
df = pd.read_excel("FinalData10132.xlsx")
y = df['PCE']
X = df.drop('PCE', axis=1)

# --- 已移除 CatBoost 原始列名保存 ---
# original_columns = X.columns.tolist()

# 统一列名格式并创建分组列（用于其他模型）
X.columns = [col.replace(' ', '_') for col in X.columns]
if 'Active_Area' not in X.columns:
    raise ValueError("数据中缺少Active_Area列")

# 修改分组标签为<100和≥100
X['area_group'] = pd.cut(X['Active_Area'],
                         bins=[0, 100, float('inf')],
                         labels=['<100', '≥100'])


# 定义模型列表 (已移除 CatBoost)，包含路径和测试集 R2
MODELS_INFO = {
    # "CatBoost": ("models/best_catboost_model_part.cbm", 0.8704), # 已注释掉 CatBoost
    "RandomForest": ("models/best_randomforest_model.pkl", 0.8047),
    "XGBoost": ("models/best_xgboost_model.pkl", 0.8783),
    "LightGBM": ("models/best_lgbm_model.pkl", 0.8688)
}

# 加载所有模型 (除了 CatBoost) 并存储 R2 值
print("\n加载模型...")
models = {}
r2_values = {} # 重新引入 R2 值字典
for name, (path, r2) in MODELS_INFO.items():
    # if name == "CatBoost": # 已移除 CatBoost 加载逻辑
    #     model = CatBoostRegressor()
    #     model.load_model(path)
    # else:
    #     model = joblib.load(path)
    # models[name] = model
    # r2_values[name] = r2
    # print(f"{name}模型加载成功!")

    # --- 新的加载逻辑 (仅加载非 CatBoost 模型) ---
    try:
        model = joblib.load(path)
        models[name] = model
        r2_values[name] = r2 # 存储 R2 值用于加权
        print(f"{name}模型加载成功! (R2: {r2})")
    except FileNotFoundError:
        print(f"警告: 找不到模型文件 {path}，将跳过 {name} 模型。")
    except Exception as e:
        print(f"加载 {name} 模型时出错 ({path}): {e}")

if not models:
     raise RuntimeError("没有成功加载任何模型，请检查模型文件路径。")




# --- 恢复并更新加权 SHAP 计算函数 ---
def calculate_weighted_shap_values(X_data):
    """计算每个模型的SHAP值，并根据R²值进行加权平均 (CatBoost 已排除)"""
    shap_values = {}
    total_r2 = sum(r2_values.values()) # 计算总 R2

    if total_r2 == 0:
        raise ValueError("所有模型的 R2 值总和为 0，无法进行加权。")

    for name, model in models.items():
        # if name == "CatBoost": # 已移除 CatBoost 处理逻辑
        #     # 对于 CatBoost，使用原始列名
        #     if use_original_columns:
        #         X_data_catboost = X_data.copy()
        #         X_data_catboost.columns = original_columns
        #         explainer = shap.TreeExplainer(model)
        #         shap_values[name] = explainer.shap_values(Pool(X_data_catboost))
        #     else:
        #         explainer = shap.TreeExplainer(model)
        #         shap_values[name] = explainer.shap_values(Pool(X_data))
        # else:
        #     explainer = shap.Explainer(model)
        #     shap_values[name] = explainer(X_data).values

        # --- 新的 SHAP 计算逻辑 (仅处理非 CatBoost 模型) ---
        try:
            print(f"Calculating SHAP for {name}...")
            explainer = shap.TreeExplainer(model) # 对于 sklearn API 模型，TreeExplainer 通常有效
            shap_values_obj = explainer(X_data)
            shap_vals = shap_values_obj.values

            # 处理可能的多维输出 (例如 XGBoost 多分类)
            if shap_vals.ndim == 3:
                if shap_vals.shape[2] == 1:
                    shap_vals = shap_vals[:, :, 0]
                else:
                    print(f"警告: {name} 的 SHAP 值是三维的 (shape: {shap_vals.shape})，假设是回归并取第一个维度。")
                    shap_vals = shap_vals[:, :, 0]

            shap_values[name] = shap_vals

            # --- 根据 R2 进行加权 ---
            weight = r2_values[name] / total_r2
            shap_values[name] *= weight
            print(f"  -> Applied weight {weight:.4f} based on R2={r2_values[name]:.4f}")

        except Exception as e: # 添加了缺失的 except 块
            print(f"计算 {name} 的 SHAP 值或应用权重时出错: {e}")
            # 如果出错，可以从 shap_values 字典中移除该模型的结果，或赋予 0 权重
            shap_values[name] = np.zeros((X_data.shape[0], X_data.shape[1]))
            # 或者简单地 pass，但最好处理错误
            # pass

    # --- 计算加权平均 SHAP 值 ---
    if not shap_values:
        raise ValueError("未能为任何模型计算加权 SHAP 值。")

    # 初始化加权平均值数组
    first_model_values = next(iter(shap_values.values()))
    weighted_shap_values = np.zeros_like(first_model_values)

    # 累加所有加权后的 SHAP 值
    for values in shap_values.values():
        weighted_shap_values += values

    return weighted_shap_values


# def plot_importance_bar(average_shap_values, X_data, title, filename, color): # 函数签名更新以反映变化
def plot_importance_bar(weighted_shap_values, X_data, title, filename, color):
    """绘制重要性柱状图（现在基于R2加权平均SHAP值）"""
    plt.figure(figsize=(12, 8))

    # 计算平均绝对SHAP值
    # mean_abs_shap = np.mean(np.abs(average_shap_values), axis=0) # 旧逻辑
    mean_abs_shap = np.mean(np.abs(weighted_shap_values), axis=0) # 使用加权值
    features = X_data.columns
    sorted_idx = np.argsort(mean_abs_shap)[-15:]  # 取top15

    # 绘制柱状图（使用指定颜色）
    plt.barh(range(len(sorted_idx)), mean_abs_shap[sorted_idx], color=color)
    plt.yticks(range(len(sorted_idx)), features[sorted_idx])
    # plt.xlabel('平均SHAP绝对值 (多模型平均)') # 旧标签
    plt.xlabel('平均SHAP绝对值 (多模型R2加权平均)') # 更新标签
    plt.title(title)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"成功保存: {filename}")


# 颜色配置
COLOR_CONFIG = {
    "all": '#046B38',  # 全部数据
    "<100": '#58B668',  # <100组
    "≥100": '#CCEA9C'  # ≥100组
}

# 1. 所有数据
print("\n分析所有数据...")
# weighted_shap_values_all = calculate_average_shap_values(X.drop('area_group', axis=1)) # 旧调用
# --- 新的调用方式 (基于 R2 加权, 不涉及 CatBoost) ---
try:
    weighted_shap_values_all = calculate_weighted_shap_values(X.drop('area_group', axis=1))

    plot_importance_bar(
        weighted_shap_values_all, # 使用新计算的加权值
        X.drop('area_group', axis=1),
        "所有样本 Top15 特征重要性 (多模型R2加权平均)", # 更新标题
        "picture/shap_top15_all_ensemble_weighted.png", # 更新文件名以反映加权
        COLOR_CONFIG["all"]
    )
except Exception as e:
    print(f"分析所有数据时出错: {e}")

# 2. 按面积分组分析（只生成<100和≥100两张图）
for area in ['<100', '≥100']:
    print(f"\n分析面积组 {area}...")
    group_mask = X['area_group'] == area
    group_X = X.loc[group_mask].drop('area_group', axis=1)

    # 处理文件名中的特殊符号
    filename_area = area.replace('<', 'lt').replace('≥', 'gte')

    # weighted_shap_values_group = calculate_average_shap_values(group_X) # 旧调用
    # --- 新的调用方式 (基于 R2 加权, 不涉及 CatBoost) ---
    try:
        weighted_shap_values_group = calculate_weighted_shap_values(group_X)

        plot_importance_bar(
            weighted_shap_values_group, # 使用新计算的加权值
            group_X,
            f"面积 {area} cm² Top15 特征重要性 (多模型R2加权平均)", # 更新标题
            f"picture/shap_top15_{filename_area}_ensemble_weighted.png", # 更新文件名
            COLOR_CONFIG[area]
        )
    except Exception as e:
        print(f"分析面积组 {area} 时出错: {e}")

print("\n=== 分析完成 ===")
print("生成的图片文件:")

print("- picture/shap_top15_all_ensemble_weighted.png") # 更新打印的文件名
print("- picture/shap_top15_lt100_ensemble_weighted.png")
print("- picture/shap_top15_gte100_ensemble_weighted.png")




