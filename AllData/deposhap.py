import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor  # 添加 LightGBM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ============== 配置部分 ==============
# 字体设置
rcParams['font.sans-serif'] = ['SimSong']
rcParams['axes.unicode_minus'] = False

# 读取数据和映射文件
df = pd.read_excel("FinalData10312All.xlsx")
mapping_df = pd.read_csv("label_mappings/full_mapping_summary.csv")

# 获取沉积方法的映射关系
deposition_mapping_df = mapping_df[mapping_df['Feature'] == 'Deposition_Method']
method_mapping = dict(zip(deposition_mapping_df['Original'], deposition_mapping_df['Encoded']))
# 创建反向映射字典（编码值到原始名称）
reverse_method_mapping = dict(zip(deposition_mapping_df['Encoded'], deposition_mapping_df['Original']))

# 定义要分析的三种沉积方法
target_methods = ['slot die-coating', 'spin-coating', 'blade-coating']
target_encoded = [method_mapping[method] for method in target_methods]

# 筛选Active_Area >= 10的数据
filtered_df = df[df['Active_Area'] >= 10].copy()

# ============== 计算各类沉积方式数量 ==============
print("计算Active_Area >= 10的各类沉积方式数量...")

# 统计各类沉积方式的数量
deposition_counts = filtered_df['Deposition_Method'].value_counts().reset_index()
deposition_counts.columns = ['Encoded_Method', 'Count']

# 将编码值映射回原始名称
deposition_counts['Method_Name'] = deposition_counts['Encoded_Method'].map(reverse_method_mapping)

# 打印统计结果
print("\n=== Active_Area >= 10的各类沉积方式数量 ===")
for _, row in deposition_counts.iterrows():
    print(f"{row['Method_Name']} (编码: {row['Encoded_Method']}): {row['Count']}条记录")

# 保存统计结果到CSV
deposition_counts.to_csv('deposition_methods_count_active_area_ge_10.csv', index=False)
print("\n沉积方式数量统计已保存到: deposition_methods_count_active_area_ge_10.csv")

# ============== 添加交互特征 ==============
print("\n创建沉积方法与有效面积的交互特征...")

# 为每种沉积方法创建交互特征
for method_name, encoded_value in zip(target_methods, target_encoded):
    # 创建交互特征: 沉积方法指示器 × 有效面积
    filtered_df[f'{method_name.replace("-", "_")}_active_area'] = (
            (filtered_df['Deposition_Method'] == encoded_value).astype(int) * filtered_df['Active_Area']
    )

print(f"已创建 {len(target_methods)} 个交互特征")

# 准备训练数据
X = filtered_df.drop(['PCE'], axis=1)
y = filtered_df['PCE']

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============== 重新训练模型 ==============
print("重新训练模型以包含交互特征...")

# 训练随机森林模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_r2 = r2_score(y_test, rf_model.predict(X_test))
print(f"随机森林模型训练完成，测试集R²: {rf_r2:.4f}")

# 训练XGBoost模型
xgb_model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)
xgb_r2 = r2_score(y_test, xgb_model.predict(X_test))
print(f"XGBoost模型训练完成，测试集R²: {xgb_r2:.4f}")

# 训练CatBoost模型
catboost_model = CatBoostRegressor(iterations=100, random_state=42, verbose=0)
catboost_model.fit(X_train, y_train)
catboost_r2 = r2_score(y_test, catboost_model.predict(X_test))
print(f"CatBoost模型训练完成，测试集R²: {catboost_r2:.4f}")

# 训练LightGBM模型
lgbm_model = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
lgbm_model.fit(X_train, y_train)
lgbm_r2 = r2_score(y_test, lgbm_model.predict(X_test))
print(f"LightGBM模型训练完成，测试集R²: {lgbm_r2:.4f}")

# 更新模型和权重
models = {
    'rf': rf_model,
    'xgb': xgb_model,
    'catboost': catboost_model,
    'lgbm': lgbm_model  # 添加 LightGBM
}

R2_SCORES = {
    'rf': rf_r2,
    'xgb': xgb_r2,
    'catboost': catboost_r2,
    'lgbm': lgbm_r2  # 添加 LightGBM
}

# 重新计算权重
total_r2 = sum(R2_SCORES.values())
for model in R2_SCORES:
    R2_SCORES[model] = R2_SCORES[model] / total_r2

print(f"模型权重: RF={R2_SCORES['rf']:.3f}, XGB={R2_SCORES['xgb']:.3f}, CatBoost={R2_SCORES['catboost']:.3f}, LGBM={R2_SCORES['lgbm']:.3f}")


# ============== SHAP计算 ==============
def calculate_weighted_shap(models, X, r2_scores):
    """计算加权平均SHAP值"""
    weighted_shap_values_list = []

    # 使用前50个样本作为背景数据
    background = X[:50] if len(X) > 50 else X

    for name, model in models.items():
        try:
            print(f"\nCalculating SHAP for {name.upper()}...")

            # 针对不同模型使用不同的解释器
            if name == 'catboost':
                explainer = shap.Explainer(model)
            else:
                explainer = shap.TreeExplainer(model, background, feature_perturbation='interventional')

            # 计算SHAP值
            shap_values = explainer(X).values
            weighted_shap_values = shap_values * r2_scores[name]
            weighted_shap_values_list.append(weighted_shap_values)

        except Exception as e:
            print(f"{name.upper()} SHAP计算失败: {str(e)}")
            continue

    if not weighted_shap_values_list:
        raise ValueError("所有模型SHAP计算失败")

    return np.sum(weighted_shap_values_list, axis=0)


# ============== 主程序 ==============
print("开始分析三种沉积方法的SHAP重要性...")

feature_names = X.columns.tolist()

# 为每种沉积方法计算SHAP值
shap_results = {}

for method_name, encoded_value in zip(target_methods, target_encoded):
    print(f"\n分析 {method_name} (编码值: {encoded_value})...")

    # 筛选当前方法的数据
    method_data = filtered_df[filtered_df['Deposition_Method'] == encoded_value].copy()

    if len(method_data) < 5:
        print(f"数据量不足({len(method_data)}条)，跳过 {method_name}")
        continue

    X_method = method_data.drop(['PCE'], axis=1)

    try:
        # 计算加权SHAP值
        shap_values = calculate_weighted_shap(models, X_method, R2_SCORES)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        shap_results[method_name] = {
            'shap_values': shap_values,
            'mean_abs_shap': mean_abs_shap,
            'data_size': len(method_data),
            'features': feature_names
        }

        print(f"{method_name}: 数据量={len(method_data)}, SHAP计算完成")

    except Exception as e:
        print(f"{method_name} SHAP计算错误: {e}")
        continue

# 创建SHAP结果表格
shap_table_data = []
for method_name, result in shap_results.items():
    for feature, importance in zip(result['features'], result['mean_abs_shap']):
        shap_table_data.append({
            'Deposition_Method': method_name,
            'Feature': feature,
            'SHAP_Importance': importance,
            'Data_Size': result['data_size']
        })

shap_df = pd.DataFrame(shap_table_data)
shap_df.to_csv('deposition_methods_weighted_shap_with_interaction.csv', index=False)
print("\n加权SHAP分析结果已保存到: deposition_methods_weighted_shap_with_interaction.csv")

# 特别关注交互特征的重要性
print("\n=== 交互特征重要性分析 ===")
interaction_features = [f'{method_name.replace("-", "_")}_active_area' for method_name in target_methods]

for method_name, result in shap_results.items():
    importance_df = pd.DataFrame({
        'Feature': result['features'],
        'Importance': result['mean_abs_shap']
    })

    # 获取交互特征的重要性
    interaction_imp = importance_df[importance_df['Feature'].isin(interaction_features)]

    print(f"\n{method_name} 的交互特征重要性:")
    for _, row in interaction_imp.iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.6f}")

    # 特别关注当前方法的交互特征
    current_method_feature = f'{method_name.replace("-", "_")}_active_area'
    current_importance = importance_df[importance_df['Feature'] == current_method_feature]['Importance'].values
    if len(current_importance) > 0:
        print(f"  当前方法交互特征 '{current_method_feature}' 的重要性: {current_importance[0]:.6f}")
        print(
            f"  在所有特征中排名: {importance_df['Importance'].rank(ascending=False).loc[importance_df['Feature'] == current_method_feature].values[0]:.0f}")

# 绘制每种沉积方法的SHAP重要性柱状图
if shap_results:
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle('加权SHAP特征重要性分析 - 不同沉积方法 (Active_Area ≥ 10)', fontsize=16)

    for i, (method_name, result) in enumerate(shap_results.items()):
        # 获取前10个最重要特征
        importance_df = pd.DataFrame({
            'Feature': result['features'],
            'Importance': result['mean_abs_shap']
        }).sort_values('Importance', ascending=False).head(10)

        # 绘制柱状图
        ax = axes[i]
        bars = ax.barh(range(len(importance_df)), importance_df['Importance'], color='lightblue')
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['Feature'])
        ax.set_xlabel('平均绝对SHAP值')
        ax.set_title(f'{method_name}\n(n={result["data_size"]})')
        ax.invert_yaxis()

        # 添加数值标签
        for j, v in enumerate(importance_df['Importance']):
            ax.text(v + 0.001, j, f'{v:.3f}', va='center')

    plt.tight_layout()
    plt.savefig('deposition_methods_weighted_shap_importance_with_interaction.png', dpi=300, bbox_inches='tight')
    print("SHAP重要性图已保存到: deposition_methods_weighted_shap_importance_with_interaction.png")

# 显示每种方法的最重要特征
print("\n=== 各方法最重要特征 ===")
for method_name, result in shap_results.items():
    importance_df = pd.DataFrame({
        'Feature': result['features'],
        'Importance': result['mean_abs_shap']
    }).sort_values('Importance', ascending=False).head(5)

    print(f"\n{method_name} (n={result['data_size']}):")
    for _, row in importance_df.iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")

print("\n分析完成！")