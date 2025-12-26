import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl

# ============== 配置部分 ==============
# 字体设置 - 改为 Times New Roman
rcParams['font.family'] = 'Times New Roman'
rcParams['axes.unicode_minus'] = False
rcParams.update({'font.size': 12})  # 增加全局字体大小

# 特征名称映射字典
FEATURE_ALIASES = {
    'FA': 'FA ratio',
    'MA': 'MA ratio',
    'Precursor_Solution_Addictive': 'Pre-Sol-Add',
    'Annealing_Temperature1': 'Annealing Tp1',
    'Precursor_Solution': 'Pre-Sol',
    'HTL_Passivator': 'HTL-Psvt',
    'Br': 'Br ratio',
    'Annealing_Time1': 'Annealing T1',
    'submodule_number': 'sub PSMs Num',
    'total_scribing_line_width(μm)': 'total Width',
    'Active_Area': 'Active-Area',
    'ETL_Passivator': 'ETL-Psvt',
    'Cs':'Cs ratio'

}

# 模型权重配置 (基于R²表现)
MODEL_WEIGHTS = {
    'lgbm': 0.7446,
    'rf': 0.6892,
    'catboost': 0.6762,
    'xgboost': 0.7630
}

# 归一化权重
total_weight = sum(MODEL_WEIGHTS.values())
for model in MODEL_WEIGHTS:
    MODEL_WEIGHTS[model] /= total_weight

print("模型权重配置:")
for model, weight in MODEL_WEIGHTS.items():
    print(f"  {model}: {weight:.4f}")

# ============== 读取数据和映射文件 ==============
df = pd.read_excel("FinalDataAll.xlsx")
mapping_df = pd.read_csv("label_mappings/full_mapping_summary.csv")

# 获取沉积方法的映射关系
deposition_mapping_df = mapping_df[mapping_df['Feature'] == 'Deposition_Method']
method_mapping = dict(zip(deposition_mapping_df['Original'], deposition_mapping_df['Encoded']))
# 创建反向映射字典（编码值到原始名称）
reverse_method_mapping = dict(zip(deposition_mapping_df['Encoded'], deposition_mapping_df['Original']))

# 定义要分析的三种沉积方法
target_methods = ['slot die-coating', 'spin-coating', 'blade-coating']
target_encoded = [method_mapping[method] for method in target_methods]

# 修改：筛选Active_Area < 10的数据
filtered_df = df[df['Active_Area'] < 10].copy()
print(f"筛选后数据量: {len(filtered_df)}条记录 (Active_Area < 10)")

# ============== 计算各类沉积方式数量 ==============
print("计算各类沉积方式数量...")

# 统计各类沉积方式的数量
deposition_counts = filtered_df['Deposition_Method'].value_counts().reset_index()
deposition_counts.columns = ['Encoded_Method', 'Count']

# 将编码值映射回原始名称
deposition_counts['Method_Name'] = deposition_counts['Encoded_Method'].map(reverse_method_mapping)

# 打印统计结果
print("\n=== 各类沉积方式数量 (Active_Area < 10) ===")
for _, row in deposition_counts.iterrows():
    print(f"{row['Method_Name']} (编码: {row['Encoded_Method']}): {row['Count']}条记录")

# 保存统计结果到CSV
deposition_counts.to_csv('deposition_methods_count_active_area_lt_10.csv', index=False)
print("\n沉积方式数量统计已保存到: deposition_methods_count_active_area_lt_10.csv")

# ============== 准备数据 ==============
print("\n准备数据...")

# 准备完整的特征和目标变量
X_full = filtered_df.drop(['PCE'], axis=1)
y_full = filtered_df['PCE']

# ============== 加载预训练模型 ==============
print("加载预训练的集成模型...")

# 定义模型路径
model_paths = {
    'xgboost': "models/best_xgb_model.pkl",
    'lgbm': "models/best_lgbm_model.pkl",
    'rf': "models/best_rf_model.pkl",
    'catboost': "models/best_catboost_model.pkl"  # 根据您提供的路径
}

# 加载所有模型
models = {}
for model_name, model_path in model_paths.items():
    if not os.path.exists(model_path):
        print(f"警告: 找不到模型文件: {model_path}")
        continue

    try:
        model = joblib.load(model_path)
        models[model_name] = model
        print(f"✅ {model_name} 模型加载成功: {model_path}")
    except Exception as e:
        print(f"❌ 加载 {model_name} 模型时出错: {e}")

if not models:
    raise FileNotFoundError("没有成功加载任何模型！")


# ============== SHAP计算 ==============
def calculate_shap_values(model, X, model_type):
    """计算单个模型的SHAP值"""
    try:
        print(f"Calculating SHAP for {model_type}...")

        # 根据模型类型选择合适的解释器
        if model_type in ['xgboost', 'lgbm', 'rf', 'catboost']:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model)

        # 计算SHAP值
        shap_values_obj = explainer(X)
        shap_values = shap_values_obj.values

        # 处理多维SHAP值
        if shap_values.ndim == 3:
            if shap_values.shape[2] == 1:
                shap_values = shap_values[:, :, 0]
            else:
                print(f"警告: SHAP值是三维的 (shape: {shap_values.shape}), 取第一个维度。")
                shap_values = shap_values[:, :, 0]

        return shap_values

    except Exception as e:
        print(f"{model_type} SHAP计算失败: {str(e)}")
        return None


def calculate_ensemble_shap(X_method, feature_names):
    """计算集成模型的加权SHAP值"""
    ensemble_shap_values = None
    total_weight = 0

    for model_name, model in models.items():
        if model_name not in MODEL_WEIGHTS:
            continue

        # 确保数据列顺序与模型期望一致
        try:
            if hasattr(model, 'get_booster'):  # XGBoost
                expected_features = model.get_booster().feature_names
            elif hasattr(model, 'feature_name_'):  # LightGBM
                expected_features = model.feature_name_
            else:
                expected_features = feature_names

            if expected_features:
                X_aligned = X_method.reindex(columns=expected_features)
            else:
                X_aligned = X_method.reindex(columns=feature_names)
        except:
            X_aligned = X_method.reindex(columns=feature_names)

        # 计算当前模型的SHAP值
        shap_values = calculate_shap_values(model, X_aligned, model_name)

        if shap_values is not None:
            weight = MODEL_WEIGHTS[model_name]

            if ensemble_shap_values is None:
                ensemble_shap_values = shap_values * weight
            else:
                ensemble_shap_values += shap_values * weight

            total_weight += weight

    if ensemble_shap_values is not None and total_weight > 0:
        # 归一化
        ensemble_shap_values /= total_weight
        return ensemble_shap_values
    else:
        return None


# ============== 主程序 ==============
print("开始分析三种沉积方法的SHAP重要性 (使用集成模型, Active_Area < 10)...")

# 获取特征名
feature_names = X_full.columns.tolist()
print(f"使用的特征数量: {len(feature_names)}")

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

    print(f"  用于SHAP计算的数据形状: {X_method.shape}")

    try:
        # 计算集成SHAP值
        ensemble_shap_values = calculate_ensemble_shap(X_method, feature_names)

        if ensemble_shap_values is not None:
            mean_abs_shap = np.abs(ensemble_shap_values).mean(axis=0)

            shap_results[method_name] = {
                'shap_values': ensemble_shap_values,
                'mean_abs_shap': mean_abs_shap,
                'data_size': len(method_data),
                'features': X_method.columns.tolist()
            }

            print(f"{method_name}: 数据量={len(method_data)}, 集成SHAP计算完成")
        else:
            print(f"{method_name}: SHAP计算失败")

    except Exception as e:
        print(f"{method_name} SHAP计算错误: {e}")
        continue

# 创建SHAP结果表格
shap_table_data = []
for method_name, result in shap_results.items():
    features = result['features']
    importances = result['mean_abs_shap']

    if len(features) != len(importances):
        min_len = min(len(features), len(importances))
        features = features[:min_len]
        importances = importances[:min_len]

    for feature, importance in zip(features, importances):
        shap_table_data.append({
            'Deposition_Method': method_name,
            'Feature': feature,
            'SHAP_Importance': importance,
            'Data_Size': result['data_size']
        })

if shap_table_data:
    shap_df = pd.DataFrame(shap_table_data)
    output_csv_file = 'deposition_methods_ensemble_shap_active_area_lt_10.csv'
    shap_df.to_csv(output_csv_file, index=False, encoding='utf-8-sig')
    print(f"\n集成模型SHAP分析结果已保存到: {output_csv_file}")
else:
    print("\n没有生成SHAP分析结果。")

# ============== 绘图部分 ==============
if shap_results:
    num_methods = len(shap_results)

    # 动态调整图片尺寸
    fig_width = 6 * num_methods  # 每个子图6英寸宽度
    fig_height = 8  # 固定高度为8英寸

    # 创建图形
    fig, axes = plt.subplots(1, num_methods, figsize=(fig_width, fig_height))

    if num_methods == 1:
        axes = [axes]

    # 修改：设置颜色为 #4d8f74 (绿色)
    bar_color = '#4d8f74'  # 绿色

    for i, (method_name, result) in enumerate(shap_results.items()):
        features = result['features']
        importances = result['mean_abs_shap']

        if len(features) != len(importances):
            min_len = min(len(features), len(importances))
            features = features[:min_len]
            importances = importances[:min_len]

        # 移除 'Deposition_Method' 特征
        non_depo_method_mask = np.array(features) != 'Deposition_Method'

        if not np.any(non_depo_method_mask):
            print(f"警告: 移除 'Deposition_Method' 后，{method_name} 没有剩余特征用于绘图。")
            continue

        # 筛选特征和重要性
        filtered_features = np.array(features)[non_depo_method_mask].tolist()
        filtered_importances = importances[non_depo_method_mask]

        # 替换特征名为别名
        aliased_features = []
        for feature in filtered_features:
            aliased_features.append(FEATURE_ALIASES.get(feature, feature))

        # 获取前15个最重要特征
        num_top_features = min(15, len(aliased_features))
        importance_df = pd.DataFrame({
            'Feature': aliased_features,
            'Importance': filtered_importances,
            'Original_Feature': filtered_features
        }).sort_values('Importance', ascending=False).head(num_top_features)

        if importance_df.empty:
            print(f"警告: {method_name} 没有足够的特征用于绘图。")
            continue

        # 绘制水平柱状图 - 修改：给每个柱子加上0.8pt的边框
        ax = axes[i]
        bars = ax.barh(range(len(importance_df)), importance_df['Importance'],
                       color=bar_color, alpha=0.8, edgecolor='black', linewidth=0.8)

        # 设置y轴标签（使用别名）- 修改：加大字体大小
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['Feature'], fontsize=12)

        # 在所有子图下方显示x轴标签和刻度值 - 修改：加大字体大小
        ax.set_xlabel('mean(|SHAP value|)', fontsize=13)

        # 设置坐标轴刻度方向向里 - 修改：刻度线宽度为0.8pt
        ax.tick_params(axis='x', direction='in', labelsize=11, width=0.8)
        ax.tick_params(axis='y', direction='in', labelsize=11, width=0.8)

        # 确保x轴刻度值在所有子图下方都显示
        ax.tick_params(axis='x', bottom=True, labelbottom=True)

        # 设置子图标题，显示方法名和数据量信息 - 修改：加大字体大小
        ax.set_title(f'{method_name} (n={result["data_size"]})',
                     fontsize=13, pad=10)

        ax.invert_yaxis()

        # 去掉网格虚线显示
        ax.grid(False)

        # 修改：将坐标轴边框加粗到0.8磅
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    plt.tight_layout()

    # 检查是否至少有一个子图被绘制
    if any(ax.has_data() for ax in axes):
        output_plot_file = 'deposition_methods_ensemble_shap_importance_active_area_lt_10_no_depo.tif'
        plt.savefig(output_plot_file, dpi=300, bbox_inches='tight', format='tiff')
        print(f"集成模型SHAP重要性图已保存到: {output_plot_file}")
    else:
        print("没有可绘制的SHAP结果（所有子图均无数据）。")
    plt.close(fig)
else:
    print("没有可绘制的SHAP结果。")

# 显示每种方法的最重要特征
print("\n=== 各方法最重要特征 (使用集成模型, Active_Area < 10) ===")
for method_name, result in shap_results.items():
    features = result['features']
    importances = result['mean_abs_shap']

    if len(features) != len(importances):
        min_len = min(len(features), len(importances))
        features = features[:min_len]
        importances = importances[:min_len]

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(10)

    print(f"\n{method_name} (n={result['data_size']}) (使用集成模型):")
    for _, row in importance_df.iterrows():
        feature_alias = FEATURE_ALIASES.get(row['Feature'], row['Feature'])
        print(f"  {feature_alias}: {row['Importance']:.4f}")

print("\n分析完成！(使用集成模型, Active_Area < 10)")