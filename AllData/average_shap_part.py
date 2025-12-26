import os
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import shap
import matplotlib.pyplot as plt
import warnings
from matplotlib import rcParams

# 配置设置
warnings.filterwarnings('ignore')
# 设置字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
rcParams.update({'font.size': 12})

# 创建输出目录
os.makedirs("picture", exist_ok=True)

# 加载数据
df = pd.read_excel("FinalData10312All.xlsx")
print(f"数据总行数: {len(df)}")

# 检查Structure列的值分布
print("\nStructure列的值分布:")
print(df['Structure'].value_counts())

# 首先检查列名
print(f"\n数据列名: {df.columns.tolist()}")

# 查看Structure列的唯一值
structure_values = df['Structure'].unique()
print(f"Structure列的唯一值: {structure_values}")

# 加载映射文件，查看Structure的映射关系
try:
    mapping_df = pd.read_csv('label_mappings/full_mapping_summary.csv')
    print("✅ 映射文件加载成功")

    # 查看Structure特征的映射关系
    structure_mapping = mapping_df[mapping_df['Feature'] == 'Structure']
    if len(structure_mapping) > 0:
        print("Structure特征映射关系:")
        for _, row in structure_mapping.iterrows():
            print(f"  {row['Original']} -> {row['Encoded']}")

        # 查找p-i-n对应的编码值
        pin_mapping = structure_mapping[structure_mapping['Original'] == 'p-i-n']
        if len(pin_mapping) > 0:
            pin_code = pin_mapping['Encoded'].iloc[0]
            print(f"找到p-i-n结构的编码值: {pin_code}")
            df_pin = df[df['Structure'] == pin_code].copy()
            print(f"筛选后数据量: {len(df_pin)} 条 (p-i-n结构)")
        else:
            print("⚠️ 映射文件中未找到p-i-n的映射，尝试使用编码值1")
            df_pin = df[df['Structure'] == 1].copy()
            print(f"筛选后数据量: {len(df_pin)} 条 (使用编码值1)")
    else:
        print("⚠️ 映射文件中未找到Structure特征，尝试使用编码值1")
        df_pin = df[df['Structure'] == 1].copy()
        print(f"筛选后数据量: {len(df_pin)} 条 (使用编码值1)")

except Exception as e:
    print(f"❌ 映射文件加载失败: {e}")
    print("⚠️ 映射文件加载失败，尝试使用编码值1")
    df_pin = df[df['Structure'] == 1].copy()
    print(f"筛选后数据量: {len(df_pin)} 条 (使用编码值1)")
    mapping_df = None

# 如果没有找到数据，使用全部数据
if len(df_pin) == 0:
    print("警告: 未找到p-i-n结构数据，使用全部数据进行分析")
    df_pin = df.copy()

y = df_pin['PCE']
X = df_pin.drop('PCE', axis=1)

# 保存原始列名
original_columns = X.columns.tolist()

# 加载所有模型和权重
print("\n加载集成模型...")
models = {}
weights = {}

# 模型权重配置（基于测试集R²）
model_configs = {
    'rf': {'path': 'models/best_rf_model.pkl', 'r2': 0.7330},
    'xgb': {'path': 'models/best_xgb_model.pkl', 'r2': 0.7611},
    'catboost': {'path': 'models/best_catboost_model.pkl', 'r2': 0.7386},
    'lgbm': {'path': 'models/best_lgbm_model.pkl', 'r2': 0.7316}
}

# 尝试不同的CatBoost模型路径
catboost_paths = [
    'models/best_catboost_model.pkl',
    'models/best_catboost_model.cbm',
    'models/best_catboost_model_part.cbm'
]

# 计算总R²用于权重归一化
total_r2 = sum(config['r2'] for config in model_configs.values())

# 加载模型并计算权重
successful_models = 0
for model_name, config in model_configs.items():
    try:
        if model_name == 'catboost':
            # 尝试多个可能的CatBoost模型路径
            catboost_loaded = False
            for path in catboost_paths:
                try:
                    if path.endswith('.cbm'):
                        model = CatBoostRegressor()
                        model.load_model(path)
                    else:
                        model = joblib.load(path)
                    catboost_loaded = True
                    print(f"✅ CatBoost模型从 {path} 加载成功")
                    break
                except Exception as e:
                    continue

            if not catboost_loaded:
                raise Exception("所有CatBoost模型路径都失败")
        else:
            model = joblib.load(config['path'])

        models[model_name] = model
        weights[model_name] = config['r2'] / total_r2
        successful_models += 1
        print(f"✅ {model_name.upper()}模型加载成功, 权重: {weights[model_name]:.4f}")
    except Exception as e:
        print(f"❌ {model_name.upper()}模型加载失败: {e}")

# 如果没有模型成功加载，退出
if successful_models == 0:
    print("❌ 所有模型加载失败，无法进行分析")
    exit(1)

print("\n模型权重汇总:")
for model_name, weight in weights.items():
    if model_name in models:
        print(f"  {model_name.upper()}: {weight:.4f}")

# 如果CatBoost加载失败，重新计算权重
if 'catboost' not in models:
    print("\n⚠️ CatBoost模型加载失败，重新计算其他模型的权重...")
    remaining_r2 = sum(config['r2'] for model_name, config in model_configs.items()
                       if model_name in models)
    for model_name in models:
        weights[model_name] = model_configs[model_name]['r2'] / remaining_r2

    print("调整后的模型权重:")
    for model_name, weight in weights.items():
        if model_name in models:
            print(f"  {model_name.upper()}: {weight:.4f}")

# 获取所有分类特征
if mapping_df is not None:
    categorical_features = mapping_df['Feature'].unique()
    print(f"分类特征: {categorical_features}")
else:
    categorical_features = []


def apply_feature_mapping(X_data, mapping_df, categorical_features):
    """应用特征映射，将分类特征转换为编码值"""
    X_mapped = X_data.copy()

    for feature in categorical_features:
        if feature in X_mapped.columns:
            # 获取该特征的映射关系
            feature_mapping = mapping_df[mapping_df['Feature'] == feature]

            if len(feature_mapping) > 0:
                # 创建映射字典
                mapping_dict = {}
                for _, row in feature_mapping.iterrows():
                    if pd.isna(row['Original']):
                        mapping_dict[None] = row['Encoded']
                    else:
                        mapping_dict[row['Original']] = row['Encoded']

                # 应用映射
                def map_value(value):
                    if pd.isna(value):
                        return mapping_dict.get(None, 0)
                    else:
                        return mapping_dict.get(value, 0)

                X_mapped[feature] = X_mapped[feature].apply(map_value)
                print(f"  映射特征 '{feature}': {len(feature_mapping)} 个映射值")
            else:
                print(f"  ⚠️ 特征 '{feature}' 在映射文件中未找到映射关系，使用默认值0")
                X_mapped[feature] = 0
        else:
            print(f"  ⚠️ 特征 '{feature}' 在数据中不存在")

    return X_mapped


def calculate_ensemble_shap_values(X_data):
    """计算集成模型的加权SHAP值"""
    # 应用特征映射
    if mapping_df is not None:
        X_data_mapped = apply_feature_mapping(X_data, mapping_df, categorical_features)
    else:
        X_data_mapped = X_data.copy()

    # 初始化加权SHAP值
    weighted_shap_values = None
    total_weight = 0

    for model_name, model in models.items():
        try:
            print(f"计算 {model_name.upper()} 的SHAP值...")

            if model_name == 'catboost':
                # 对于CatBoost，使用Pool
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(Pool(X_data_mapped))
            else:
                # 对于其他模型
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_data_mapped)

            # 加权SHAP值
            model_weight = weights[model_name]
            if weighted_shap_values is None:
                weighted_shap_values = shap_values * model_weight
            else:
                weighted_shap_values += shap_values * model_weight

            total_weight += model_weight
            print(f"  {model_name.upper()} SHAP值计算完成")

        except Exception as e:
            print(f"  ❌ {model_name.upper()} SHAP值计算失败: {e}")

    # 归一化加权SHAP值
    if weighted_shap_values is not None:
        weighted_shap_values /= total_weight
        print("集成模型SHAP值计算完成")
        return weighted_shap_values
    else:
        raise Exception("所有模型的SHAP值计算都失败了")


def format_feature_name(feature_name):
    """格式化特征名称，将特定特征替换为更友好的显示名称"""
    replacements = {
        'FA': 'FA ratio',
        'MA': 'MA ratio',
        'Br': 'Br ratio',
        'Precursor_Solution': 'Pre_Sol',
        'P2etching_Power(W)': 'P2Power',
        'P1etching_Power(W)': 'P1Power',
        'P3etching_Power(W)': 'P3Power',
        'HTL_Passivator': 'HTL_Psvt',
        'HTL-Addictive': 'HTL-Add',
        'P2etching_Power_percentage(%)': 'P2Power%',
        'total_scribing_line_width(μm)': 'totalWidth',
        'P2Width(μm)': 'P2Width',
        'ETL_Passivator': 'ETL_Psvt',
        'ETL-Addictive': 'ETL-Add',
        'Annealing_Time2': 'AnnealingT2',
        'Precursor_Solution_Addictive': 'Pre_Sol_Add',
        'Annealing_Temperature1': 'AnnealingTp1',
        'Annealing_Time1': 'AnnealingT1'
    }

    # 检查是否是完全匹配的特征名
    if feature_name in replacements:
        return replacements[feature_name]

    # 检查是否是包含这些关键词的特征名
    for key, replacement in replacements.items():
        if key in feature_name:
            return feature_name.replace(key, replacement)

    return feature_name


def plot_importance_bar(shap_values, X_data, filename, color):
    """绘制重要性柱状图（带自定义颜色）"""
    # 设置图形大小
    fig_width_cm = 8
    fig_height_cm = 12
    fig_width_inch = fig_width_cm / 2.54
    fig_height_inch = fig_height_cm / 2.54

    # 创建图形和子图
    fig, ax = plt.subplots(figsize=(fig_width_inch, fig_height_inch))

    # 计算平均绝对SHAP值
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    features = X_data.columns

    # 定义要从图中排除的特征列表
    exclude_features = [
        'P1Wavelength(nm)',
        'P2Wavelength(nm)',
        'P3Wavelength(nm)',
        'total_scribing_line_width(μm)',
        'P1Width(μm)',
        'P2Width(μm)',
        'P3Width(μm)',
        'Active_Area',
        'GFF',
        'P1etching_frequency(kHz)',
        'P1etching_Power(W)',
        'P1etching_Power_percentage(%)',
        'P2etching_frequency(kHz)',
        'P2etching_Power(W)',
        'P2etching_Power_percentage(%)',
        'P3etching_frequency(kHz)',
        'P3etching_Power(W)',
        'P3etching_Power_percentage(%)',
        'P1_P2Scribing_Spacing(μm)',
        'P2_P3Scribing_Spacing(μm)',
        'brand',
        'Structure'  # 添加Structure到排除列表
    ]

    # 创建掩码，排除指定的特征
    mask = ~features.isin(exclude_features)
    filtered_features = features[mask]
    filtered_shap = mean_abs_shap[mask]

    # 取top15（排除指定特征后）
    if len(filtered_features) > 15:
        sorted_idx = np.argsort(filtered_shap)[-15:]
        top_features = [filtered_features[i] for i in sorted_idx]
        top_shap = filtered_shap[sorted_idx]
    else:
        sorted_idx = np.argsort(filtered_shap)
        top_features = [filtered_features[i] for i in sorted_idx]
        top_shap = filtered_shap[sorted_idx]

    # 格式化特征名称
    formatted_features = [format_feature_name(feature) for feature in top_features]

    # 绘制柱状图（使用指定颜色）
    bars = ax.barh(range(len(top_features)), top_shap, color=color, height=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(formatted_features, fontname='Times New Roman', fontsize=10)
    ax.set_xlabel('mean absolute SHAP value', fontname='Times New Roman', fontsize=10)

    # 设置x轴刻度字体
    ax.tick_params(axis='x', labelsize=9, width=0.5)
    ax.tick_params(axis='y', labelsize=9, width=0.5)

    # 设置坐标轴粗细为0.5pt
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # 调整布局，确保所有元素都能显示
    plt.tight_layout(pad=2.0)

    # 保存为TIFF格式
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1, format='tiff')
    plt.close()
    print(f"成功保存: {filename}")

    # 返回完整的特征重要性数据（不排除任何特征），用于CSV保存
    importance_df = pd.DataFrame({
        'feature': features,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)

    return importance_df


# 颜色配置
COLOR_CONFIG = {
    "all": '#046B38'  # 只需要全部数据的颜色
}

# 分析所有p-i-n结构数据
print("\n分析p-i-n结构数据...")
try:
    # 计算集成模型的加权SHAP值
    shap_values_all = calculate_ensemble_shap_values(X)
    importance_df = plot_importance_bar(
        shap_values_all,
        X,
        "picture/shap_top15_pin_structure_ensemble.tif",  # 改为TIFF格式
        COLOR_CONFIG["all"]
    )

    # 保存特征重要性数据到CSV（包含所有特征）
    importance_df.to_csv("picture/feature_importance_pin_structure_ensemble.csv", index=False)
    print("特征重要性数据已保存到: picture/feature_importance_pin_structure_ensemble.csv")

    # 打印前15个最重要的特征（包含所有特征）
    print("\n=== Top 15 最重要特征 (集成模型) ===")
    for i, (feature, importance) in enumerate(zip(importance_df['feature'][:15], importance_df['mean_abs_shap'][:15]),
                                              1):
        print(f"{i}. {feature}: {importance:.6f}")

    print("\n=== 集成模型分析完成 ===")
    print("生成的图片文件: picture/shap_top15_pin_structure_ensemble.tif")
    print("生成的数据文件: picture/feature_importance_pin_structure_ensemble.csv")

except Exception as e:
    print(f"集成模型分析过程中出错: {e}")
    print("尝试使用单个模型作为替代...")

    # 如果集成分析失败，尝试使用性能最好的单个模型（XGBoost）
    try:
        print("使用XGBoost模型作为替代...")
        xgb_model = models.get('xgb')
        if xgb_model is not None:
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X)

            importance_df = plot_importance_bar(
                shap_values,
                X,
                "picture/shap_top15_pin_structure_xgb.tif",
                COLOR_CONFIG["all"]
            )

            importance_df.to_csv("picture/feature_importance_pin_structure_xgb.csv", index=False)
            print("XGBoost模型特征重要性数据已保存")
        else:
            print("XGBoost模型不可用")
    except Exception as e2:
        print(f"替代方法也失败: {e2}")