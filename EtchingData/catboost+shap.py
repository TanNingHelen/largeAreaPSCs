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
rcParams.update({'font.size': 10})

# 创建输出目录
os.makedirs("picture", exist_ok=True)

# 加载数据
df = pd.read_excel("FinalData.xlsx")
y = df['PCE']
X = df.drop('PCE', axis=1)

# 保存原始列名（用于 CatBoost）
original_columns = X.columns.tolist()

# 统一列名格式（用于其他处理）
X.columns = [col.replace(' ', '_') for col in X.columns]

# 加载CatBoost模型
print("\n加载CatBoost模型...")
catboost_model = CatBoostRegressor()
catboost_model.load_model("models/best_catboost_model.cbm")
print("CatBoost模型加载成功!")


def calculate_shap_values(X_data, use_original_columns=False):
    """计算CatBoost模型的SHAP值"""
    if use_original_columns:
        # 对于 CatBoost，使用原始列名
        X_data_catboost = X_data.copy()
        X_data_catboost.columns = original_columns
        explainer = shap.TreeExplainer(catboost_model)
        shap_values = explainer.shap_values(Pool(X_data_catboost))
    else:
        explainer = shap.TreeExplainer(catboost_model)
        shap_values = explainer.shap_values(Pool(X_data))
    return shap_values


def format_feature_name(feature_name):
    """格式化特征名称，将特定特征替换为更友好的显示名称"""
    replacements = {
        'FA': 'FA ratio',
        'MA': 'MA ratio',
        'Br': 'Br ratio',
        'Precursor_Solution': 'Pre_Sol',
        'P2etching_Power(W)': 'P2 Power',
        'P1etching_Power(W)': 'P1 Power',
        'P3etching_Power(W)': 'P3 Power',
        'HTL_Passivator': 'HTL-Psvt',
        'HTL-Addictive': 'HTL-Add',
        'P2etching_Power_percentage(%)': 'P2 Power per',
        'total_scribing_line_width(μm)': 'total Width',
        'P2Width(μm)': 'P2 Width',
        'P1Width(μm)': 'P1 Width',
        'P3Width(μm)': 'P3 Width',
        'P1Wavelength(nm)': 'P1 Wavelength',
        'P2Wavelength(nm)': 'P2 Wavelength',
        'P3Wavelength(nm)': 'P3 Wavelength',
        'P1etching_frequency(kHz)': 'P1 Frequency',
        'P2etching_frequency(kHz)': 'P2 Frequency',
        'P3etching_frequency(kHz)': 'P3 Frequency',
        'P1_P2Scribing_Spacing(μm)': 'P1P2 Spacing',
        'P2_P3Scribing_Spacing(μm)': 'P2P3 Spacing',
        'P1etching_Power_percentage(%)': 'P1 Power per',
        'P3etching_Power_percentage(%)': 'P3 Power per',
        'P2Scan_Velocity': 'P2 Velocity',
        'P3Scan_Velocity': 'P3 Velocity',
        'P1Scan_Velocity(mm/s)': 'P1 Velocity',
        'P1Spot_Size(μm)': 'P1 Spot Size',
        'P2Spot_Size(μm)': 'P2 Spot Size',
        'P3Spot_Size(μm)': 'P3 Spot Size'

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
    # 设置图形大小（8cm宽，高度根据特征数量调整）
    fig_width_cm = 8
    fig_height_cm = 15  # 增加高度以容纳更多特征和避免标签被截断
    fig_width_inch = fig_width_cm / 2.54
    fig_height_inch = fig_height_cm / 2.54

    # 创建图形和子图，调整布局
    fig, ax = plt.subplots(figsize=(fig_width_inch, fig_height_inch))

    # 计算平均绝对SHAP值
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    features = X_data.columns
    sorted_idx = np.argsort(mean_abs_shap)[-20:]  # 取top20

    # 格式化特征名称
    formatted_features = [format_feature_name(features[i]) for i in sorted_idx]

    # 绘制柱状图（使用指定颜色）
    bars = ax.barh(range(len(sorted_idx)), mean_abs_shap[sorted_idx], color=color, height=0.7)
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels(formatted_features, fontname='Times New Roman')
    ax.set_xlabel('mean absolute SHAP value', fontname='Times New Roman', fontsize=10)

    # 设置x轴刻度字体
    ax.tick_params(axis='x', labelsize=9, width=0.5)
    ax.tick_params(axis='y', labelsize=9, width=0.5)

    # 设置坐标轴粗细为0.5pt
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # 调整布局，确保所有元素都能显示
    plt.tight_layout(pad=2.0)  # 增加内边距

    # 保存图像为TIFF格式，确保所有元素都在图像内
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1, format='tiff')
    plt.close()
    print(f"成功保存: {filename}")

    # 返回特征重要性数据，用于后续分析
    importance_df = pd.DataFrame({
        'feature': features,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)

    return importance_df


def plot_laser_scribing_importance(shap_values, X_data, filename):
    """绘制激光划刻参数重要性排序柱状图"""
    # 定义激光划刻参数列表 - 修改：使用下划线格式的特征名
    laser_scribing_features = [
        'P1Wavelength(nm)',
        'P2Wavelength(nm)',
        'P3Wavelength(nm)',
        'total_scribing_line_width(μm)',
        'P1Width(μm)',
        'P2Width(μm)',
        'P3Width(μm)',
        'GFF',
        'P1Scan_Velocity(mm/s)',
        'P1Spot_Size(μm)',  # 改为下划线
        'P1etching_frequency(kHz)',
        'P1etching_Power(W)',
        'P1etching_Power_percentage(%)',
        'P2Scan_Velocity',
        'P2Spot_Size(μm)',  # 改为下划线
        'P2etching_frequency(kHz)',
        'P2etching_Power(W)',
        'P2etching_Power_percentage(%)',
        'P3Scan_Velocity',
        'P3Spot_Size(μm)',  # 改为下划线
        'P3etching_frequency(kHz)',
        'P3etching_Power(W)',
        'P3etching_Power_percentage(%)',
        'P1_P2Scribing_Spacing(μm)',
        'P2_P3Scribing_Spacing(μm)',
        'brand',
        'subsubmodule_number',
        'Type'
    ]

    # 添加调试信息：打印所有激光划刻特征和实际数据中的特征
    print("\n=== 激光划刻特征匹配调试 ===")
    print("激光划刻特征列表:")
    for feature in laser_scribing_features:
        print(f"  - {feature}")

    print("\n数据中的特征（前30个）:")
    for i, feature in enumerate(X_data.columns[:30]):
        print(f"  {i + 1}. {feature}")

    # 查找与激光划刻相关的所有特征
    print("\n查找包含以下关键词的特征:")
    laser_keywords = ['P1', 'P2', 'P3', 'Spot', 'Size', 'Width', 'Wavelength', 'Scan', 'Velocity', 'etching', 'Power',
                      'frequency', 'Spacing']
    for keyword in laser_keywords:
        matching_features = [f for f in X_data.columns if keyword.lower() in f.lower()]
        if matching_features:
            print(f"关键词 '{keyword}': {matching_features}")

    # 筛选存在于数据中的激光划刻参数
    available_laser_features = [f for f in laser_scribing_features if f in X_data.columns]
    print(f"\n找到 {len(available_laser_features)} 个激光划刻参数:")
    for f in available_laser_features:
        print(f"  - {f}")

    # 如果某些特征没找到，尝试使用更灵活的方式查找
    missing_features = [f for f in laser_scribing_features if f not in X_data.columns]
    if missing_features:
        print(f"\n未直接找到的特征: {missing_features}")
        print("尝试使用部分匹配查找...")
        for missing_feature in missing_features:
            # 移除单位部分进行匹配
            base_name = missing_feature.split('(')[0] if '(' in missing_feature else missing_feature
            matching = [f for f in X_data.columns if base_name.lower() in f.lower()]
            if matching:
                print(f"  对于 '{missing_feature}'，找到可能匹配的特征: {matching}")
                # 使用第一个匹配
                available_laser_features.extend(matching[:1])

    # 设置图形大小
    fig_width_cm = 10
    fig_height_cm = 12
    fig_width_inch = fig_width_cm / 2.54
    fig_height_inch = fig_height_cm / 2.54

    # 创建图形和子图
    fig, ax = plt.subplots(figsize=(fig_width_inch, fig_height_inch))

    # 计算平均绝对SHAP值
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    features = X_data.columns

    # 获取激光划刻参数的重要性
    laser_importance = []
    for feature in available_laser_features:
        if feature in features:
            idx = list(features).index(feature)
            laser_importance.append((feature, mean_abs_shap[idx]))

    # 按重要性排序
    laser_importance.sort(key=lambda x: x[1], reverse=True)

    # 准备绘图数据
    laser_features = [item[0] for item in laser_importance]
    laser_values = [item[1] for item in laser_importance]

    # 格式化特征名称
    formatted_laser_features = [format_feature_name(feature) for feature in laser_features]

    # 绘制柱状图 - 修改：添加边框和调整样式
    bars = ax.barh(range(len(laser_features)), laser_values,
                   color='#1f77b4', height=0.7, edgecolor='black', linewidth=0.8)

    ax.set_yticks(range(len(laser_features)))
    ax.set_yticklabels(formatted_laser_features, fontname='Times New Roman')
    ax.set_xlabel('mean(|SHAP value|)', fontname='Times New Roman', fontsize=10)
    # ax.set_title('Laser Scribing Parameters Importance', fontname='Times New Roman', fontsize=12)

    # 修改：设置坐标轴刻度方向向里，宽度为0.8pt
    ax.tick_params(axis='x', labelsize=9, direction='in', width=0.8)
    ax.tick_params(axis='y', labelsize=9, direction='in', width=0.8)

    # 修改：设置坐标轴边框为0.8pt
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # 调整布局
    plt.tight_layout(pad=2.0)

    # 保存为TIFF格式
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1, format='tiff')
    plt.close()
    print(f"成功保存激光划刻参数重要性图: {filename}")

    # 返回激光划刻参数重要性数据
    laser_importance_df = pd.DataFrame({
        'feature': laser_features,
        'mean_abs_shap': laser_values
    }).sort_values('mean_abs_shap', ascending=False)

    return laser_importance_df


# 颜色配置
COLOR_CONFIG = {
    "all": '#046B38',  # 绿色
    "small_area": '#1f77b4',  # 蓝色
    "medium_area": '#ff7f0e',  # 橙色
    "large_area": '#d62728'  # 红色
}

# 分析所有数据
print("\n分析所有数据...")
# 对于 CatBoost，使用原始列名
shap_values_all = calculate_shap_values(X, use_original_columns=True)
importance_df_all = plot_importance_bar(
    shap_values_all,
    X,  # 这里使用修改后的列名
    "picture/shap_top20_all_catboost.tif",  # 改为TIFF格式
    COLOR_CONFIG["all"]
)

# 绘制激光划刻参数重要性图
print("\n绘制激光划刻参数重要性图...")
laser_importance_df = plot_laser_scribing_importance(
    shap_values_all,
    X,
    "picture/laser_scribing_importance.tif"
)

# 根据Active_Area分组
active_area_col = df.columns[df.columns.str.contains('Active_Area', case=False)][0]  # 找到Active_Area列名
print(f"\nActive_Area列名为: {active_area_col}")

# 分组条件
small_area_mask = (df[active_area_col] >= 1) & (df[active_area_col] < 10)
medium_area_mask = (df[active_area_col] >= 10) & (df[active_area_col] < 100)
large_area_mask = df[active_area_col] >= 100

print(f"小面积组 (1-10): {sum(small_area_mask)} 个样本")
print(f"中面积组 (10-100): {sum(medium_area_mask)} 个样本")
print(f"大面积组 (>=100): {sum(large_area_mask)} 个样本")

# 分析小面积组 (1 <= Active_Area < 10)
if sum(small_area_mask) > 0:
    print("\n分析小面积组 (1 <= Active_Area < 10)...")
    X_small = X[small_area_mask]
    y_small = y[small_area_mask]

    # 重新构建Pool数据以确保列名匹配
    X_small_original = X_small.copy()
    X_small_original.columns = original_columns
    shap_values_small = calculate_shap_values(X_small_original, use_original_columns=True)

    importance_df_small = plot_importance_bar(
        shap_values_small,
        X_small,
        "picture/shap_top20_small_area_catboost.tif",  # 改为TIFF格式
        COLOR_CONFIG["small_area"]
    )
else:
    print("\n小面积组 (1 <= Active_Area < 10) 无数据")
    importance_df_small = None

# 分析中面积组 (10 <= Active_Area < 100)
if sum(medium_area_mask) > 0:
    print("\n分析中面积组 (10 <= Active_Area < 100)...")
    X_medium = X[medium_area_mask]
    y_medium = y[medium_area_mask]

    # 重新构建Pool数据以确保列名匹配
    X_medium_original = X_medium.copy()
    X_medium_original.columns = original_columns
    shap_values_medium = calculate_shap_values(X_medium_original, use_original_columns=True)

    importance_df_medium = plot_importance_bar(
        shap_values_medium,
        X_medium,
        "picture/shap_top20_medium_area_catboost.tif",  # 改为TIFF格式
        COLOR_CONFIG["medium_area"]
    )
else:
    print("\n中面积组 (10 <= Active_Area < 100) 无数据")
    importance_df_medium = None

# 分析大面积组 (Active_Area >= 100)
if sum(large_area_mask) > 0:
    print("\n分析大面积组 (Active_Area >= 100)...")
    X_large = X[large_area_mask]
    y_large = y[large_area_mask]

    # 重新构建Pool数据以确保列名匹配
    X_large_original = X_large.copy()
    X_large_original.columns = original_columns
    shap_values_large = calculate_shap_values(X_large_original, use_original_columns=True)

    importance_df_large = plot_importance_bar(
        shap_values_large,
        X_large,
        "picture/shap_top20_large_area_catboost.tif",  # 改为TIFF格式
        COLOR_CONFIG["large_area"]
    )
else:
    print("\n大面积组 (Active_Area >= 100) 无数据")
    importance_df_large = None

# 保存所有特征重要性数据到CSV
importance_df_all.to_csv("picture/feature_importance_all_catboost.csv", index=False)
print("全部数据特征重要性已保存到: picture/feature_importance_all_catboost.csv")

# 保存激光划刻参数重要性数据到CSV
laser_importance_df.to_csv("picture/laser_scribing_importance.csv", index=False)
print("激光划刻参数重要性已保存到: picture/laser_scribing_importance.csv")

if importance_df_small is not None:
    importance_df_small.to_csv("picture/feature_importance_small_area_catboost.csv", index=False)
    print("小面积组特征重要性已保存到: picture/feature_importance_small_area_catboost.csv")

if importance_df_medium is not None:
    importance_df_medium.to_csv("picture/feature_importance_medium_area_catboost.csv", index=False)
    print("中面积组特征重要性已保存到: picture/feature_importance_medium_area_catboost.csv")

if importance_df_large is not None:
    importance_df_large.to_csv("picture/feature_importance_large_area_catboost.csv", index=False)
    print("大面积组特征重要性已保存到: picture/feature_importance_large_area_catboost.csv")

# 打印各组前20个最重要的特征
print("\n=== 全部数据 Top 20 最重要特征 ===")
for i, (feature, importance) in enumerate(
        zip(importance_df_all['feature'][:20], importance_df_all['mean_abs_shap'][:20]), 1):
    print(f"{i}. {feature}: {importance:.6f}")

# 打印激光划刻参数重要性
print("\n=== 激光划刻参数重要性排序 ===")
for i, (feature, importance) in enumerate(
        zip(laser_importance_df['feature'], laser_importance_df['mean_abs_shap']), 1):
    print(f"{i}. {feature}: {importance:.6f}")

if importance_df_small is not None:
    print("\n=== 小面积组 (1-10) Top 20 最重要特征 ===")
    for i, (feature, importance) in enumerate(
            zip(importance_df_small['feature'][:20], importance_df_small['mean_abs_shap'][:20]), 1):
        print(f"{i}. {feature}: {importance:.6f}")

if importance_df_medium is not None:
    print("\n=== 中面积组 (10-100) Top 20 最重要特征 ===")
    for i, (feature, importance) in enumerate(
            zip(importance_df_medium['feature'][:20], importance_df_medium['mean_abs_shap'][:20]), 1):
        print(f"{i}. {feature}: {importance:.6f}")

if importance_df_large is not None:
    print("\n=== 大面积组 (>=100) Top 20 最重要特征 ===")
    for i, (feature, importance) in enumerate(
            zip(importance_df_large['feature'][:20], importance_df_large['mean_abs_shap'][:20]), 1):
        print(f"{i}. {feature}: {importance:.6f}")

print("\n=== 分析完成 ===")
print("生成的TIFF图片文件:")
print("- picture/shap_top20_all_catboost.tif")
print("- picture/laser_scribing_importance.tif")
print("- picture/shap_top20_small_area_catboost.tif")
print("- picture/shap_top20_medium_area_catboost.tif")
print("- picture/shap_top20_large_area_catboost.tif")
print("生成的数据文件:")
print("- picture/feature_importance_all_catboost.csv")
print("- picture/laser_scribing_importance.csv")
print("- picture/feature_importance_small_area_catboost.csv")
print("- picture/feature_importance_medium_area_catboost.csv")
print("- picture/feature_importance_large_area_catboost.csv")