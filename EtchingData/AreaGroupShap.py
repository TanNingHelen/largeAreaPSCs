import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib import rcParams
from catboost import CatBoostRegressor, Pool
import warnings

# ============== 配置部分 ==============
warnings.filterwarnings('ignore')

# 字体设置为Times New Roman
rcParams['font.family'] = 'Times New Roman'
rcParams['axes.unicode_minus'] = False

# 颜色方案保持原样
ACTIVE_AREA_BINS = {
    '<10': '#4d8f74',  # 绿
    '10-100': '#a94837',  # 红
    '≥100': '#ffbe7a'  # 黄
}
BIN_EDGES = [-np.inf, 10, 100, np.inf]
TOP_FEATURES = 15  # 每组显示的特征数量

# 特征名称映射
FEATURE_NAME_MAPPING = {
    'FA': 'FA ratio',
    'MA': 'MA ratio',
    'Br': 'Br ratio',
    'Precursor_Solution': 'Pre-Sol',
    'P2etching_Power(W)': 'P2 Power',
    'P1etching_Power(W)': 'P1 Power',
    'P3etching_Power(W)': 'P3 Power',
    'HTL_Passivator': 'HTL-Psvt',
    'HTL-Addictive': 'HTL-Add',
    'P2etching_Power_percentage(%)': 'P2 Power per',
    'total_scribing_line_width(μm)': 'total Width',
    'P2Width(μm)': 'P2 Width',
    'submodule_number': 'sub PSMs Num',
    'P2Scan_Velocity': 'P2 Velocity',
    'Active_Area': 'Active-Area'
}

# 子图标题映射 - 修改为small-area, medium-area, large-area
TITLE_MAPPING = {
    '<10': 'small-area',
    '10-100': 'medium-area',
    '≥100': 'large-area'
}


# ============== 数据加载和预处理 ==============
def load_and_preprocess():
    """加载CatBoost模型和原始数据"""
    # 加载CatBoost模型
    model = CatBoostRegressor()
    model.load_model("models/best_catboost_model.cbm")

    # 加载数据
    df = pd.read_excel("FinalData.xlsx")

    # 保存原始列名（用于CatBoost）
    original_columns = df.drop(['PCE'], axis=1).columns.tolist()

    # 统一列名格式（替换空格为下划线）
    df.columns = [col.replace(' ', '_') for col in df.columns]

    y = df['PCE']
    X = df.drop(['PCE'], axis=1)

    return model, X, y, original_columns


def format_feature_name(feature_name):
    """格式化特征名称，将特定特征替换为更友好的显示名称"""
    # 检查是否是完全匹配的特征名
    if feature_name in FEATURE_NAME_MAPPING:
        return FEATURE_NAME_MAPPING[feature_name]

    # 检查是否是包含这些关键词的特征名
    for key, replacement in FEATURE_NAME_MAPPING.items():
        if key in feature_name:
            return feature_name.replace(key, replacement)

    return feature_name


def calculate_shap_values_for_group(model, X_group, use_original_columns=False, original_columns=None):
    """计算特定分组的SHAP值"""
    try:
        if use_original_columns and original_columns is not None:
            # 对于CatBoost，使用原始列名
            X_group_catboost = X_group.copy()
            X_group_catboost.columns = original_columns
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(Pool(X_group_catboost))
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(Pool(X_group))
        return shap_values
    except Exception as e:
        print(f"计算SHAP值时出错: {e}")
        return None


def plot_group_comparison(model, X_data, y_data, original_columns):
    """分组对比图（按各分组自身重要性排序）"""
    # 确保Active_Area列存在
    active_area_col = 'Active_Area'
    if active_area_col not in X_data.columns:
        # 尝试找到类似的列名
        area_cols = [col for col in X_data.columns if 'active' in col.lower() or 'area' in col.lower()]
        if area_cols:
            active_area_col = area_cols[0]
        else:
            print("警告: 未找到Active_Area列，无法进行分组分析")
            return None

    # 根据Active_Area进行分组
    small_area_mask = (X_data[active_area_col] >= 1) & (X_data[active_area_col] < 10)
    medium_area_mask = (X_data[active_area_col] >= 10) & (X_data[active_area_col] < 100)
    large_area_mask = X_data[active_area_col] >= 100

    masks = {
        '<10': small_area_mask,
        '10-100': medium_area_mask,
        '≥100': large_area_mask
    }

    # 准备各分组的SHAP均值
    group_data = {}

    for label, mask in masks.items():
        if sum(mask) > 5:  # 仅处理有足够样本的分组
            print(f"\n分析{TITLE_MAPPING[label]}组...")
            print(f"样本数量: {sum(mask)}")

            X_group = X_data[mask]

            # 为该分组单独计算SHAP值
            shap_values_group = calculate_shap_values_for_group(
                model, X_group, use_original_columns=True, original_columns=original_columns
            )

            if shap_values_group is not None:
                # 计算平均绝对SHAP值
                group_mean_abs = np.abs(shap_values_group).mean(axis=0)

                # 排序
                sorted_idx = np.argsort(group_mean_abs)[::-1][:TOP_FEATURES]

                # 获取特征名称
                features = X_group.columns.tolist()

                group_data[label] = {
                    'values': group_mean_abs[sorted_idx],
                    'features': [format_feature_name(features[i]) for i in sorted_idx],
                    'raw_features': [features[i] for i in sorted_idx],
                    'count': sum(mask),
                    'shap_values': shap_values_group
                }

                # 打印调试信息
                print(f"\n=== {TITLE_MAPPING[label]} 组特征排序 ===")
                for i, (feature, shap_val) in enumerate(
                        zip(group_data[label]['raw_features'], group_data[label]['values'])):
                    print(f"{i + 1}. {feature}: {shap_val:.6f}")
            else:
                print(f"无法计算{TITLE_MAPPING[label]}组的SHAP值")

    # 设置图形尺寸（16cm宽）
    fig_width_cm = 16
    fig_height_cm = 10
    fig_width_inch = fig_width_cm / 2.54
    fig_height_inch = fig_height_cm / 2.54

    # 绘制对比图
    fig, axes = plt.subplots(1, 3, figsize=(fig_width_inch, fig_height_inch))

    # 找到所有组中最大的SHAP值，用于统一x轴范围
    max_val = 0
    for label in ACTIVE_AREA_BINS.keys():
        if label in group_data:
            max_val = max(max_val, np.max(group_data[label]['values']))
    max_val = max_val * 1.1  # 添加10%的余量

    for i, (ax, (label, color)) in enumerate(zip(axes, ACTIVE_AREA_BINS.items())):
        if label in group_data:
            data = group_data[label]
            ax.barh(
                data['features'][::-1],  # 倒序显示以保持重要特征在上方
                data['values'][::-1],
                color=color,
                edgecolor='black',  # 添加黑色边框
                linewidth=0.8  # 边框宽度
            )
            # 使用新的标题映射：small-area, medium-area, large-area
            ax.set_title(TITLE_MAPPING[label], fontname='Times New Roman')
            ax.set_xlim(0, max_val)  # 统一x轴范围

            # 设置坐标轴边框宽度为0.8pt
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)

            # 设置刻度线宽度为0.8pt，方向向里
            ax.tick_params(axis='both', which='major', width=0.8, direction='in')
            ax.tick_params(axis='both', which='minor', width=0.8, direction='in')

            # 设置刻度标签字体为Times New Roman
            ax.tick_params(axis='both', labelsize=10)
            for label_text in ax.get_xticklabels():
                label_text.set_fontname('Times New Roman')
            for label_text in ax.get_yticklabels():
                label_text.set_fontname('Times New Roman')
        else:
            ax.text(0.5, 0.5, f"No data for {TITLE_MAPPING[label]}",
                    ha='center', va='center', transform=ax.transAxes,
                    fontname='Times New Roman')

    axes[1].set_xlabel('mean(|SHAP value|)', fontname='Times New Roman', fontsize=11)

    # 移除其他子图的x轴标签
    axes[0].set_xlabel('')
    axes[2].set_xlabel('')

    plt.tight_layout()
    # 保存为TIFF格式
    plt.savefig("picture_predict/Group_Comparison.tif", dpi=300, bbox_inches='tight', format='tiff')
    plt.close()

    return group_data


# ============== 主程序 ==============
if __name__ == "__main__":
    os.makedirs("picture_predict", exist_ok=True)

    # 1. 数据加载和预处理
    print("加载模型和数据...")
    model, X_data, y_data, original_columns = load_and_preprocess()
    print(f"模型加载成功! 特征数量: {len(X_data.columns)}")

    # 打印前几个特征名
    print("前10个特征名:")
    for i, feature in enumerate(X_data.columns[:10]):
        print(f"  {i + 1}. {feature}")

    # 2. 计算各分组的SHAP值并绘制分组对比图
    print("\n计算各分组的SHAP值并绘制图表...")
    group_data = plot_group_comparison(model, X_data, y_data, original_columns)

    if group_data:
        # 3. 打印各组的详细信息
        print("\n" + "=" * 60)
        print("各组特征重要性详情")
        print("=" * 60)

        for label in ACTIVE_AREA_BINS:
            if label in group_data:
                print(f"\n{TITLE_MAPPING[label]}组 (样本数: {group_data[label]['count']}):")
                for i, (feature, shap_val) in enumerate(
                        zip(group_data[label]['raw_features'], group_data[label]['values'])):
                    print(f"  {i + 1:2d}. {feature:30s}: {shap_val:.6f}")

    print("\n生成图像：")
    print("- 分组对比: picture_predict/Group_Comparison.tif")