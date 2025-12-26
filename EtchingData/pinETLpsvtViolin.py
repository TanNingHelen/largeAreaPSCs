import os
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib import rcParams

# 配置设置
warnings.filterwarnings('ignore')
# 设置字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
rcParams.update({'font.size': 16})  # 加大全局字体大小到16（参照第二段代码）

# 创建输出目录
os.makedirs("picture", exist_ok=True)

# 颜色配置（参照第二段代码的颜色）
positive_color = '#a94837'  # 红色，积极色（参照第二段代码）
negative_color = '#4d8f74'  # 绿色，消极色（参照第二段代码）


def load_label_mappings():
    """加载标签映射关系"""
    mapping_file = "label_mappings/full_mapping_summary.csv"
    if os.path.exists(mapping_file):
        mappings = pd.read_csv(mapping_file)
        label_dict = {}
        for feat in mappings['Feature'].unique():
            sub_df = mappings[mappings['Feature'] == feat]
            label_dict[feat] = dict(zip(sub_df['Encoded'].astype(str), sub_df['Original']))
        return label_dict
    return {}


def load_data_and_single_model():
    """加载数据和单个CatBoost模型"""
    df = pd.read_excel("FinalData.xlsx")
    print(f"数据总行数: {len(df)}")

    # 使用全部数据进行分析
    y = df['PCE']
    X = df.drop(['PCE'], axis=1)

    # 加载单个CatBoost模型
    print("加载CatBoost模型...")
    try:
        model = CatBoostRegressor()
        model.load_model('models/best_catboost_model.cbm')
        print("✅ CatBoost模型加载成功")
    except Exception as e:
        print(f"❌ CatBoost模型加载失败: {e}")
        exit(1)

    return X, y, model


def calculate_single_model_shap_values(X, model):
    """计算单个模型的SHAP值"""

    print("计算CatBoost模型的SHAP值...")
    try:
        # 对于CatBoost，使用Pool
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(Pool(X))
        print("CatBoost SHAP值计算完成")
        return shap_values
    except Exception as e:
        print(f"❌ CatBoost SHAP值计算失败: {e}")
        raise Exception("SHAP值计算失败")


def plot_etl_passivator_shap_beeswarm(X, y, shap_values, label_mappings):
    """绘制ETL_Passivator特征的典型SHAP beeswarm图"""

    # 获取ETL_Passivator特征的索引和SHAP值
    etl_passivator_index = X.columns.get_loc('ETL_Passivator')
    etl_passivator_shap_values = shap_values[:, etl_passivator_index]
    etl_passivator_feature_values = X['ETL_Passivator']

    # 获取ETL_Passivator的映射关系
    etl_passivator_mapping = label_mappings.get('ETL_Passivator', {})

    # 统计每个取值的频率，并按频率排序
    value_counts = etl_passivator_feature_values.value_counts()

    # 排除取值为0的点（不再排除1）
    if 0 in value_counts.index:
        value_counts = value_counts.drop(0)

    print(f"ETL_Passivator特征有 {len(value_counts)} 个不同取值:")

    # 选择前13个最常见的取值，剩下的合并为others
    top_values = value_counts.head(13)
    other_values = value_counts.tail(max(0, len(value_counts) - 13))

    # 准备绘图数据和计算SHAP均值
    plot_data = []
    shap_means = {}

    # 处理前13个取值
    for value, count in top_values.items():
        # 获取映射后的标签
        value_str = str(value)
        display_value = etl_passivator_mapping.get(value_str, f"Value {value}")

        # 使用该特征取值的点
        used_mask = etl_passivator_feature_values == value
        used_shap_values = etl_passivator_shap_values[used_mask]

        # 计算该取值的SHAP均值
        shap_mean = np.mean(used_shap_values)
        shap_means[display_value] = shap_mean

        print(f"  {display_value}: {count} 个样本, SHAP均值: {shap_mean:.4f}")

        for shap_val in used_shap_values:
            plot_data.append({
                'ETL_Passivator Value': display_value,
                'SHAP Value': shap_val,
                'Count': count,
                'Type': 'Used'
            })

    # 从others中挑出value为27和28的数据
    special_values = [27, 28]
    special_data = {}

    for value in special_values:
        if value in other_values.index:
            # 获取映射后的标签
            value_str = str(value)
            display_value = etl_passivator_mapping.get(value_str, f"Value {value}")

            # 使用该特征取值的点
            used_mask = etl_passivator_feature_values == value
            used_shap_values = etl_passivator_shap_values[used_mask]

            # 计算该取值的SHAP均值
            shap_mean = np.mean(used_shap_values)
            special_data[display_value] = {
                'shap_values': used_shap_values,
                'count': other_values[value],
                'shap_mean': shap_mean
            }

            print(f"  {display_value}: {other_values[value]} 个样本, SHAP均值: {shap_mean:.4f} (从others中提取)")

            for shap_val in used_shap_values:
                plot_data.append({
                    'ETL_Passivator Value': display_value,
                    'SHAP Value': shap_val,
                    'Count': other_values[value],
                    'Type': 'Used'
                })

            # 从other_values中移除这些特殊值
            other_values = other_values.drop(value)

    # 将特殊值添加到shap_means中
    for display_value, data in special_data.items():
        shap_means[display_value] = data['shap_mean']

    # 处理剩余的others（不包含27和28）
    if len(other_values) > 0:
        others_count = other_values.sum()
        others_mask = etl_passivator_feature_values.isin(other_values.index)
        others_shap_values = etl_passivator_shap_values[others_mask]

        # 计算others的SHAP均值
        others_shap_mean = np.mean(others_shap_values)
        shap_means['others'] = others_shap_mean

        print(
            f"  others: {others_count} 个样本 (包含 {len(other_values)} 个取值, 不含27和28), SHAP均值: {others_shap_mean:.4f}")

        for shap_val in others_shap_values:
            plot_data.append({
                'ETL_Passivator Value': 'others',
                'SHAP Value': shap_val,
                'Count': others_count,
                'Type': 'Used'
            })

    plot_df = pd.DataFrame(plot_data)

    if plot_df.empty:
        print("没有足够的数据绘制ETL_Passivator特征的beeswarm图")
        return

    # 按照SHAP均值绝对值大小排序（从大到小）
    sorted_categories = sorted(shap_means.keys(),
                               key=lambda x: abs(shap_means[x]),
                               reverse=True)

    # 创建排序后的DataFrame
    plot_df_sorted = plot_df.copy()
    plot_df_sorted['ETL_Passivator Value'] = pd.Categorical(
        plot_df_sorted['ETL_Passivator Value'],
        categories=sorted_categories,
        ordered=True
    )
    plot_df_sorted = plot_df_sorted.sort_values('ETL_Passivator Value')

    # 为每个取值添加标签（只显示特征值和SHAP均值，不显示样本数量）
    label_map = {}
    for value in plot_df_sorted['ETL_Passivator Value'].unique():
        shap_mean = shap_means[value]
        label_map[value] = f"{value} (mean={shap_mean:.3f})"

    plot_df_sorted['ETL_Passivator Value with Stats'] = plot_df_sorted['ETL_Passivator Value'].map(label_map)

    # 设置图形尺寸（参照第二段代码）
    fig_width_inch = 16  # 增加图形宽度（从14增加到16）
    fig_height_inch = 12  # 增加图形高度（从10增加到12）

    # 创建图形
    plt.figure(figsize=(fig_width_inch, fig_height_inch), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')

    # 使用更直接的方法绘制beeswarm图
    # 为每个特征值创建数据点
    y_positions = {}
    # 注意：Matplotlib的y轴是从下到上递增，所以我们要反转顺序
    # 这样SHAP均值绝对值最大的在顶部
    for i, feature in enumerate(sorted_categories):
        y_positions[feature] = len(sorted_categories) - 1 - i

    # 绘制每个特征值的点
    for feature in sorted_categories:
        feature_data = plot_df_sorted[plot_df_sorted['ETL_Passivator Value'] == feature]
        if len(feature_data) > 0:
            y_pos = y_positions[feature]

            # 使用简单的抖动来避免重叠
            jitter = np.random.normal(0, 0.1, len(feature_data))
            y_jittered = y_pos + jitter

            # 根据SHAP值正负设置颜色
            colors = [positive_color if x > 0 else negative_color for x in feature_data['SHAP Value']]

            # 修改：增大点的大小到200（参照第二段代码）
            ax.scatter(feature_data['SHAP Value'], y_jittered,
                       c=colors, s=200, alpha=0.7, edgecolors='white', linewidth=1.0)

    # 设置y轴标签
    # 注意：我们需要按照y_positions的值排序标签
    sorted_y_positions = sorted(y_positions.items(), key=lambda x: x[1])
    y_ticks = [pos for _, pos in sorted_y_positions]
    y_labels = [label_map[feature] for feature, _ in sorted_y_positions]

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    # 添加零线（参照第二段代码的线宽）
    plt.axvline(0, color='black', linestyle='-', linewidth=2.0, alpha=0.8)

    # 设置标签 - 加大字体大小（参照第二段代码）
    plt.xlabel("SHAP Value (Impact on PCE)", fontname='Times New Roman', fontsize=18, fontweight='bold')
    plt.ylabel("ETL Passivator Values", fontname='Times New Roman', fontsize=18, fontweight='bold')

    # 设置坐标轴边框粗细（参照第二段代码）
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # 修改：设置刻度线方向向内，并加大宽度（参照第二段代码）
    ax.tick_params(axis='both', which='major', width=1.5, length=8, direction='in')
    ax.tick_params(axis='both', which='minor', width=1.0, length=5, direction='in')

    # 修改：加大刻度标签字体大小（参照第二段代码）
    ax.tick_params(axis='both', labelsize=16)  # 加大刻度标签字体
    for label in ax.get_xticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(16)  # 加大x轴刻度字体
    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(15)  # y轴字体稍小一点

    # 添加图例（参照第二段代码的样式和位置）
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=positive_color,
               markersize=18, label='Positive Impact', linestyle='None'),  # 加大图例标记大小
        Line2D([0], [0], marker='o', color='w', markerfacecolor=negative_color,
               markersize=18, label='Negative Impact', linestyle='None')  # 加大图例标记大小
    ]

    # 修改：将图例放在右下角，使用loc='lower right'，去掉图例边框（参照第二段代码）
    ax.legend(handles=legend_elements, loc='lower right',
              prop={'family': 'Times New Roman', 'size': 16},  # 加大图例字体
              facecolor='white', frameon=False)  # 去掉图例边框

    # 添加网格线以便更好地阅读数值（参照第二段代码的线宽）
    ax.grid(axis='x', linestyle='--', alpha=0.3, linewidth=1.0)

    # 调整布局（参照第二段代码的内边距）
    plt.tight_layout(pad=3.5)  # 增加内边距

    # 保存为高分辨率TIFF格式
    plt.savefig("picture/ETL_Passivator_SHAP_Beeswarm_CatBoost.tif", dpi=600, bbox_inches='tight',
                pad_inches=0.5, format='tiff', facecolor='white')
    plt.close()

    print("成功保存ETL_Passivator特征SHAP Beeswarm图 (CatBoost模型): picture/ETL_Passivator_SHAP_Beeswarm_CatBoost.tif")

    # 输出SHAP均值表格 - 包含带符号的SHAP均值
    print("\nETL_Passivator特征取值SHAP均值统计:")
    print("=" * 70)
    print(f"{'特征取值':<25} {'样本数量':<10} {'SHAP均值':<15} {'SHAP均值绝对值':<15}")
    print("-" * 70)

    for category in sorted_categories:
        count = plot_df_sorted[plot_df_sorted['ETL_Passivator Value'] == category]['Count'].iloc[0]
        mean_val = shap_means[category]
        abs_mean = abs(mean_val)
        # 保留符号的SHAP均值
        print(f"{category:<25} {count:<10} {mean_val:<15.4f} {abs_mean:<15.4f}")

    print("=" * 70)

    return plot_df_sorted, shap_means


if __name__ == "__main__":
    # 加载数据与单个模型
    X, y, model = load_data_and_single_model()
    label_mappings = load_label_mappings()

    # 检查ETL_Passivator特征是否存在
    if 'ETL_Passivator' not in X.columns:
        print("错误: 数据中未找到ETL_Passivator特征")
        print(f"可用特征: {X.columns.tolist()}")
    else:
        print(f"\n找到ETL_Passivator特征")

        # 计算单个模型的SHAP值
        print("计算CatBoost模型SHAP值...")
        shap_values = calculate_single_model_shap_values(X, model)

        # 绘制ETL_Passivator特征的典型SHAP beeswarm图
        print("正在生成ETL_Passivator特征SHAP Beeswarm图...")
        plot_df, shap_means = plot_etl_passivator_shap_beeswarm(X, y, shap_values, label_mappings)

        print("\nCatBoost模型分析完成！")
        print("生成的图片文件:")
        print("- picture/ETL_Passivator_SHAP_Beeswarm_CatBoost.tif (CatBoost模型SHAP beeswarm图)")