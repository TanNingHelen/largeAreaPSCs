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
        # 尝试多种编码方式
        encodings_to_try = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252', 'gbk', 'gb2312', 'utf-8-sig']

        for encoding in encodings_to_try:
            try:
                mappings = pd.read_csv(mapping_file, encoding=encoding)
                print(f"✅ 使用 {encoding} 编码成功读取映射文件")

                label_dict = {}
                for feat in mappings['Feature'].unique():
                    sub_df = mappings[mappings['Feature'] == feat]
                    label_dict[feat] = dict(zip(sub_df['Encoded'].astype(str), sub_df['Original']))
                return label_dict

            except UnicodeDecodeError:
                print(f"⚠️  {encoding} 编码失败，尝试其他编码...")
                continue
            except Exception as e:
                print(f"⚠️  使用 {encoding} 编码读取时出错: {e}")
                continue

        # 如果所有编码都失败，尝试使用错误处理
        try:
            print("⚠️ 所有编码尝试失败，尝试使用错误处理模式...")
            mappings = pd.read_csv(mapping_file, encoding='utf-8', errors='replace')
            print("✅ 使用错误处理模式成功读取映射文件（无法解码的字符已替换）")

            label_dict = {}
            for feat in mappings['Feature'].unique():
                sub_df = mappings[mappings['Feature'] == feat]
                label_dict[feat] = dict(zip(sub_df['Encoded'].astype(str), sub_df['Original']))
            return label_dict

        except Exception as e:
            print(f"❌ 最终尝试读取映射文件失败: {e}")
            return {}

    print("⚠️ 未找到映射文件")
    return {}


def load_data_and_ensemble_models():
    """加载数据和集成模型"""
    df = pd.read_excel("FinalDataAll.xlsx")
    print(f"数据总行数: {len(df)}")

    # 使用全部数据进行分析
    y = df['PCE']
    X = df.drop(['PCE'], axis=1)

    # 模型权重配置（基于测试集R²）
    model_configs = {
        'rf': {'path': 'models/best_rf_model.pkl', 'r2': 0.6892},
        'xgb': {'path': 'models/best_xgb_model.pkl', 'r2': 0.7630},
        'catboost': {'path': 'models/best_catboost_model.pkl', 'r2': 0.6762},
        'lgbm': {'path': 'models/best_lgbm_model.pkl', 'r2': 0.7446}
    }

    # 尝试不同的CatBoost模型路径
    catboost_paths = [
        'models/best_catboost_model.pkl',
        'models/best_catboost_model.cbm'
    ]

    # 计算总R²用于权重归一化
    total_r2 = sum(config['r2'] for config in model_configs.values())

    # 加载模型并计算权重
    models = {}
    weights = {}
    successful_models = 0

    print("加载集成模型...")
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

    return X, y, models, weights


def calculate_ensemble_shap_values(X, models, weights):
    """计算集成模型的加权SHAP值"""

    # 初始化加权SHAP值
    weighted_shap_values = None
    total_weight = 0

    for model_name, model in models.items():
        try:
            print(f"计算 {model_name.upper()} 的SHAP值...")

            if model_name == 'catboost':
                # 对于CatBoost，使用Pool
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(Pool(X))
            else:
                # 对于其他模型
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)

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


def plot_htl_addictive_shap_beeswarm(X, y, ensemble_shap_values, label_mappings):
    """绘制HTL-Addictive特征的典型SHAP beeswarm图"""

    # 获取HTL-Addictive特征的索引和SHAP值
    htl_addictive_index = X.columns.get_loc('HTL-Addictive')
    htl_addictive_shap_values = ensemble_shap_values[:, htl_addictive_index]
    htl_addictive_feature_values = X['HTL-Addictive']

    # 获取HTL-Addictive的映射关系
    htl_addictive_mapping = label_mappings.get('HTL-Addictive', {})

    # 统计每个取值的频率，并按频率排序
    value_counts = htl_addictive_feature_values.value_counts()

    # 排除取值为0的点
    if 0 in value_counts.index:
        value_counts = value_counts.drop(0)

    print(f"HTL-Addictive特征有 {len(value_counts)} 个不同取值:")

    # 查找DMPU+PEAI对应的编码值
    dmpu_peai_encoded = None
    for encoded_val, display_name in htl_addictive_mapping.items():
        if display_name == "DMPU+PEAI":
            dmpu_peai_encoded = int(encoded_val)
            break

    # 选择前13个最常见的取值（为DMPU+PEAI留一个位置）
    top_values = value_counts.head(13)

    # 从剩余的值中提取DMPU+PEAI（如果存在）
    remaining_values = value_counts.tail(max(0, len(value_counts) - 13))

    # 强制提取DMPU+PEAI
    dmpu_peai_data = None
    if dmpu_peai_encoded is not None and dmpu_peai_encoded in remaining_values.index:
        dmpu_peai_count = remaining_values[dmpu_peai_encoded]
        dmpu_peai_data = {
            'value': dmpu_peai_encoded,
            'count': dmpu_peai_count,
            'display_name': 'DMPU+PEAI'
        }
        # 从remaining_values中移除DMPU+PEAI
        remaining_values = remaining_values.drop(dmpu_peai_encoded)
        print(f"  强制提取 DMPU+PEAI: {dmpu_peai_count} 个样本")
    elif dmpu_peai_encoded is not None and dmpu_peai_encoded in top_values.index:
        # 如果DMPU+PEAI已经在前13个中，也单独记录
        dmpu_peai_count = top_values[dmpu_peai_encoded]
        dmpu_peai_data = {
            'value': dmpu_peai_encoded,
            'count': dmpu_peai_count,
            'display_name': 'DMPU+PEAI'
        }
        print(f"  DMPU+PEAI已在前13个中: {dmpu_peai_count} 个样本")

    # 处理剩余的others（不包含DMPU+PEAI）
    other_values = remaining_values

    # 准备绘图数据和计算SHAP均值
    plot_data = []
    shap_means = {}

    # 处理前13个取值
    for value, count in top_values.items():
        # 获取映射后的标签
        value_str = str(value)
        display_value = htl_addictive_mapping.get(value_str, f"Value {value}")

        # 使用该特征取值的点
        used_mask = htl_addictive_feature_values == value
        used_shap_values = htl_addictive_shap_values[used_mask]

        # 计算该取值的SHAP均值
        shap_mean = np.mean(used_shap_values)
        shap_means[display_value] = shap_mean

        print(f"  {display_value}: {count} 个样本, SHAP均值: {shap_mean:.4f}")

        for shap_val in used_shap_values:
            plot_data.append({
                'HTL-Addictive Value': display_value,
                'SHAP Value': shap_val,
                'Count': count,
                'Type': 'Used'
            })

    # 处理DMPU+PEAI（如果存在）
    if dmpu_peai_data is not None:
        value = dmpu_peai_data['value']
        count = dmpu_peai_data['count']
        display_value = dmpu_peai_data['display_name']

        # 使用该特征取值的点
        used_mask = htl_addictive_feature_values == value
        used_shap_values = htl_addictive_shap_values[used_mask]

        # 计算该取值的SHAP均值
        shap_mean = np.mean(used_shap_values)
        shap_means[display_value] = shap_mean

        print(f"  {display_value}: {count} 个样本, SHAP均值: {shap_mean:.4f} (强制提取)")

        for shap_val in used_shap_values:
            plot_data.append({
                'HTL-Addictive Value': display_value,
                'SHAP Value': shap_val,
                'Count': count,
                'Type': 'Used'
            })

    # 处理others（第15个及之后的取值，不包含DMPU+PEAI）
    if len(other_values) > 0:
        others_count = other_values.sum()
        others_mask = htl_addictive_feature_values.isin(other_values.index)
        others_shap_values = htl_addictive_shap_values[others_mask]

        # 计算others的SHAP均值
        others_shap_mean = np.mean(others_shap_values)
        shap_means['others'] = others_shap_mean

        print(
            f"  others: {others_count} 个样本 (包含 {len(other_values)} 个取值, 不含DMPU+PEAI), SHAP均值: {others_shap_mean:.4f}")

        for shap_val in others_shap_values:
            plot_data.append({
                'HTL-Addictive Value': 'others',
                'SHAP Value': shap_val,
                'Count': others_count,
                'Type': 'Used'
            })

    plot_df = pd.DataFrame(plot_data)

    if plot_df.empty:
        print("没有足够的数据绘制HTL-Addictive特征的beeswarm图")
        return

    # 按照SHAP均值绝对值大小排序（从大到小）
    sorted_categories = sorted(shap_means.keys(),
                               key=lambda x: abs(shap_means[x]),
                               reverse=True)

    # 创建排序后的DataFrame
    plot_df_sorted = plot_df.copy()
    plot_df_sorted['HTL-Addictive Value'] = pd.Categorical(
        plot_df_sorted['HTL-Addictive Value'],
        categories=sorted_categories,
        ordered=True
    )
    plot_df_sorted = plot_df_sorted.sort_values('HTL-Addictive Value')

    # 为每个取值添加标签（只显示特征值和SHAP均值，不显示样本数量）
    label_map = {}
    for value in plot_df_sorted['HTL-Addictive Value'].unique():
        shap_mean = shap_means[value]
        label_map[value] = f"{value} (mean={shap_mean:.3f})"

    plot_df_sorted['HTL-Addictive Value with Stats'] = plot_df_sorted['HTL-Addictive Value'].map(label_map)

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
        feature_data = plot_df_sorted[plot_df_sorted['HTL-Addictive Value'] == feature]
        if len(feature_data) > 0:
            y_pos = y_positions[feature]

            # 使用简单的抖动来避免重叠
            jitter = np.random.normal(0, 0.1, len(feature_data))
            y_jittered = y_pos + jitter

            # 根据SHAP值正负设置颜色（使用第二段代码的颜色）
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
    plt.ylabel("HTL-Addictive Values", fontname='Times New Roman', fontsize=18, fontweight='bold')

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
    plt.savefig("picture/HTL_Addictive_SHAP_Beeswarm_Ensemble.tif", dpi=600, bbox_inches='tight',
                pad_inches=0.5, format='tiff', facecolor='white')
    plt.close()

    print("成功保存HTL-Addictive特征SHAP Beeswarm图 (集成模型): picture/HTL_Addictive_SHAP_Beeswarm_Ensemble.tif")

    # 输出SHAP均值表格 - 包含带符号的SHAP均值
    print("\nHTL-Addictive特征取值SHAP均值统计:")
    print("=" * 70)
    print(f"{'特征取值':<25} {'样本数量':<10} {'SHAP均值':<15} {'SHAP均值绝对值':<15}")
    print("-" * 70)

    for category in sorted_categories:
        count = plot_df_sorted[plot_df_sorted['HTL-Addictive Value'] == category]['Count'].iloc[0]
        mean_val = shap_means[category]
        abs_mean = abs(mean_val)
        # 保留符号的SHAP均值
        print(f"{category:<25} {count:<10} {mean_val:<15.4f} {abs_mean:<15.4f}")

    print("=" * 70)

    return plot_df_sorted, shap_means


if __name__ == "__main__":
    # 加载数据与集成模型
    X, y, models, weights = load_data_and_ensemble_models()
    label_mappings = load_label_mappings()

    # 检查HTL-Addictive特征是否存在
    if 'HTL-Addictive' not in X.columns:
        print("错误: 数据中未找到HTL-Addictive特征")
        print(f"可用特征: {X.columns.tolist()}")
    else:
        print(f"\n找到HTL-Addictive特征")

        # 计算集成模型的加权SHAP值
        print("计算集成模型SHAP值...")
        ensemble_shap_values = calculate_ensemble_shap_values(X, models, weights)

        # 绘制HTL-Addictive特征的典型SHAP beeswarm图
        print("正在生成HTL-Addictive特征SHAP Beeswarm图...")
        plot_df, shap_means = plot_htl_addictive_shap_beeswarm(X, y, ensemble_shap_values, label_mappings)

        print("\n集成模型分析完成！")
        print("生成的图片文件:")
        print("- picture/HTL_Addictive_SHAP_Beeswarm_Ensemble.tif (典型SHAP beeswarm图)")