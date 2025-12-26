import os
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
rcParams.update({'font.size': 12})

# 创建输出目录
os.makedirs("picture", exist_ok=True)

# 颜色配置 - 使用参考代码的颜色方案
positive_color = '#fa541c'  # 橙色，积极色
negative_color = '#52c41a'  # 绿色，消极色


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


def load_data_and_model():
    """加载数据和模型"""
    df = pd.read_excel("FinalData.xlsx")
    print(f"数据总行数: {len(df)}")

    # 使用全部数据进行分析
    y = df['PCE']
    X = df.drop(['PCE'], axis=1)

    # 加载CatBoost模型
    print("加载CatBoost模型...")
    model = CatBoostRegressor()
    model.load_model("models/best_catboost_model.cbm")
    print("CatBoost模型加载成功!")

    return X, y, model


def plot_htl2_shap_violin(X, y, model, label_mappings):
    """绘制HTL-2特征各取值的SHAP小提琴图"""

    # 计算SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Pool(X))

    # 获取HTL-2特征的索引和SHAP值
    htl2_index = X.columns.get_loc('HTL-2')
    htl2_shap_values = shap_values[:, htl2_index]
    htl2_feature_values = X['HTL-2']

    # 获取HTL-2的映射关系
    htl2_mapping = label_mappings.get('HTL-2', {})

    # 准备绘图数据 - 排除取值为0的点
    plot_data = []
    unique_values = htl2_feature_values.unique()

    print(f"HTL-2特征有 {len(unique_values)} 个不同取值:")

    # 统计非零值的数量
    non_zero_count = 0

    for value in unique_values:
        mask = htl2_feature_values == value
        count = sum(mask)

        # 排除取值为0的点
        if count > 0 and value != 0:  # 确保有样本且不是0值
            # 获取映射后的标签
            value_str = str(value)
            display_value = htl2_mapping.get(value_str, f"Value {value}")

            print(f"  {display_value}: {count} 个样本")
            non_zero_count += count

            for shap_val in htl2_shap_values[mask]:
                plot_data.append({
                    'HTL-2 Value': display_value,
                    'SHAP Value': shap_val,
                    'Type': '提升PCE' if shap_val > 0 else '降低PCE'
                })

    print(f"非零取值总样本数: {non_zero_count}")

    plot_df = pd.DataFrame(plot_data)

    if plot_df.empty:
        print("没有足够的数据绘制HTL-2特征的小提琴图")
        return

    # 设置图形大小 - 根据非零取值数量调整高度
    fig_width_cm = 14
    fig_height_cm = max(8, non_zero_count * 0.5)  # 动态调整高度
    fig_width_inch = fig_width_cm / 2.54
    fig_height_inch = fig_height_cm / 2.54

    # 创建图形
    plt.figure(figsize=(fig_width_inch, fig_height_inch), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')

    # 绘制小提琴图 - 使用参考代码的风格
    violin = sns.violinplot(
        data=plot_df,
        x='SHAP Value',
        y='HTL-2 Value',
        hue='Type',
        split=True,
        palette={'提升PCE': positive_color, '降低PCE': negative_color},
        inner="quartile",
        bw_method=0.2,
        linewidth=0
    )

    # 设置线条样式（参考代码中的设置）
    for l in violin.lines:
        l.set_linestyle('solid')

    # 添加零线
    plt.axvline(0, color='black', linestyle='solid', linewidth=0.5)

    # 设置标签 - 使用英文
    plt.xlabel("SHAP Value (Impact on PCE)", fontname='Times New Roman', fontsize=12)
    plt.ylabel("HTL-2 Values", fontname='Times New Roman', fontsize=12)

    # 设置坐标轴粗细为0.5pt
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # 设置刻度线宽度为0.5pt
    ax.tick_params(axis='both', which='major', width=0.5)

    # 设置刻度标签字体为Times New Roman
    ax.tick_params(axis='both', labelsize=10)
    for label in ax.get_xticklabels():
        label.set_fontname('Times New Roman')
    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')

    # 设置图例
    plt.legend(title='Impact Direction', title_fontproperties={'family': 'Times New Roman'},
               prop={'family': 'Times New Roman'}, facecolor='white', frameon=False)

    # 调整布局
    plt.tight_layout()

    # 保存为TIFF格式
    plt.savefig("picture/HTL2_SHAP_Violin_no_zero.tif", dpi=300, bbox_inches='tight',
                pad_inches=0.1, format='tiff', facecolor='white')
    plt.close()

    print("成功保存HTL-2特征SHAP小提琴图 (排除零值): picture/HTL2_SHAP_Violin_no_zero.tif")

    # 同时创建一个点图版本，因为小提琴图在样本少时可能不够清晰
    create_dot_plot(plot_df, non_zero_count)

    return plot_df


def create_dot_plot(plot_df, non_zero_count):
    """创建点图版本，更清晰地显示少量数据点"""

    # 设置图形大小
    fig_width_cm = 14
    fig_height_cm = max(8, non_zero_count * 0.4)  # 动态调整高度
    fig_width_inch = fig_width_cm / 2.54
    fig_height_inch = fig_height_cm / 2.54

    # 创建图形
    plt.figure(figsize=(fig_width_inch, fig_height_inch), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')

    # 绘制点图 - 使用较大的点
    sns.stripplot(
        data=plot_df,
        x='SHAP Value',
        y='HTL-2 Value',
        hue='Type',
        palette={'提升PCE': positive_color, '降低PCE': negative_color},
        size=8,  # 增大点的大小
        alpha=0.7,
        jitter=True
    )

    # 添加零线
    plt.axvline(0, color='black', linestyle='solid', linewidth=0.5)

    # 设置标签
    plt.xlabel("SHAP Value (Impact on PCE)", fontname='Times New Roman', fontsize=12)
    plt.ylabel("HTL-2 Values", fontname='Times New Roman', fontsize=12)

    # 设置坐标轴粗细为0.5pt
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # 设置刻度线宽度为0.5pt
    ax.tick_params(axis='both', which='major', width=0.5)

    # 设置刻度标签字体为Times New Roman
    ax.tick_params(axis='both', labelsize=10)
    for label in ax.get_xticklabels():
        label.set_fontname('Times New Roman')
    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')

    # 设置图例
    plt.legend(title='Impact Direction', title_fontproperties={'family': 'Times New Roman'},
               prop={'family': 'Times New Roman'}, facecolor='white', frameon=False)

    # 调整布局
    plt.tight_layout()

    # 保存为TIFF格式
    plt.savefig("picture/HTL2_SHAP_DotPlot_no_zero.tif", dpi=300, bbox_inches='tight',
                pad_inches=0.1, format='tiff', facecolor='white')
    plt.close()

    print("成功保存HTL-2特征SHAP点图 (排除零值): picture/HTL2_SHAP_DotPlot_no_zero.tif")


if __name__ == "__main__":
    # 加载数据与模型
    X, y, model = load_data_and_model()
    label_mappings = load_label_mappings()

    # 检查HTL-2特征是否存在
    if 'HTL-2' not in X.columns:
        print("错误: 数据中未找到HTL-2特征")
        print(f"可用特征: {X.columns.tolist()}")
    else:
        print(f"\n找到HTL-2特征")

        # 绘制HTL-2特征的小提琴图
        print("正在生成HTL-2特征SHAP小提琴图...")
        plot_df = plot_htl2_shap_violin(X, y, model, label_mappings)

        print("\n分析完成！")
        print("生成的图片文件:")
        print("- picture/HTL2_SHAP_Violin_no_zero.tif")
        print("- picture/HTL2_SHAP_DotPlot_no_zero.tif")