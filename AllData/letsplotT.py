import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os


def calc_corr(a, b):
    a_avg = sum(a) / len(a)
    b_avg = sum(b) / len(b)
    cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])
    sq = math.sqrt(sum([(x - a_avg) ** 2 for x in a]) * sum([(x - b_avg) ** 2 for x in b]))
    corr_factor = cov_ab / sq
    return corr_factor




#用来绘制带隙预测
def letsplot(train, trainpre, test, testpre, modelname='RF', target='Bandgap', save_fig=True, save_path=None):
    fontsize = 12
    plt.figure(figsize=(3, 3))
    plt.style.use('default')
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.rcParams['font.family'] = "Times New Roman"

    # 计算R2值
    train_r2 = metrics.r2_score(train, trainpre)
    test_r2 = metrics.r2_score(test, testpre)

    a = plt.scatter(train, trainpre, s=25, c="#4d8f74")  # 修改训练集颜色,绿色
    plt.plot([train.min(), train.max()], [train.min(), train.max()], 'k:', lw=1.5)
    plt.xlabel('Actual Bandgap (eV)', fontsize=fontsize)
    plt.ylabel('Predicted Bandgap (eV)', fontsize=fontsize)
    plt.tick_params(direction='in')
    # plt.title('{} model for {} prediction'.format(modelname, target), fontsize=fontsize)

    b = plt.scatter(test, testpre, s=25, c="#a94837", marker='D')  # 修改测试集颜色,红色

    # 去掉图例边框
    plt.legend((a, b), ('Train', 'Test'), fontsize=fontsize - 1,
               handletextpad=0.1, borderpad=0.1, frameon=False)

    # 设置x轴和y轴以0.5为刻度
    # 首先确定合适的刻度范围
    all_values = np.concatenate([train, trainpre, test, testpre])
    min_val = np.floor(all_values.min() * 2) / 2  # 向下取整到0.5的倍数
    max_val = np.ceil(all_values.max() * 2) / 2  # 向上取整到0.5的倍数

    # 生成0.5为步长的刻度
    ticks = np.arange(min_val, max_val + 0.5, 0.5)
    plt.xticks(ticks)
    plt.yticks(ticks)

    # 确保x轴和y轴范围一致
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.tight_layout()
    # 保存图片部分
    if save_fig:
        if save_path is None:
            # 如果没有指定路径，使用默认文件名
            save_path = f"{modelname}_{target}_prediction.tif"

        # 确保目录存在
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

        # 保存为高清TIFF格式，600 DPI
        plt.savefig(save_path, dpi=600, format='tiff', bbox_inches='tight')
        print(f"✅ 图片已保存为: {save_path}")

    plt.show()

    print('Train r:', calc_corr(train, trainpre))
    print('Train R2:', train_r2)
    print('Train RMSE:', np.sqrt(metrics.mean_squared_error(train, trainpre)))
    print('--------------------------------------')
    print('Test r:', calc_corr(test, testpre))
    print('Test R2:', test_r2)
    print('Test RMSE:', np.sqrt(metrics.mean_squared_error(test, testpre)))


# Discovering Outliers Based on Model Predictive Functions
def find_outliers(model, X, y, sigma=2):
    try:
        y_pred = pd.Series(model.predict(X), index=y.index)
    except:
        model.fit(X, y)
        y_pred = pd.Series(model.predict(X), index=y.index)

    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()

    # Calculate the parameter z defined by the outlier, where data with | z | greater than σ will be considered abnormal
    z = (resid - mean_resid) / std_resid
    outliers = z[abs(z) > sigma].index

    print('R2 = ', model.score(X, y))
    print('MSE = ', mean_squared_error(y, y_pred))
    print('------------------------------------------')
    print('mean of residuals', mean_resid)
    print('std of residuals', std_resid)
    print('------------------------------------------')
    print(f'find {len(outliers)}', 'outliers： ')
    print(outliers.tolist())

    plt.figure(figsize=(15, 5))

    ax_131 = plt.subplot(1, 3, 1)
    plt.plot(y, y_pred, '.')
    plt.plot(y.loc[outliers], y_pred.loc[outliers], 'ro')
    plt.legend(['Accepted', 'Outliers'])
    plt.xlabel('y')
    plt.ylabel('y_pred');

    ax_132 = plt.subplot(1, 3, 2)
    plt.plot(y, y - y_pred, '.')
    plt.plot(y.loc[outliers], y.loc[outliers] - y_pred.loc[outliers], 'ro')
    plt.legend(['Accepted', 'Outliers'])
    plt.xlabel('y')
    plt.ylabel('y - y_pred');

    ax_133 = plt.subplot(1, 3, 3)
    z.plot.hist(bins=50, ax=ax_133)
    z.loc[outliers].plot.hist(color='r', bins=50, ax=ax_133)
    plt.legend(['Accepted', 'Outliers'])
    plt.xlabel('z')

    return outliers


def getdata(y_train, y_train_hat, y_test, y_test_hat):
    """获取用于绘图的数据"""
    train_df = pd.DataFrame({
        'Actual PCE (%)': y_train,
        'Predicted PCE (%)': y_train_hat,
        'type': 'Train'
    })

    test_df = pd.DataFrame({
        'Actual PCE (%)': y_test,
        'Predicted PCE (%)': y_test_hat,
        'type': 'Test'
    })

    return pd.concat([train_df, test_df], ignore_index=True)


def myscatterplot(y_train, y_train_hat, y_test, y_test_hat, modelname="ML", target="PCE", plot_height=8, savepic=False,
                  picname='picname'):
    """绘制散点图，R和R²使用斜体显示"""

    # 直接使用数据，不调用外部函数
    train_actual = y_train.flatten() if hasattr(y_train, 'flatten') else y_train
    train_pred = y_train_hat.flatten() if hasattr(y_train_hat, 'flatten') else y_train_hat
    test_actual = y_test.flatten() if hasattr(y_test, 'flatten') else y_test
    test_pred = y_test_hat.flatten() if hasattr(y_test_hat, 'flatten') else y_test_hat

    # 创建DataFrame
    train_df = pd.DataFrame({
        'Actual': train_actual,
        'Predicted': train_pred,
        'Dataset': 'Train'
    })

    test_df = pd.DataFrame({
        'Actual': test_actual,
        'Predicted': test_pred,
        'Dataset': 'Test'
    })

    data = pd.concat([train_df, test_df], ignore_index=True)

    plot_aspect = 1.2
    plot_palette = ["#4d8f74", "#a94837"]  # 绿色和红色
    face_color = "white"
    spine_color = "white"
    label_size = 15

    fig, ax = plt.subplots(figsize=(plot_height, plot_height), dpi=300)

    # 绘制散点图
    sns.scatterplot(x='Actual', y='Predicted', hue='Dataset', data=data, s=90, alpha=.65,
                    edgecolor='black', palette=plot_palette, ax=ax)

    # 设置图表属性
    ax.set_facecolor(face_color)
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_color(spine_color)
    ax.tick_params(labelsize=label_size, direction='in')

    ax.grid(which='major', ls='--', c='k', alpha=0.6)
    ax.set_xlim(left=-0.1, right=26)
    ax.set_ylim(bottom=-0.1, top=26)

    ax.set_title(f"{modelname} for {target} prediction",
                 fontdict={"size": 23, "color": "k", 'family': 'Times New Roman'})

    # 修复：使用正确的大写PCE，不调用capitalize()
    ax.set_xlabel('Actual PCE (%)', fontdict={'fontsize': 25, 'family': 'Times New Roman'})
    ax.set_ylabel('Predicted PCE (%)', fontdict={'fontsize': 25, 'family': 'Times New Roman'})

    # 添加对角线
    ax.plot([-0.5, 25.5], [-0.5, 25.5], linestyle='--', color='gray', linewidth=2)

    # 添加图例
    plt.legend(loc='upper left', fontsize=16)

    # 计算和显示指标
    train_r2 = r2_score(train_actual, train_pred)
    test_corr = np.corrcoef(test_actual, test_pred)[0, 1]
    test_r2 = r2_score(test_actual, test_pred)
    test_rmse = np.sqrt(mean_squared_error(test_actual, test_pred))
    train_mae = mean_absolute_error(train_actual, train_pred)
    test_mae = mean_absolute_error(test_actual, test_pred)

    # 使用mathtext格式显示斜体R和R²
    # 注意：R用$R$表示斜体，R²用$R^2$表示
    train_text1 = 'Train $R^2$: {:.4f}'.format(train_r2)  # R²斜体
    test_text1 = 'Test $R$: {:.4f}'.format(test_corr)  # R斜体
    test_text2 = 'Test $R^2$: {:.4f}'.format(test_r2)  # R²斜体
    test_rmse_text = 'Test RMSE: {:.3f}'.format(test_rmse)

    # 使用MathText渲染，确保数学符号正确显示
    ax.text(0.67, 0.25, train_text1, transform=ax.transAxes, fontsize=15, va='top', ha='left',
            fontfamily='Times New Roman')
    ax.text(0.67, 0.19, test_text1, transform=ax.transAxes, fontsize=15, va='top', ha='left',
            fontfamily='Times New Roman')
    ax.text(0.67, 0.13, test_text2, transform=ax.transAxes, fontsize=15, va='top', ha='left',
            fontfamily='Times New Roman')
    ax.text(0.67, 0.07, test_rmse_text, transform=ax.transAxes, fontsize=15, va='top', ha='left',
            fontfamily='Times New Roman')

    # 保存图片
    if savepic:
        os.makedirs('./img', exist_ok=True)

        # 保存PNG格式（保持原有功能）
        plt.savefig(f'./img/{picname}.png', bbox_inches='tight', dpi=300, transparent=True)
        print(f"✅ PNG图片已保存到: ./img/{picname}.png")

        # 新增：同时保存TIFF格式
        plt.savefig(f'./img/{picname}.tiff', bbox_inches='tight', dpi=300, format='tiff',
                    facecolor='white', edgecolor='none')
        print(f"✅ TIFF图片已保存到: ./img/{picname}.tiff")
    # 打印指标
    # print(f"\n📊 模型性能指标:")
    # print(f"Train R²: {train_r2:.4f}")
    # print(f"Train MAE: {train_mae:.4f}")
    # print(f"Train RMSE: {np.sqrt(mean_squared_error(train_actual, train_pred)):.4f}")
    # print(f"Test R: {test_corr:.4f}")
    # print(f"Test R²: {test_r2:.4f}")
    # print(f"Test MAE: {test_mae:.4f}")
    # print(f"Test RMSE: {test_rmse:.4f}")
    plt.show()


# 如果你需要完全兼容原来的调用方式，但使用新的绘图函数，这里是一个增强版本
def myscatterplot_enhanced(y_train, y_train_hat, y_test, y_test_hat, modelname="ML", target="PCE", plot_height=8,
                           savepic=False,
                           picname='picname'):



    # 数据展平处理
    def flatten_data(data):
        if hasattr(data, 'flatten'):
            return data.flatten()
        elif isinstance(data, pd.Series):
            return data.values
        else:
            return data

    y_train_flat = flatten_data(y_train)
    y_train_hat_flat = flatten_data(y_train_hat)
    y_test_flat = flatten_data(y_test)
    y_test_hat_flat = flatten_data(y_test_hat)

    # 创建数据框
    data = pd.DataFrame({
        'Actual PCE (%)': np.concatenate([y_train_flat, y_test_flat]),
        'Predicted PCE (%)': np.concatenate([y_train_hat_flat, y_test_hat_flat]),
        'type': ['Train'] * len(y_train_flat) + ['Test'] * len(y_test_flat)
    })

    print(f"📊 数据统计:")
    print(f"  训练集样本数: {len(y_train_flat)}")
    print(f"  测试集样本数: {len(y_test_flat)}")
    print(f"  实际PCE范围: [{data['Actual PCE (%)'].min():.2f}, {data['Actual PCE (%)'].max():.2f}]")
    print(f"  预测PCE范围: [{data['Predicted PCE (%)'].min():.2f}, {data['Predicted PCE (%)'].max():.2f}]")

    # 设置样式
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.6})
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'cm'  # 使用Computer Modern字体渲染数学符号

    # 创建图形
    fig, ax = plt.subplots(figsize=(plot_height, plot_height), dpi=300)

    # 绘制散点图
    sns.scatterplot(
        x='Actual PCE (%)',
        y='Predicted PCE (%)',
        hue='type',
        hue_order=['Train', 'Test'],
        data=data,
        s=90,
        alpha=0.65,
        edgecolor='black',
        palette=["#4d8f74", "#a94837"],
        ax=ax
    )

    # 设置坐标轴
    ax.set_xlabel('Actual PCE (%)', fontsize=25, fontname='Times New Roman')
    ax.set_ylabel('Predicted PCE (%)', fontsize=25, fontname='Times New Roman')
    ax.set_xlim(-0.5, 25.5)
    ax.set_ylim(-0.5, 25.5)

    # 设置刻度
    ax.tick_params(axis='both', which='major', labelsize=15, direction='in')

    # 设置标题
    ax.set_title(f"{modelname} for {target} Prediction",
                 fontsize=23, fontname='Times New Roman', pad=20)

    # 添加对角线
    ax.plot([-0.5, 25.5], [-0.5, 25.5], linestyle='--', color='gray', linewidth=2, alpha=0.8)

    # 计算指标
    train_r2 = r2_score(y_train_flat, y_train_hat_flat)
    test_r2 = r2_score(y_test_flat, y_test_hat_flat)
    test_corr = np.corrcoef(y_test_flat, y_test_hat_flat)[0, 1]
    test_rmse = np.sqrt(mean_squared_error(y_test_flat, y_test_hat_flat))

    # 使用MathText渲染斜体R和R²
    # 注意：$R$ 表示斜体R，$R^2$ 表示斜体R²
    train_text = f'Train $R^2$ = {train_r2:.4f}'
    test_r_text = f'Test $R$ = {test_corr:.4f}'
    test_r2_text = f'Test $R^2$ = {test_r2:.4f}'
    test_rmse_text = f'Test RMSE = {test_rmse:.3f}'

    # 添加指标文本
    ax.text(0.65, 0.25, train_text, transform=ax.transAxes, fontsize=15,
            va='top', ha='left', fontname='Times New Roman')
    ax.text(0.65, 0.20, test_r_text, transform=ax.transAxes, fontsize=15,
            va='top', ha='left', fontname='Times New Roman')
    ax.text(0.65, 0.15, test_r2_text, transform=ax.transAxes, fontsize=15,
            va='top', ha='left', fontname='Times New Roman')
    ax.text(0.65, 0.10, test_rmse_text, transform=ax.transAxes, fontsize=15,
            va='top', ha='left', fontname='Times New Roman')

    # 调整图例
    ax.legend(title='Dataset', title_fontsize=14, fontsize=13, loc='upper left')

    # 设置背景颜色
    ax.set_facecolor('white')

    # 调整边框
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1)

    # 保存图片
    if savepic:
        os.makedirs('./img', exist_ok=True)
        plt.savefig(f'./img/{picname}.png', bbox_inches='tight', dpi=300, facecolor='white', edgecolor='none')
        print(f"✅ 图片已保存到: ./img/{picname}.png")

    # 打印详细指标
    print(f"\n📈 详细性能指标:")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Train MAE: {mean_absolute_error(y_train_flat, y_train_hat_flat):.4f}")
    print(f"  Train RMSE: {np.sqrt(mean_squared_error(y_train_flat, y_train_hat_flat)):.4f}")
    print(f"  Test R: {test_corr:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Test MAE: {mean_absolute_error(y_test_flat, y_test_hat_flat):.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")

    plt.tight_layout()
    plt.show()

def save_plot_data(y_train, y_train_hat, y_test, y_test_hat, savename):
    data = {'y_train': y_train,
            'y_train_predict': y_train_hat,
            'y_test': y_test,
            'y_test_predict': y_test_hat}
    df = pd.DataFrame(data)
    df.to_csv('./img/{}.csv'.format(savename), index=False)


def save_arrays_with_nan(y_train, y_train_hat, y_test, y_test_hat, savename):
    max_length = max(len(y_train), len(y_train_hat), len(y_test), len(y_test_hat))

    filled_y_train = np.pad(y_train, (0, max_length - len(y_train)), mode='constant', constant_values=np.nan)
    filled_y_train_hat = np.pad(y_train_hat, (0, max_length - len(y_train_hat)), mode='constant',
                                constant_values=np.nan)
    filled_y_test = np.pad(y_test, (0, max_length - len(y_test)), mode='constant', constant_values=np.nan)
    filled_y_test_hat = np.pad(y_test_hat, (0, max_length - len(y_test_hat)), mode='constant', constant_values=np.nan)

    df = pd.DataFrame({'y_train': filled_y_train,
                       'y_train_predict': filled_y_train_hat,
                       'y_test': filled_y_test,
                       'y_test_predict': filled_y_test_hat})
    df.to_csv('./img/{}.csv'.format(savename), index=False)



