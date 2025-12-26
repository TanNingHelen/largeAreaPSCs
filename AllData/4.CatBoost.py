import numpy as np
import os
import joblib
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from letsplotT import myscatterplot

# 创建必要的目录
os.makedirs("picture_predict", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("img", exist_ok=True)

# 加载数据
df1 = pd.read_excel(r"FinalDataAll.xlsx")
Y = df1['PCE']
X = df1.drop(['PCE'], axis=1)

# CatBoost不需要标准化，直接使用原始数据
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=12)

# 检查是否已有保存的模型
MODEL_PATH = "models/best_catboost_model.pkl"
grid_search = None  # 用于保存网格搜索对象

if os.path.exists(MODEL_PATH):
    print("Loading pre-trained CatBoost model...")
    best_cb = joblib.load(MODEL_PATH)
else:
    print("Training new CatBoost model...")

    # 定义CatBoost参数网格，总训练次数控制在500以内
    # 计算：参数组合数 × 10 ≤ 500 → 参数组合数 ≤ 50
    param_grid = {
        'iterations': [100, 200, 300],  # 3个选项
        'depth': [4, 6, 8],  # 3个选项
        'learning_rate': [0.01, 0.05, 0.1],  # 3个选项
        'l2_leaf_reg': [1, 3, 5],  # 3个选项
        'random_strength': [0.1, 1]  # 2个选项
    }

    # 总参数组合数: 3 × 3 × 3 × 3 × 2 = 162
    # 总训练次数: 162 × 10 = 1620 > 500

    # 需要减少参数组合
    param_grid = {
        'iterations': [100, 200],  # 2个选项
        'depth': [4, 6, 8],  # 3个选项
        'learning_rate': [0.01, 0.05, 0.1],  # 3个选项
        'l2_leaf_reg': [1, 3],  # 2个选项
        'random_strength': [1]  # 1个选项
    }

    # 总参数组合数: 2 × 3 × 3 × 2 × 1 = 36
    # 总训练次数: 36 × 10 = 360 < 500

    # 计算总参数组合数
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)
    total_training_runs = total_combinations * 10
    print(f"Total parameter combinations: {total_combinations}")
    print(f"Total training runs (with 10-fold CV): {total_training_runs}")

    # 创建基础CatBoost模型
    cb = CatBoostRegressor(
        random_seed=42,
        verbose=False,  # 减少训练时的输出
        thread_count=-1  # 使用所有CPU核心
    )

    # 10折交叉验证
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=cb,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        scoring='r2',
        verbose=2,
        return_train_score=True
    )

    print("开始CatBoost模型训练...")
    grid_search.fit(X_train, y_train)

    best_cb = grid_search.best_estimator_

    # 保存模型
    joblib.dump(best_cb, MODEL_PATH)

    print("\n=== Best Parameters ===")
    print(grid_search.best_params_)
    print(f"Best CV R²: {grid_search.best_score_:.4f}")

    # 输出交叉验证结果
    results_df = pd.DataFrame(grid_search.cv_results_)
    print("\nTop 5 parameter combinations:")
    top_5 = results_df.nlargest(5, 'mean_test_score')[
        ['mean_test_score', 'std_test_score', 'params']
    ]
    for i, row in top_5.iterrows():
        print(f"R²: {row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}")
        print(f"  Params: {row['params']}")

# 预测结果
y_train_pred = best_cb.predict(X_train)
y_test_pred = best_cb.predict(X_test)


# ========== 评估指标部分 ==========
def calculate_metrics(y_true, y_pred):
    r = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r, r2, mae, rmse


train_r, train_r2, train_mae, train_rmse = calculate_metrics(y_train, y_train_pred)
test_r, test_r2, test_mae, test_rmse = calculate_metrics(y_test, y_test_pred)

print("\n=== Training Set Metrics ===")
print(f"R: {train_r:.4f}")
print(f"R²: {train_r2:.4f}")
print(f"MAE: {train_mae:.4f}")
print(f"RMSE: {train_rmse:.4f}")

print("\n=== Test Set Metrics ===")
print(f"R: {test_r:.4f}")
print(f"R²: {test_r2:.4f}")
print(f"MAE: {test_mae:.4f}")
print(f"RMSE: {test_rmse:.4f}")

# 使用myscatterplot绘图
try:
    myscatterplot(
        y_train.values,
        y_train_pred,
        y_test.values,
        y_test_pred,
        modelname="CatBoost",
        target="PCE",
        plot_height=8,
        savepic=True,
        picname='CatBoost_PCE_prediction'
    )
    print("Plot saved to: img/CatBoost_PCE_prediction.png")
except Exception as e:
    print(f"Error using myscatterplot: {str(e)}")
    # 备用绘图方案
    plt.figure(figsize=(10, 8))
    plt.scatter(y_train, y_train_pred, color='#E48963', s=80, alpha=0.7, edgecolor='k', linewidth=0.5, label='Train')
    plt.scatter(y_test, y_test_pred, color='#1458C4', s=80, alpha=0.7, edgecolor='k', linewidth=0.5, marker='D',
                label='Test')
    max_val = max(Y.max(), y_train_pred.max(), y_test_pred.max())
    min_val = min(Y.min(), y_train_pred.min(), y_test_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5)

    plt.xlabel('Actual PCE (%)', fontsize=14)
    plt.ylabel('Predicted PCE (%)', fontsize=14)
    plt.title('CatBoost: Actual vs Predicted PCE', fontsize=16)
    plt.legend(fontsize=12)

    plt.text(0.05, 0.9, f'Train R² = {train_r2:.3f}', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.05, 0.85, f'Test R² = {test_r2:.3f}', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.05, 0.8, f'Test MAE = {test_mae:.3f}', transform=plt.gca().transAxes, fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig("picture_predict/CatBoost_PCE_prediction.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Custom plot saved to: picture_predict/CatBoost_PCE_prediction.png")

# 输出最佳模型的详细参数
print("\n" + "=" * 60)
print("CATBOOST MODEL DETAILED PARAMETERS")
print("=" * 60)

# 获取模型的所有参数
model_params = best_cb.get_all_params()

# 分类显示参数
print("\n=== 核心超参数 ===")
core_params = ['iterations', 'depth', 'learning_rate', 'l2_leaf_reg',
               'random_strength', 'bagging_temperature', 'border_count']
for param in core_params:
    if param in model_params:
        print(f"{param}: {model_params[param]}")

print("\n=== 训练控制参数 ===")
training_params = ['random_seed', 'thread_count', 'verbose', 'task_type',
                   'loss_function', 'eval_metric', 'early_stopping_rounds']
for param in training_params:
    if param in model_params:
        print(f"{param}: {model_params[param]}")

print("\n=== 过拟合控制参数 ===")
overfit_params = ['rsm', 'od_type', 'od_pval', 'od_wait', 'max_ctr_complexity']
for param in overfit_params:
    if param in model_params:
        print(f"{param}: {model_params[param]}")

print("\n=== 特征处理参数 ===")
feature_params = ['cat_features', 'one_hot_max_size', 'feature_border_type', 'nan_mode']
for param in feature_params:
    if param in model_params:
        print(f"{param}: {model_params[param]}")

# 如果是网格搜索得到的模型，显示网格搜索信息
if grid_search is not None:
    print("\n=== 网格搜索信息 ===")
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证分数 (R²): {grid_search.best_score_:.4f}")
    print(f"搜索的参数组合总数: {len(grid_search.cv_results_['params'])}")

    # 显示前5个最佳参数组合
    print("\n=== 前5个最佳参数组合 ===")
    results_df = pd.DataFrame(grid_search.cv_results_)
    top_5 = results_df.nlargest(5, 'mean_test_score')
    for i, (_, row) in enumerate(top_5.iterrows()):
        print(f"Rank {i + 1}: R² = {row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}")
        print(f"  Parameters: {row['params']}")
else:
    print("\n=== 模型来源 ===")
    print("模型从文件加载，未进行新的网格搜索")




