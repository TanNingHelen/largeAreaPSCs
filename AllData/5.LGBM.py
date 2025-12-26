import numpy as np
import os
import joblib
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib

# 忽略所有警告
import warnings

warnings.filterwarnings('ignore')

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from letsplotT import myscatterplot

# 创建目录结构
os.makedirs("picture", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("img", exist_ok=True)

# 加载数据并处理列名中的空格
df = pd.read_excel("FinalDataAll.xlsx")
df.columns = [c.replace(' ', '_') for c in df.columns]  # 将空格替换为下划线
y = df['PCE']
X = df.drop(['PCE'], axis=1)  # 假设所有特征都是连续变量

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# 模型路径
MODEL_PATH = "models/best_lgbm_model.pkl"
grid_search = None  # 用于保存网格搜索对象

if os.path.exists(MODEL_PATH):
    print("Loading pre-trained LGBM model...")
    model = joblib.load(MODEL_PATH)
else:
    print("Training new LGBM model...")

    # 调整参数网格使总运行次数在500次以内 (10折交叉验证)
    # 目标组合数：50以内 (50*10=500)
    param_grid = {
        'num_leaves': [31, 50],  # 2个选项
        'max_depth': [6, 10],  # 2个选项
        'learning_rate': [0.05, 0.1],  # 2个选项
        'n_estimators': [200, 300],  # 2个选项
        'min_child_samples': [10, 20],  # 2个选项
        'subsample': [0.8, 1.0]  # 2个选项
    }

    # 计算总参数组合数和CV运行次数
    total_combinations = 1
    for key in param_grid:
        total_combinations *= len(param_grid[key])
    total_cv_runs = total_combinations * 10  # 使用10折交叉验证

    print(f"Total parameter combinations: {total_combinations}")
    print(f"Total CV runs (10-fold): {total_cv_runs}")

    lgb_estimator = lgb.LGBMRegressor(random_state=42, silent=True)

    # 使用10折交叉验证
    grid_search = GridSearchCV(
        estimator=lgb_estimator,
        param_grid=param_grid,
        cv=10,  # 使用10折交叉验证
        scoring='r2',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)

    # 保存最佳模型
    model = grid_search.best_estimator_
    joblib.dump(model, MODEL_PATH)

    print("\n=== Best Parameters ===")
    print(grid_search.best_params_)
    print(f"Best CV R²: {grid_search.best_score_:.4f}")

# 预测
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


# ========== 评估指标计算 ==========
def calculate_metrics(y_true, y_pred):
    """计算所有评估指标"""
    r = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r, r2, mae, rmse


# 计算训练集指标
train_r, train_r2, train_mae, train_rmse = calculate_metrics(y_train, y_train_pred)

# 计算测试集指标
test_r, test_r2, test_mae, test_rmse = calculate_metrics(y_test, y_test_pred)

# 打印训练集指标
print("\n=== Training Set Metrics ===")
print(f"R: {train_r:.4f}")
print(f"R²: {train_r2:.4f}")
print(f"MAE: {train_mae:.4f}")
print(f"RMSE: {train_rmse:.4f}")

# 打印测试集指标
print("\n=== Test Set Metrics ===")
print(f"R: {test_r:.4f}")
print(f"R²: {test_r2:.4f}")
print(f"MAE: {test_mae:.4f}")
print(f"RMSE: {test_rmse:.4f}")

# 使用 myscatterplot 绘图
try:
    myscatterplot(
        y_train.values,
        y_train_pred,
        y_test.values,
        y_test_pred,
        modelname="LightGBM",
        target="PCE",
        plot_height=8,
        savepic=True,
        picname='LGBM_PCE_prediction'
    )
    print("Plot saved to: img/LGBM_PCE_prediction.png")
except Exception as e:
    print(f"Error using myscatterplot: {str(e)}")
    # 备用绘图方案
    plt.figure(figsize=(10, 8))
    plt.scatter(y_train, y_train_pred, color='#E48963', s=80, alpha=0.7,
                edgecolor='k', linewidth=0.5, label='Train')
    plt.scatter(y_test, y_test_pred, color='#1458C4', s=80, alpha=0.7,
                edgecolor='k', linewidth=0.5, marker='D', label='Test')
    max_val = max(y_train.max(), y_test.max())
    min_val = min(y_train.min(), y_test.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5)

    plt.xlabel('Actual PCE (%)', fontsize=14)
    plt.ylabel('Predicted PCE (%)', fontsize=14)
    plt.title('LightGBM: Actual vs Predicted PCE', fontsize=16)
    plt.legend(fontsize=12)

    # 添加统计信息
    plt.text(0.05, 0.9, f'Train R² = {train_r2:.3f}',
             transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.05, 0.85, f'Test R² = {test_r2:.3f}',
             transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.05, 0.8, f'Test MAE = {test_mae:.3f}',
             transform=plt.gca().transAxes, fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig("picture/LGBM_PCE_prediction.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Custom plot saved to: picture/LGBM_PCE_prediction.png")

print("\n" + "=" * 60)
print("LIGHTGBM MODEL DETAILED PARAMETERS")
print("=" * 60)

# 获取模型的所有参数
model_params = model.get_params()

# 分类显示参数
print("\n=== 核心超参数 ===")
core_params = ['num_leaves', 'max_depth', 'learning_rate', 'n_estimators',
               'min_child_samples', 'subsample', 'colsample_bytree']
for param in core_params:
    if param in model_params:
        print(f"{param}: {model_params[param]}")

print("\n=== 训练控制参数 ===")
training_params = ['random_state', 'boosting_type', 'objective', 'metric',
                   'n_jobs', 'silent', 'verbosity']
for param in training_params:
    if param in model_params:
        print(f"{param}: {model_params[param]}")

print("\n=== 正则化参数 ===")
regularization_params = ['reg_alpha', 'reg_lambda', 'min_split_gain', 'min_child_weight']
for param in regularization_params:
    if param in model_params:
        print(f"{param}: {model_params[param]}")

print("\n=== 其他参数 ===")
other_params = ['importance_type', 'class_weight', 'subsample_for_bin', 'min_child_samples']
for param in other_params:
    if param in model_params and param not in core_params + training_params + regularization_params:
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

# 显示模型统计信息
print("\n=== 模型统计信息 ===")
print(f"模型保存路径: {MODEL_PATH}")
print(f"提升迭代次数 (n_estimators): {model.n_estimators}")
print(f"模型是否已训练: {'是' if hasattr(model, '_Booster') else '否'}")

if hasattr(model, 'n_features_in_'):
    print(f"输入特征数量: {model.n_features_in_}")
    print(f"特征名称: {list(X.columns)}")
else:
    print(f"特征数量: {X.shape[1]}")
    print(f"特征名称: {list(X.columns)}")


