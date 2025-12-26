import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
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
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=12)

# 检查是否已有保存的模型
MODEL_PATH = "models/best_rf_model.pkl"

if os.path.exists(MODEL_PATH):
    print("Loading pre-trained Random Forest model...")
    best_rf = joblib.load(MODEL_PATH)
    print("加载现有模型，无法显示网格搜索的最佳参数，但可以显示模型的实际参数")
else:
    print("Training new Random Forest model...")

    # 重新设计参数网格，保持10折交叉验证，总折数控制在1000以内
    # 计算：参数组合数 × 10 ≤ 1000 → 参数组合数 ≤ 100
    param_grid = {
        'n_estimators': [50, 100, 150],  # 3个选项
        'max_depth': [10, 20, 30, None],  # 4个选项
        'min_samples_split': [2, 5, 10],  # 3个选项
        'min_samples_leaf': [1, 2, 4],  # 3个选项
        'max_features': ['sqrt', 'log2']  # 2个选项
    }

    # 总参数组合数: 3 × 4 × 3 × 3 × 2 = 216
    # 总训练次数: 216 × 10 = 2160 > 1000

    # 需要进一步减少参数组合
    param_grid = {
        'n_estimators': [50, 100],  # 2个选项
        'max_depth': [10, 20, None],  # 3个选项
        'min_samples_split': [2, 5],  # 2个选项
        'min_samples_leaf': [1, 2],  # 2个选项
        'max_features': ['sqrt', 'log2']  # 2个选项
    }

    # 总参数组合数: 2 × 3 × 2 × 2 × 2 = 48
    # 总训练次数: 48 × 10 = 480 < 1000

    rf = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
        bootstrap=True,
        oob_score=True  # 启用袋外分数评估
    )

    # 保持10折交叉验证
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        scoring='r2',
        verbose=2
    )

    print("开始随机森林模型训练...")
    print(f"参数组合数: {2 * 3 * 2 * 2 * 2}")
    print(f"交叉验证折数: {cv.n_splits}")
    print(f"总训练次数: {2 * 3 * 2 * 2 * 2 * cv.n_splits}")

    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    joblib.dump(best_rf, MODEL_PATH)

    print("\n=== Best Parameters from Grid Search ===")
    print(grid_search.best_params_)
    print(f"Best CV R²: {grid_search.best_score_:.4f}")

    # 输出袋外分数
    if hasattr(best_rf, 'oob_score_'):
        print(f"OOB Score: {best_rf.oob_score_:.4f}")

# 预测结果
y_train_pred = best_rf.predict(X_train)
y_test_pred = best_rf.predict(X_test)


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

# ========== 评估指标部分结束 ==========

# 使用myscatterplot绘图
try:
    myscatterplot(
        y_train.values,
        y_train_pred,
        y_test.values,
        y_test_pred,
        modelname="Random Forest",
        target="PCE",
        plot_height=8,
        savepic=True,
        picname='RF_PCE_prediction'
    )
    print("Plot saved to: img/RF_PCE_prediction.png")
except Exception as e:
    print(f"Error using myscatterplot: {str(e)}")
    # 简化的备用绘图方案
    plt.figure(figsize=(10, 8))
    plt.scatter(y_train, y_train_pred, color='#E48963', s=80, alpha=0.7, edgecolor='k', linewidth=0.5, label='Train')
    plt.scatter(y_test, y_test_pred, color='#1458C4', s=80, alpha=0.7, edgecolor='k', linewidth=0.5, marker='D',
                label='Test')
    max_val = max(y_train.max(), y_test.max())
    min_val = min(y_train.min(), y_test.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5)

    plt.xlabel('Actual PCE (%)', fontsize=14)
    plt.ylabel('Predicted PCE (%)', fontsize=14)
    plt.title('Random Forest: Actual vs Predicted PCE', fontsize=16)
    plt.legend(fontsize=12)

    plt.text(0.05, 0.9, f'Train R² = {train_r2:.3f}', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.05, 0.85, f'Test R² = {test_r2:.3f}', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.05, 0.8, f'Test MAE = {test_mae:.3f}', transform=plt.gca().transAxes, fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig("picture_predict/RF_PCE_prediction.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Custom plot saved to: picture_predict/RF_PCE_prediction.png")

print("\n" + "=" * 60)
print("BEST MODEL PARAMETERS")
print("=" * 60)

# 获取模型的所有参数
model_params = best_rf.get_params()

# 分类显示参数
print("\n=== 核心参数 ===")
core_params = ['n_estimators', 'max_depth', 'min_samples_split',
               'min_samples_leaf', 'max_features', 'bootstrap']
for param in core_params:
    if param in model_params:
        print(f"{param}: {model_params[param]}")

print("\n=== 训练控制参数 ===")
training_params = ['random_state', 'n_jobs', 'verbose', 'warm_start']
for param in training_params:
    if param in model_params:
        print(f"{param}: {model_params[param]}")

print("\n=== 其他重要参数 ===")
other_params = ['criterion', 'max_leaf_nodes', 'min_impurity_decrease',
                'min_weight_fraction_leaf', 'oob_score', 'ccp_alpha']
for param in other_params:
    if param in model_params:
        print(f"{param}: {model_params[param]}")

# 显示树的统计信息
print("\n=== 模型统计信息 ===")
print(f"模型保存路径: {MODEL_PATH}")
print(f"森林中的树数量: {len(best_rf.estimators_)}")
print(f"特征数量: {best_rf.n_features_in_}")
if hasattr(best_rf, 'n_features_in_'):
    print(f"特征名称: {list(X.columns)}")




print("\n" + "=" * 60)