import numpy as np
import os
import joblib
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from letsplotT import myscatterplot
import traceback
import shutil
import sys


# 日志记录函数
def log_message(message, log_file="xgboost_training.log"):
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    with open(log_file, "a") as f:
        f.write(full_message + "\n")


# 创建必要的目录
os.makedirs("picture_predict", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("img", exist_ok=True)

# 模型路径
MODEL_PATH = "models/best_xgb_model.pkl"
BACKUP_MODEL_PATH = "models/backup_xgb_model.pkl"


def calculate_metrics(y_true, y_pred):
    """计算所有评估指标"""
    r = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r, r2, mae, rmse


def load_or_train_model():
    try:
        # 加载数据
        log_message("Loading data...")
        df1 = pd.read_excel(r"FinalDataAll.xlsx")
        Y = df1['PCE']
        X = df1.drop(['PCE'], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=12)

        # 尝试加载现有模型
        if os.path.exists(MODEL_PATH):
            log_message("Found saved model, attempting to load...")
            try:
                # 备份模型
                if os.path.exists(BACKUP_MODEL_PATH):
                    os.remove(BACKUP_MODEL_PATH)
                shutil.copy(MODEL_PATH, BACKUP_MODEL_PATH)

                best_xgb = joblib.load(MODEL_PATH)
                test_pred = best_xgb.predict(X_test.iloc[:1])
                log_message(f"Model validation passed. Test prediction sample: {test_pred[0]}")
                return best_xgb, X_train, X_test, y_train, y_test, None
            except Exception as e:
                log_message(f"Model validation failed: {str(e)}")
                if os.path.exists(BACKUP_MODEL_PATH):
                    best_xgb = joblib.load(BACKUP_MODEL_PATH)
                    log_message("Backup model loaded")
                    return best_xgb, X_train, X_test, y_train, y_test, None
                else:
                    log_message("No backup model found, proceeding with training...")

        # 训练新模型
        log_message("Training new XGBoost model...")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.05],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'gamma': [0, 0.1],
            'reg_alpha': [0, 0.1],
            'reg_lambda': [0, 0.1]
        }

        xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=10,
            n_jobs=-1,
            scoring='r2',
            verbose=2
        )

        log_message("Starting grid search...")
        grid_search.fit(X_train, y_train)

        best_xgb = grid_search.best_estimator_
        joblib.dump(best_xgb, MODEL_PATH)

        log_message(f"\n=== Grid Search Results ===")
        log_message(f"Best Parameters: {grid_search.best_params_}")
        log_message(f"Best CV R²: {grid_search.best_score_:.4f}")

        # 输出所有候选参数和分数
        log_message("\n=== All Candidate Parameters and Scores ===")
        results = pd.DataFrame(grid_search.cv_results_)
        log_message(f"Total parameter combinations tested: {len(results)}")

        # 按分数排序显示前5个
        top_results = results.sort_values('mean_test_score', ascending=False).head(5)
        for i, (_, row) in enumerate(top_results.iterrows()):
            log_message(f"Rank {i + 1}: params={row['params']}, score={row['mean_test_score']:.4f}")

        return best_xgb, X_train, X_test, y_train, y_test, grid_search

    except Exception as e:
        log_message(f"Error in load_or_train_model: {str(e)}\n{traceback.format_exc()}")
        raise


def evaluate_model(model, X_train, X_test, y_train, y_test):
    try:
        # 训练集预测和评估
        y_train_pred = model.predict(X_train)
        train_r, train_r2, train_mae, train_rmse = calculate_metrics(y_train, y_train_pred)

        # 测试集预测和评估
        y_test_pred = model.predict(X_test)
        test_r, test_r2, test_mae, test_rmse = calculate_metrics(y_test, y_test_pred)

        # 打印训练集指标
        log_message("\n=== Training Set Metrics ===")
        log_message(f"R: {train_r:.4f}")
        log_message(f"R²: {train_r2:.4f}")
        log_message(f"MAE: {train_mae:.4f}")
        log_message(f"RMSE: {train_rmse:.4f}")

        # 打印测试集指标
        log_message("\n=== Test Set Metrics ===")
        log_message(f"R: {test_r:.4f}")
        log_message(f"R²: {test_r2:.4f}")
        log_message(f"MAE: {test_mae:.4f}")
        log_message(f"RMSE: {test_rmse:.4f}")

        return y_train_pred, y_test_pred, test_r2, test_mae, test_rmse
    except Exception as e:
        log_message(f"Error in evaluate_model: {str(e)}")
        raise


def plot_results(model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, test_r2, test_mae):
    try:

        # 预测结果图
        try:
            myscatterplot(
                y_train.values,
                y_train_pred,
                y_test.values,
                y_test_pred,
                modelname="XGBoost",
                target="PCE",
                plot_height=8,
                savepic=True,
                picname='XGB_PCE_prediction'
            )
        except Exception as e:
            log_message(f"Error using myscatterplot: {str(e)}, using fallback plotting")
            plt.figure(figsize=(10, 8))
            plt.scatter(y_train, y_train_pred, color='#D399C2', s=80, alpha=0.7,
                        edgecolor='k', linewidth=0.5, label='Train')
            plt.scatter(y_test, y_test_pred, color='#DFCA33', s=80, alpha=0.7,
                        edgecolor='k', linewidth=0.5, marker='D', label='Test')
            max_val = max(y_train.max(), y_test.max())
            min_val = min(y_train.min(), y_test.min())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5)
            plt.xlabel('Actual PCE (%)', fontsize=14)
            plt.ylabel('Predicted PCE (%)', fontsize=14)
            plt.title('XGBoost: Actual vs Predicted PCE', fontsize=16)
            plt.legend(fontsize=12)
            plt.text(0.05, 0.9, f'Train R² = {r2_score(y_train, y_train_pred):.3f}',
                     transform=plt.gca().transAxes, fontsize=12)
            plt.text(0.05, 0.85, f'Test R² = {test_r2:.3f}',
                     transform=plt.gca().transAxes, fontsize=12)
            plt.text(0.05, 0.8, f'Test MAE = {test_mae:.3f}',
                     transform=plt.gca().transAxes, fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.savefig("picture_predict/XGB_PCE_prediction.png", dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        log_message(f"Error in plot_results: {str(e)}")
        raise


def print_model_parameters(model, grid_search=None):
    """输出模型的详细参数"""
    log_message("\n" + "=" * 60)
    log_message("XGBOOST MODEL DETAILED PARAMETERS")
    log_message("=" * 60)

    # 获取模型的所有参数
    model_params = model.get_params()

    # 分类显示参数
    log_message("\n=== 核心超参数 ===")
    core_params = ['n_estimators', 'max_depth', 'learning_rate', 'subsample',
                   'colsample_bytree', 'gamma', 'reg_alpha', 'reg_lambda']
    for param in core_params:
        if param in model_params:
            log_message(f"{param}: {model_params[param]}")

    log_message("\n=== 训练控制参数 ===")
    training_params = ['random_state', 'n_jobs', 'booster', 'objective', 'base_score']
    for param in training_params:
        if param in model_params:
            log_message(f"{param}: {model_params[param]}")

    log_message("\n=== 树结构参数 ===")
    tree_params = ['max_leaves', 'max_bin', 'tree_method', 'grow_policy']
    for param in tree_params:
        if param in model_params:
            log_message(f"{param}: {model_params[param]}")

    # 如果是网格搜索得到的模型，显示网格搜索信息
    if grid_search is not None:
        log_message("\n=== 网格搜索信息 ===")
        log_message(f"最佳参数: {grid_search.best_params_}")
        log_message(f"最佳交叉验证分数 (R²): {grid_search.best_score_:.4f}")
        log_message(f"搜索的参数组合总数: {len(grid_search.cv_results_['params'])}")

    # 显示模型统计信息
    log_message("\n=== 模型统计信息 ===")
    log_message(f"模型保存路径: {MODEL_PATH}")
    log_message(f"树的棵数 (n_estimators): {model.n_estimators}")
    log_message("\n" + "=" * 60)


def main():
    try:
        best_xgb, X_train, X_test, y_train, y_test, grid_search = load_or_train_model()
        y_train_pred, y_test_pred, test_r2, test_mae, test_rmse = evaluate_model(
            best_xgb, X_train, X_test, y_train, y_test
        )

        # 输出模型参数
        print_model_parameters(best_xgb, grid_search)

        plot_results(
            best_xgb, X_train, X_test, y_train, y_test,
            y_train_pred, y_test_pred, test_r2, test_mae
        )

        log_message(f"\nModel saved to: {MODEL_PATH}")
        log_message("Feature importance plot saved to: picture_predict/XGB_feature_importance.png")
        log_message("Prediction plots saved to:")
        log_message("- img/XGB_PCE_prediction.png (if myscatterplot succeeded)")
        log_message("- picture_predict/XGB_PCE_prediction.png (fallback plot)")



    except Exception as e:
        log_message(f"\nFatal error: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()