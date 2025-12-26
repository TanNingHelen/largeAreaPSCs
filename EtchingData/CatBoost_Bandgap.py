import numpy as np
import os
import joblib
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# 创建目录结构
os.makedirs("models", exist_ok=True)
os.makedirs("img", exist_ok=True)

# 加载数据
df = pd.read_excel("FinalData.xlsx")

# 选择特定特征：Cs, MA, FA, I, Br
feature_columns = ['Cs', 'MA', 'FA', 'Pb','I', 'Br','Cl']
target_column = 'Bandgap'

# 检查特征是否存在
missing_features = [col for col in feature_columns if col not in df.columns]
if missing_features:
    print(f"❌ 缺失特征: {missing_features}")
    exit()

print("✅ 所有必需特征都存在")

# 准备数据
X = df[feature_columns]
y = df[target_column]

print(f"数据形状: {X.shape}")
print(f"特征列表: {feature_columns}")
print(f"目标变量: {target_column}")
print(f"Bandgap统计: 最小值={y.min():.4f}, 最大值={y.max():.4f}, 均值={y.mean():.4f}")

# 显示特征统计信息
print("\n=== 特征统计信息 ===")
for col in feature_columns:
    print(f"{col}: 最小值={X[col].min():.4f}, 最大值={X[col].max():.4f}, 均值={X[col].mean():.4f}")

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

# 模型路径
MODEL_PATH = "models/best_catboost_bandgap.cbm"


def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    r = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r, r2, mae, rmse


def plot_feature_importance(model, feature_names):
    """绘制特征重要性图"""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.get_feature_importance()
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance for Bandgap Prediction')
    plt.tight_layout()
    plt.savefig('img/feature_importance_bandgap_simple.png', dpi=300, bbox_inches='tight')
    plt.close()

    return importance_df


def plot_predictions(y_train_true, y_train_pred, y_test_true, y_test_pred):
    """绘制预测结果图"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(y_train_true, y_train_pred, alpha=0.6, color='blue', label='Training')
    plt.plot([y_train_true.min(), y_train_true.max()], [y_train_true.min(), y_train_true.max()], 'r--', linewidth=2)
    plt.xlabel('Actual Bandgap (eV)')
    plt.ylabel('Predicted Bandgap (eV)')
    plt.title('Training Set')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(y_test_true, y_test_pred, alpha=0.6, color='green', label='Test')
    plt.plot([y_test_true.min(), y_test_true.max()], [y_test_true.min(), y_test_true.max()], 'r--', linewidth=2)
    plt.xlabel('Actual Bandgap (eV)')
    plt.ylabel('Predicted Bandgap (eV)')
    plt.title('Test Set')
    plt.legend()

    plt.tight_layout()
    plt.savefig('img/bandgap_prediction_simple.png', dpi=300, bbox_inches='tight')
    plt.close()


# 检查模型是否存在
model_exists = os.path.exists(MODEL_PATH)
if model_exists:
    try:
        model = CatBoostRegressor()
        model.load_model(MODEL_PATH)
        print("✅ 加载预训练的CatBoost Bandgap模型...")
    except Exception as e:
        print(f"❌ 加载现有模型失败: {str(e)}，重新训练...")
        model_exists = False

if not model_exists:
    print("🚀 训练新的CatBoost Bandgap模型...")

    # 简化的参数网格（特征少，不需要复杂参数）
    param_grid = {
        'iterations': [300, 500, 800],
        'depth': [4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5],
        'random_strength': [1, 2],
    }

    # 使用交叉验证找到最佳参数
    best_score = -np.inf
    best_params = None
    best_model = None

    print("🔍 进行参数搜索...")

    for iterations in param_grid['iterations']:
        for depth in param_grid['depth']:
            for lr in param_grid['learning_rate']:
                for l2 in param_grid['l2_leaf_reg']:
                    for random_strength in param_grid['random_strength']:

                        model = CatBoostRegressor(
                            iterations=iterations,
                            depth=depth,
                            learning_rate=lr,
                            l2_leaf_reg=l2,
                            random_strength=random_strength,
                            loss_function='RMSE',
                            eval_metric='R2',
                            random_seed=42,
                            verbose=False,
                            thread_count=-1
                        )

                        # 使用交叉验证评估
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                        mean_score = cv_scores.mean()

                        if mean_score > best_score:
                            best_score = mean_score
                            best_params = {
                                'iterations': iterations,
                                'depth': depth,
                                'learning_rate': lr,
                                'l2_leaf_reg': l2,
                                'random_strength': random_strength
                            }
                            best_model = model

    # 用最佳参数训练最终模型
    print(f"🎯 最佳参数: {best_params}")
    print(f"最佳交叉验证R²: {best_score:.4f}")

    model = CatBoostRegressor(**best_params, random_seed=42, verbose=100)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50, verbose=False)

    # 保存模型
    model.save_model(MODEL_PATH)
    print("✅ 模型保存完成")

# 预测结果
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 计算评估指标
train_r, train_r2, train_mae, train_rmse = calculate_metrics(y_train, y_train_pred)
test_r, test_r2, test_mae, test_rmse = calculate_metrics(y_test, y_test_pred)

print("\n" + "=" * 50)
print("=== 最终模型性能 - Bandgap预测 ===")
print("=" * 50)
print("\n=== 训练集指标 ===")
print(f"R: {train_r:.4f}")
print(f"R²: {train_r2:.4f}")
print(f"MAE: {train_mae:.4f} eV")
print(f"RMSE: {train_rmse:.4f} eV")

print("\n=== 测试集指标 ===")
print(f"R: {test_r:.4f}")
print(f"R²: {test_r2:.4f}")
print(f"MAE: {test_mae:.4f} eV")
print(f"RMSE: {test_rmse:.4f} eV")

# 计算过拟合程度
overfit_gap = train_r2 - test_r2
print(f"\n=== 过拟合分析 ===")
print(f"训练集-测试集R²差距: {overfit_gap:.4f}")
if overfit_gap > 0.2:
    print("⚠️  检测到明显过拟合!")
elif overfit_gap > 0.1:
    print("ℹ️  中等程度过拟合")
else:
    print("✅ 泛化性能良好")

# 特征重要性分析
print("\n=== 特征重要性分析 ===")
importance_df = plot_feature_importance(model, feature_columns)
print("特征重要性排序:")
for idx, row in importance_df.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# 绘制预测结果
plot_predictions(y_train, y_train_pred, y_test, y_test_pred)

# 保存预测结果
print("\n=== 保存预测结果 ===")
results_df = pd.DataFrame({
    'Actual_Bandgap': pd.concat([y_train, y_test]),
    'Predicted_Bandgap': np.concatenate([y_train_pred, y_test_pred]),
    'Dataset': ['Training'] * len(y_train) + ['Test'] * len(y_test)
})

# 添加特征信息
for col in feature_columns:
    results_df[col] = pd.concat([X_train[col], X_test[col]]).values

results_df.to_csv('models/bandgap_predictions_simple.csv', index=False)
print(f"预测结果保存到: models/bandgap_predictions_simple.csv")

# 保存模型性能信息
model_info = {
    'train_r2': train_r2,
    'test_r2': test_r2,
    'train_mae': train_mae,
    'test_mae': test_mae,
    'train_rmse': train_rmse,
    'test_rmse': test_rmse,
    'overfit_gap': overfit_gap,
    'features': ','.join(feature_columns)
}

model_info_df = pd.DataFrame([model_info])
model_info_df.to_csv('models/bandgap_model_performance_simple.csv', index=False)

print("\n=== 模型信息 ===")
print(f"Bandgap模型保存到: {MODEL_PATH}")
print(f"特征重要性图保存到: img/feature_importance_bandgap_simple.png")
print(f"预测结果图保存到: img/bandgap_prediction_simple.png")
print(f"训练集样本数: {len(y_train)}")
print(f"测试集样本数: {len(y_test)}")

# 额外统计信息
print(f"\n=== Bandgap预测统计 ===")
print(f"实际Bandgap范围: {y.min():.4f} - {y.max():.4f} eV")
print(f"预测Bandgap范围: {results_df['Predicted_Bandgap'].min():.4f} - {results_df['Predicted_Bandgap'].max():.4f} eV")
print(f"测试集MAE相对误差: {test_mae / y.mean() * 100:.2f}%")

print("\n🎉 Bandgap预测模型训练完成!")