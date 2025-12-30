import numpy as np
import os
import joblib
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib

# Ignore all warnings
import warnings

warnings.filterwarnings('ignore')

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from letsplotT import myscatterplot

# Create directory structure
os.makedirs("picture", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("img", exist_ok=True)

# Load data and handle spaces in column names
df = pd.read_excel("FinalDataAll.xlsx")
df.columns = [c.replace(' ', '_') for c in df.columns]  # Replace spaces with underscores
y = df['PCE']
X = df.drop(['PCE'], axis=1)  # Assume all features are continuous variables

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# Model path
MODEL_PATH = "models/best_lgbm_model.pkl"
grid_search = None  # For saving grid search object

if os.path.exists(MODEL_PATH):
    print("Loading pre-trained LGBM model...")
    model = joblib.load(MODEL_PATH)
else:
    print("Training new LGBM model...")

    # Adjust parameter grid to keep total runs within 500 (10-fold cross-validation)
    # Target combinations: Within 50 (50*10=500)
    param_grid = {
        'num_leaves': [31, 50],  # 2 options
        'max_depth': [6, 10],  # 2 options
        'learning_rate': [0.05, 0.1],  # 2 options
        'n_estimators': [200, 300],  # 2 options
        'min_child_samples': [10, 20],  # 2 options
        'subsample': [0.8, 1.0]  # 2 options
    }

    # Calculate total parameter combinations and CV runs
    total_combinations = 1
    for key in param_grid:
        total_combinations *= len(param_grid[key])
    total_cv_runs = total_combinations * 10  # Using 10-fold cross-validation

    print(f"Total parameter combinations: {total_combinations}")
    print(f"Total CV runs (10-fold): {total_cv_runs}")

    lgb_estimator = lgb.LGBMRegressor(random_state=42, silent=True)

    # Use 10-fold cross-validation
    grid_search = GridSearchCV(
        estimator=lgb_estimator,
        param_grid=param_grid,
        cv=10,  # Use 10-fold cross-validation
        scoring='r2',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)

    # Save best model
    model = grid_search.best_estimator_
    joblib.dump(model, MODEL_PATH)

    print("\n=== Best Parameters ===")
    print(grid_search.best_params_)
    print(f"Best CV R²: {grid_search.best_score_:.4f}")

# Prediction
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


# ========== Evaluation Metrics Calculation ==========
def calculate_metrics(y_true, y_pred):
    """Calculate all evaluation metrics"""
    r = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r, r2, mae, rmse


# Calculate training set metrics
train_r, train_r2, train_mae, train_rmse = calculate_metrics(y_train, y_train_pred)

# Calculate test set metrics
test_r, test_r2, test_mae, test_rmse = calculate_metrics(y_test, y_test_pred)

# Print training set metrics
print("\n=== Training Set Metrics ===")
print(f"R: {train_r:.4f}")
print(f"R²: {train_r2:.4f}")
print(f"MAE: {train_mae:.4f}")
print(f"RMSE: {train_rmse:.4f}")

# Print test set metrics
print("\n=== Test Set Metrics ===")
print(f"R: {test_r:.4f}")
print(f"R²: {test_r2:.4f}")
print(f"MAE: {test_mae:.4f}")
print(f"RMSE: {test_rmse:.4f}")

# Use myscatterplot for plotting
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
    # Alternative plotting solution
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

    # Add statistical information
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

# Get all parameters of the model
model_params = model.get_params()

# Display parameters by category
print("\n=== Core Hyperparameters ===")
core_params = ['num_leaves', 'max_depth', 'learning_rate', 'n_estimators',
               'min_child_samples', 'subsample', 'colsample_bytree']
for param in core_params:
    if param in model_params:
        print(f"{param}: {model_params[param]}")

print("\n=== Training Control Parameters ===")
training_params = ['random_state', 'boosting_type', 'objective', 'metric',
                   'n_jobs', 'silent', 'verbosity']
for param in training_params:
    if param in model_params:
        print(f"{param}: {model_params[param]}")

print("\n=== Regularization Parameters ===")
regularization_params = ['reg_alpha', 'reg_lambda', 'min_split_gain', 'min_child_weight']
for param in regularization_params:
    if param in model_params:
        print(f"{param}: {model_params[param]}")

print("\n=== Other Parameters ===")
other_params = ['importance_type', 'class_weight', 'subsample_for_bin', 'min_child_samples']
for param in other_params:
    if param in model_params and param not in core_params + training_params + regularization_params:
        print(f"{param}: {model_params[param]}")

# If model was obtained through grid search, display grid search information
if grid_search is not None:
    print("\n=== Grid Search Information ===")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score (R²): {grid_search.best_score_:.4f}")
    print(f"Total parameter combinations searched: {len(grid_search.cv_results_['params'])}")

    # Display top 5 best parameter combinations
    print("\n=== Top 5 Best Parameter Combinations ===")
    results_df = pd.DataFrame(grid_search.cv_results_)
    top_5 = results_df.nlargest(5, 'mean_test_score')
    for i, (_, row) in enumerate(top_5.iterrows()):
        print(f"Rank {i + 1}: R² = {row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}")
        print(f"  Parameters: {row['params']}")
else:
    print("\n=== Model Source ===")
    print("Model loaded from file, no new grid search performed")

# Display model statistics
print("\n=== Model Statistics ===")
print(f"Model saved path: {MODEL_PATH}")
print(f"Boosting iterations (n_estimators): {model.n_estimators}")
print(f"Is model trained: {'Yes' if hasattr(model, '_Booster') else 'No'}")

if hasattr(model, 'n_features_in_'):
    print(f"Number of input features: {model.n_features_in_}")
    print(f"Feature names: {list(X.columns)}")
else:
    print(f"Number of features: {X.shape[1]}")
    print(f"Feature names: {list(X.columns)}")