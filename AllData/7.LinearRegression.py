import numpy as np
import os
import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from letsplotT import myscatterplot

# Create necessary directories
os.makedirs("picture_predict", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("img", exist_ok=True)

# Load data
df1 = pd.read_excel(r"FinalDataAll.xlsx")
Y = df1['PCE']
X = df1.drop(['PCE'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,
                                                    random_state=42)  # Use 42 for better reproducibility

# Check if saved model exists
MODEL_PATH = "models/best_linear_model.pkl"
best_model_name = None
best_params = None

if os.path.exists(MODEL_PATH):
    print("Loading pre-trained Linear Regression model...")
    best_model = joblib.load(MODEL_PATH)
else:
    print("Training new Linear Regression model...")

    # Create feature engineering pipeline
    # 1. Standardize features (important for regularized models)
    # 2. Can add polynomial features (optional, increases model complexity)

    # Define different linear models and parameter grids, using Pipeline
    models = {
        'LinearRegression': {
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('model', LinearRegression())
            ]),
            'params': {
                'model__fit_intercept': [True, False],
                'model__positive': [True, False]
            }
        },
        'Ridge': {
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('model', Ridge(random_state=42))
            ]),
            'params': {
                'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],  # Extended range
                'model__fit_intercept': [True, False],
                'model__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga']
            }
        },
        'Lasso': {
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('model', Lasso(random_state=42, max_iter=10000))
            ]),
            'params': {
                'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],  # Extended range
                'model__fit_intercept': [True, False],
                'model__selection': ['cyclic', 'random'],
                'model__tol': [1e-4, 1e-3, 1e-2]
            }
        },
        'ElasticNet': {
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('model', ElasticNet(random_state=42, max_iter=10000))
            ]),
            'params': {
                'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],  # More detailed l1_ratio
                'model__fit_intercept': [True, False],
                'model__tol': [1e-4, 1e-3, 1e-2]
            }
        }
    }

    # Add Ridge regression with polynomial features
    models['Ridge_Poly2'] = {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('model', Ridge(random_state=42))
        ]),
        'params': {
            'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
            'model__fit_intercept': [True, False],
            'model__solver': ['auto', 'svd', 'cholesky']
        }
    }

    best_score = -np.inf
    best_model = None
    best_model_name = None
    best_params = None

    # Use stratified 10-fold cross-validation
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    for model_name, model_info in models.items():
        print(f"\n=== Training {model_name} ===")

        grid_search = GridSearchCV(
            estimator=model_info['pipeline'],
            param_grid=model_info['params'],
            cv=cv,
            n_jobs=-1,
            scoring='r2',
            verbose=1,
            return_train_score=True  # Record training scores
        )

        grid_search.fit(X_train, y_train)

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
            best_model_name = model_name
            best_params = grid_search.best_params_

        print(f"Best {model_name} CV R²: {grid_search.best_score_:.4f}")
        print(f"Best {model_name} parameters: {grid_search.best_params_}")

        # Check overfitting degree
        train_score = grid_search.cv_results_['mean_train_score'][grid_search.best_index_]
        test_score = grid_search.best_score_
        overfit_gap = train_score - test_score
        print(f"Train R²: {train_score:.4f}, Test R²: {test_score:.4f}, Gap: {overfit_gap:.4f}")

    # Save best model
    joblib.dump(best_model, MODEL_PATH)

    print(f"\n=== Best Overall Model ===")
    print(f"Model: {best_model_name}")
    print(f"Best CV R²: {best_score:.4f}")
    print(f"Best Parameters: {best_params}")

# Predict results
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)


# ========== Evaluation Metrics Section ==========
def calculate_metrics(y_true, y_pred):
    r = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r, r2, mae, rmse


train_r, train_r2, train_mae, train_rmse = calculate_metrics(y_train, y_train_pred)
test_r, test_r2, test_mae, test_rmse = calculate_metrics(y_test, y_test_pred)

print("\n" + "=" * 50)
print("Training Set Metrics:")
print(f"R: {train_r:.4f}")
print(f"R²: {train_r2:.4f}")
print(f"MAE: {train_mae:.4f}")
print(f"RMSE: {train_rmse:.4f}")

print("\nTest Set Metrics:")
print(f"R: {test_r:.4f}")
print(f"R²: {test_r2:.4f}")
print(f"MAE: {test_mae:.4f}")
print(f"RMSE: {test_rmse:.4f}")
print("=" * 50)

# Use myscatterplot for plotting
try:
    myscatterplot(
        y_train.values,
        y_train_pred,
        y_test.values,
        y_test_pred,
        modelname="Linear Regression",
        target="PCE",
        plot_height=8,
        savepic=True,
        picname='Linear_PCE_prediction'
    )
    print("Plot saved to: img/Linear_PCE_prediction.png")
except Exception as e:
    print(f"Error using myscatterplot: {str(e)}")
    # Fallback plotting scheme
    plt.figure(figsize=(10, 8))
    plt.scatter(y_train, y_train_pred, color='#E48963', s=80, alpha=0.7, edgecolor='k', linewidth=0.5, label='Train')
    plt.scatter(y_test, y_test_pred, color='#1458C4', s=80, alpha=0.7, edgecolor='k', linewidth=0.5, marker='D',
                label='Test')
    max_val = max(y_train.max(), y_test.max())
    min_val = min(y_train.min(), y_test.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5)

    plt.xlabel('Actual PCE (%)', fontsize=14)
    plt.ylabel('Predicted PCE (%)', fontsize=14)
    plt.title(f'Linear Regression: Actual vs Predicted PCE', fontsize=16)
    plt.legend(fontsize=12)

    plt.text(0.05, 0.9, f'Train R² = {train_r2:.3f}', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.05, 0.85, f'Test R² = {test_r2:.3f}', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.05, 0.80, f'Test MAE = {test_mae:.3f}', transform=plt.gca().transAxes, fontsize=12)

    # Add equation line (if it's the simplest linear regression)
    if hasattr(best_model, 'coef_') and hasattr(best_model, 'intercept_'):
        if len(best_model.coef_.shape) == 0 or best_model.coef_.shape[0] == 1:
            slope = best_model.coef_[0] if hasattr(best_model.coef_, '__len__') else best_model.coef_
            intercept = best_model.intercept_
            plt.text(0.05, 0.75, f'y = {slope:.3f}x + {intercept:.3f}',
                     transform=plt.gca().transAxes, fontsize=10)

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig("picture_predict/Linear_PCE_prediction.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Custom plot saved to: picture_predict/Linear_PCE_prediction.png")

# ========== Output Best Model Parameters ==========
print("\n" + "=" * 60)
print("LINEAR REGRESSION MODEL DETAILED PARAMETERS")
print("=" * 60)

print(f"\n=== Best Model Information ===")
print(f"Model Type: {best_model_name}")
print(f"Model Save Path: {MODEL_PATH}")

# Extract model from pipeline if applicable
if hasattr(best_model, 'named_steps'):
    print("\n=== Pipeline Structure ===")
    for step_name, step in best_model.named_steps.items():
        print(f"Step: {step_name}, Type: {type(step).__name__}")

    # Get the actual regression model
    if 'model' in best_model.named_steps:
        reg_model = best_model.named_steps['model']
    else:
        # Try to find the regression model in the pipeline
        for step_name, step in best_model.named_steps.items():
            if hasattr(step, 'coef_'):
                reg_model = step
                break
        else:
            reg_model = best_model
else:
    reg_model = best_model

print(f"\n=== Regression Model Type ===")
print(type(reg_model).__name__)

# Get model parameters
model_params = reg_model.get_params()

print("\n=== Model Hyperparameters ===")
for param_name, param_value in model_params.items():
    print(f"{param_name}: {param_value}")








