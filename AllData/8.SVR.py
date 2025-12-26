import numpy as np
import os
import joblib
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
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

# Standardize features and target (SVR is sensitive to feature scale)
X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = X_scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Standardize target variable
y_scaled = y_scaler.fit_transform(Y.values.reshape(-1, 1)).ravel()

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=12)

# Check if saved model exists
MODEL_PATH = "models/best_svr_model.pkl"
X_SCALER_PATH = "models/svr_x_scaler.pkl"
Y_SCALER_PATH = "models/svr_y_scaler.pkl"
grid_search = None  # For storing grid search object

if os.path.exists(MODEL_PATH) and os.path.exists(X_SCALER_PATH) and os.path.exists(Y_SCALER_PATH):
    print("Loading pre-trained SVR model...")
    best_svr = joblib.load(MODEL_PATH)
    X_scaler = joblib.load(X_SCALER_PATH)
    y_scaler = joblib.load(Y_SCALER_PATH)
else:
    print("Training new SVR model...")

    # Redesign parameter grid to control total training runs under 500
    # Calculation: parameter combinations × 10 ≤ 500 → parameter combinations ≤ 50

    # Original parameter grid calculation:
    # kernel: 3 options, C: 4 options, gamma: 5 options, epsilon: 3 options, degree: 2 options
    # Total parameter combinations: 3 × 4 × 5 × 3 × 2 = 360
    # Total training runs: 360 × 10 = 3600 > 500

    # Optimized parameter grid
    param_grid = {
        'kernel': ['rbf', 'linear'],  # Reduced from 3 to 2 options
        'C': [0.1, 1, 10],  # Reduced from 4 to 3 options
        'gamma': ['scale', 0.01, 0.1],  # Reduced from 5 to 3 options
        'epsilon': [0.01, 0.1]  # Reduced from 3 to 2 options
    }

    # Total parameter combinations: 2 × 3 × 3 × 2 = 36
    # Total training runs: 36 × 10 = 360 < 500

    # Calculate total parameter combinations
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)
    total_training_runs = total_combinations * 10
    print(f"Total parameter combinations: {total_combinations}")
    print(f"Total training runs (with 10-fold CV): {total_training_runs}")

    # Create base SVR model
    svr = SVR(max_iter=10000)

    # 10-fold cross validation
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=svr,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        scoring='r2',
        verbose=2,
        return_train_score=True
    )

    print("Starting SVR model training...")
    grid_search.fit(X_train, y_train)

    best_svr = grid_search.best_estimator_

    # Save model and scalers
    joblib.dump(best_svr, MODEL_PATH)
    joblib.dump(X_scaler, X_SCALER_PATH)
    joblib.dump(y_scaler, Y_SCALER_PATH)

    print("\n=== Best Parameters ===")
    print(grid_search.best_params_)
    print(f"Best CV R²: {grid_search.best_score_:.4f}")

    # Output cross-validation results
    results_df = pd.DataFrame(grid_search.cv_results_)
    print("\nTop 5 parameter combinations:")
    top_5 = results_df.nlargest(5, 'mean_test_score')[
        ['mean_test_score', 'std_test_score', 'params']
    ]
    for i, row in top_5.iterrows():
        print(f"R²: {row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}")
        print(f"  Params: {row['params']}")

# Predict results (in standardized space)
y_train_pred_scaled = best_svr.predict(X_train)
y_test_pred_scaled = best_svr.predict(X_test)

# Convert predictions back to original space
y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()

# Convert true values back to original space
y_train_original = y_scaler.inverse_transform(y_train.reshape(-1, 1)).ravel()
y_test_original = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()


# ========== Evaluation Metrics Section ==========
def calculate_metrics(y_true, y_pred):
    r = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r, r2, mae, rmse


train_r, train_r2, train_mae, train_rmse = calculate_metrics(y_train_original, y_train_pred)
test_r, test_r2, test_mae, test_rmse = calculate_metrics(y_test_original, y_test_pred)

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

# Use myscatterplot for plotting
try:
    myscatterplot(
        y_train_original,
        y_train_pred,
        y_test_original,
        y_test_pred,
        modelname="SVR",
        target="PCE",
        plot_height=8,
        savepic=True,
        picname='SVR_PCE_prediction'
    )
    print("Plot saved to: img/SVR_PCE_prediction.png")
except Exception as e:
    print(f"Error using myscatterplot: {str(e)}")
    # Fallback plotting scheme
    plt.figure(figsize=(10, 8))
    plt.scatter(y_train_original, y_train_pred, color='#E48963', s=80, alpha=0.7, edgecolor='k', linewidth=0.5,
                label='Train')
    plt.scatter(y_test_original, y_test_pred, color='#1458C4', s=80, alpha=0.7, edgecolor='k', linewidth=0.5,
                marker='D',
                label='Test')
    max_val = max(Y.max(), y_train_pred.max(), y_test_pred.max())
    min_val = min(Y.min(), y_train_pred.min(), y_test_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5)

    plt.xlabel('Actual PCE (%)', fontsize=14)
    plt.ylabel('Predicted PCE (%)', fontsize=14)
    plt.title('SVR: Actual vs Predicted PCE', fontsize=16)
    plt.legend(fontsize=12)

    plt.text(0.05, 0.9, f'Train R² = {train_r2:.3f}', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.05, 0.85, f'Test R² = {test_r2:.3f}', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.05, 0.8, f'Test MAE = {test_mae:.3f}', transform=plt.gca().transAxes, fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig("picture_predict/SVR_PCE_prediction.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Custom plot saved to: picture_predict/SVR_PCE_prediction.png")

# ========== Output Model Core Parameters ==========
print("\n" + "=" * 60)
print("SVR MODEL CORE PARAMETERS")
print("=" * 60)

# Get model parameters
model_params = best_svr.get_params()

# Display core parameters
print("\n=== Core Hyperparameters ===")
core_params = ['C', 'epsilon', 'kernel', 'gamma', 'degree', 'coef0', 'shrinking',
               'tol', 'cache_size', 'max_iter', 'verbose']
for param in core_params:
    if param in model_params:
        print(f"{param}: {model_params[param]}")

print("\n=== Model Configuration ===")
config_params = ['decision_function_shape', 'break_ties', 'random_state']
for param in config_params:
    if param in model_params:
        print(f"{param}: {model_params[param]}")

# If the model was obtained through grid search, display grid search information
if grid_search is not None:
    print("\n=== Grid Search Information ===")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Score (R²): {grid_search.best_score_:.4f}")
    print(f"Total Parameter Combinations Searched: {len(grid_search.cv_results_['params'])}")

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
print("\n=== Model Statistical Information ===")
print(f"Model Save Path: {MODEL_PATH}")
print(f"X Scaler Save Path: {X_SCALER_PATH}")
print(f"Y Scaler Save Path: {Y_SCALER_PATH}")
print(f"Kernel: {best_svr.kernel}")
print(f"Number of support vectors: {len(best_svr.support_vectors_)}")
print(f"Number of features: {best_svr.n_features_in_ if hasattr(best_svr, 'n_features_in_') else X.shape[1]}")

# Display support vector statistics
print("\n=== Support Vector Statistics ===")
print(f"Support vector indices: {len(best_svr.support_)}")
print(f"Support vector dual coefficients shape: {best_svr.dual_coef_.shape}")
print(f"Intercept: {best_svr.intercept_[0] if hasattr(best_svr.intercept_, '__len__') else best_svr.intercept_}")

# Display model performance summary
print("\n=== Model Performance Summary ===")
print(f"Training Set R²: {train_r2:.4f}")
print(f"Test Set R²: {test_r2:.4f}")
print(f"Test Set MAE: {test_mae:.4f}")
print(f"Test Set RMSE: {test_rmse:.4f}")

print("\n=== Prediction Statistics ===")
print(f"Training predictions range: {y_train_pred.min():.4f} - {y_train_pred.max():.4f}")
print(f"Test predictions range: {y_test_pred.min():.4f} - {y_test_pred.max():.4f}")
print(f"Actual values range: {Y.min():.4f} - {Y.max():.4f}")

print("\n=== File Information ===")
print(f"Model file: {MODEL_PATH}")
print(f"X scaler file: {X_SCALER_PATH}")
print(f"Y scaler file: {Y_SCALER_PATH}")
print("Plots saved to:")
print("- img/SVR_PCE_prediction.png (if myscatterplot succeeded)")
print("- picture_predict/SVR_PCE_prediction.png (fallback plot)")

print("\n" + "=" * 60)
print("SVR MODEL PARAMETERS OUTPUT COMPLETE")
print("=" * 60)