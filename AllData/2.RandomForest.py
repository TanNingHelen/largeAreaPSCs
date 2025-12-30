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

# Create necessary directories
os.makedirs("picture_predict", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("img", exist_ok=True)

# Load data
df1 = pd.read_excel(r"FinalDataAll.xlsx")
Y = df1['PCE']
X = df1.drop(['PCE'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=12)

# Check if saved model exists
MODEL_PATH = "models/best_rf_model.pkl"

if os.path.exists(MODEL_PATH):
    print("Loading pre-trained Random Forest model...")
    best_rf = joblib.load(MODEL_PATH)
    print("Loading existing model, cannot display best parameters from grid search, but can display actual model parameters")
else:
    print("Training new Random Forest model...")

    # Redesign parameter grid, maintain 10-fold cross-validation, total folds controlled within 1000
    # Calculation: parameter combinations × 10 ≤ 1000 → parameter combinations ≤ 100
    param_grid = {
        'n_estimators': [50, 100],  # 2 options
        'max_depth': [10, 20, None],  # 3 options
        'min_samples_split': [2, 5],  # 2 options
        'min_samples_leaf': [1, 2],  # 2 options
        'max_features': ['sqrt', 'log2']  # 2 options
    }

    # Total parameter combinations: 2 × 3 × 2 × 2 × 2 = 48
    # Total training times: 48 × 10 = 480 < 1000

    rf = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
        bootstrap=True,
        oob_score=True  # Enable out-of-bag score evaluation
    )

    # Maintain 10-fold cross-validation
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        scoring='r2',
        verbose=2
    )

    print("Starting Random Forest model training...")
    print(f"Parameter combinations: {2 * 3 * 2 * 2 * 2}")
    print(f"Cross-validation folds: {cv.n_splits}")
    print(f"Total training times: {2 * 3 * 2 * 2 * 2 * cv.n_splits}")

    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    joblib.dump(best_rf, MODEL_PATH)

    print("\n=== Best Parameters from Grid Search ===")
    print(grid_search.best_params_)
    print(f"Best CV R²: {grid_search.best_score_:.4f}")

    # Output out-of-bag score
    if hasattr(best_rf, 'oob_score_'):
        print(f"OOB Score: {best_rf.oob_score_:.4f}")

# Predict results
y_train_pred = best_rf.predict(X_train)
y_test_pred = best_rf.predict(X_test)


# ========== Evaluation Metrics Section ==========
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

# ========== End of Evaluation Metrics Section ==========

# Use myscatterplot for plotting
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

print("\n" + "=" * 60)
print("BEST MODEL PARAMETERS")
print("=" * 60)

# Get all parameters of the model
model_params = best_rf.get_params()

# Display parameters by category
print("\n=== Core Parameters ===")
core_params = ['n_estimators', 'max_depth', 'min_samples_split',
               'min_samples_leaf', 'max_features', 'bootstrap']
for param in core_params:
    if param in model_params:
        print(f"{param}: {model_params[param]}")

print("\n=== Training Control Parameters ===")
training_params = ['random_state', 'n_jobs', 'verbose', 'warm_start']
for param in training_params:
    if param in model_params:
        print(f"{param}: {model_params[param]}")

print("\n=== Other Important Parameters ===")
other_params = ['criterion', 'max_leaf_nodes', 'min_impurity_decrease',
                'min_weight_fraction_leaf', 'oob_score', 'ccp_alpha']
for param in other_params:
    if param in model_params:
        print(f"{param}: {model_params[param]}")

# Display tree statistics
print("\n=== Model Statistics ===")
print(f"Model saved path: {MODEL_PATH}")
print(f"Number of trees in forest: {len(best_rf.estimators_)}")
print(f"Number of features: {best_rf.n_features_in_}")
if hasattr(best_rf, 'n_features_in_'):
    print(f"Feature names: {list(X.columns)}")

print("\n" + "=" * 60)