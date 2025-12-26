import numpy as np
import os
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from letsplotT import myscatterplot

# Create necessary directories
os.makedirs("picture_predict/picture_predict", exist_ok=True)
os.makedirs("picture_predict/models", exist_ok=True)
os.makedirs("picture_predict/img", exist_ok=True)

# Load data
df1 = pd.read_excel(r"FinalDataAll.xlsx")
Y = df1['PCE']
X = df1.drop(['PCE'], axis=1)

# Decision tree doesn't require standardization, use original data directly
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=12)

# Check if saved model exists
MODEL_PATH = "models/best_dt_model.pkl"
grid_search = None  # For storing grid search object

if os.path.exists(MODEL_PATH):
    print("Loading pre-trained Decision Tree model...")
    best_dt = joblib.load(MODEL_PATH)
else:
    print("Training new Decision Tree model...")

    param_grid = {
        'criterion': ['squared_error'],  # 1 option
        'max_depth': [None, 10, 20],  # 3 options
        'min_samples_split': [2, 5, 10],  # 3 options
        'min_samples_leaf': [1, 2],  # 2 options
        'max_features': [None, 'sqrt']  # 2 options
    }

    # Total parameter combinations: 1 × 3 × 3 × 2 × 2 = 36
    # Total training runs: 36 × 10 = 360 < 500

    # Calculate total parameter combinations
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)
    total_training_runs = total_combinations * 10
    print(f"Total parameter combinations: {total_combinations}")
    print(f"Total training runs (with 10-fold CV): {total_training_runs}")

    # Create base decision tree model
    dt = DecisionTreeRegressor(random_state=42)

    # 10-fold cross validation
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=dt,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        scoring='r2',
        verbose=2,
        return_train_score=True
    )

    print("Starting decision tree model training...")
    grid_search.fit(X_train, y_train)

    best_dt = grid_search.best_estimator_

    # Save model
    joblib.dump(best_dt, MODEL_PATH)

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

# Predict results
y_train_pred = best_dt.predict(X_train)
y_test_pred = best_dt.predict(X_test)


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
        modelname="Decision Tree",
        target="PCE",
        plot_height=8,
        savepic=True,
        picname='DT_PCE_prediction'
    )
    print("Plot saved to: picture_predict/img/DT_PCE_prediction.png")
except Exception as e:
    print(f"Error using myscatterplot: {str(e)}")
    # Fallback plotting scheme
    plt.figure(figsize=(10, 8))
    plt.scatter(y_train, y_train_pred, color='#E48963', s=80, alpha=0.7, edgecolor='k', linewidth=0.5, label='Train')
    plt.scatter(y_test, y_test_pred, color='#1458C4', s=80, alpha=0.7, edgecolor='k', linewidth=0.5, marker='D',
                label='Test')
    max_val = max(Y.max(), y_train_pred.max(), y_test_pred.max())
    min_val = min(Y.min(), y_train_pred.min(), y_test_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5)

    plt.xlabel('Actual PCE (%)', fontsize=14)
    plt.ylabel('Predicted PCE (%)', fontsize=14)
    plt.title('Decision Tree: Actual vs Predicted PCE', fontsize=16)
    plt.legend(fontsize=12)

    plt.text(0.05, 0.9, f'Train R² = {train_r2:.3f}', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.05, 0.85, f'Test R² = {test_r2:.3f}', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.05, 0.8, f'Test MAE = {test_mae:.3f}', transform=plt.gca().transAxes, fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig("picture_predict/DT_PCE_prediction.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Custom plot saved to: picture_predict/DT_PCE_prediction.png")

# ========== Output Model Core Parameters ==========
print("\n" + "=" * 60)
print("DECISION TREE MODEL CORE PARAMETERS")
print("=" * 60)

# Get model parameters
model_params = best_dt.get_params()

# Display core parameters
print("\n=== Core Hyperparameters ===")
core_params = ['criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf',
               'min_weight_fraction_leaf', 'max_features', 'random_state', 'max_leaf_nodes']
for param in core_params:
    if param in model_params:
        print(f"{param}: {model_params[param]}")

print("\n=== Tree Splitting Parameters ===")
splitting_params = ['splitter', 'min_impurity_decrease', 'ccp_alpha']
for param in splitting_params:
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
print(f"Tree Depth: {best_dt.get_depth()}")
print(f"Number of Leaves: {best_dt.get_n_leaves()}")
print(f"Number of Features: {best_dt.n_features_in_ if hasattr(best_dt, 'n_features_in_') else X.shape[1]}")
print(f"Feature Names: {list(X.columns)}")

# Display feature importance
print("\n=== Feature Importance (Top 10) ===")
try:
    feature_importances = best_dt.feature_importances_
    if feature_importances is not None and len(feature_importances) > 0:
        indices = np.argsort(feature_importances)[::-1]
        feature_names = list(X.columns)

        for i in range(min(10, len(indices))):
            importance_value = feature_importances[indices[i]]
            print(f"{i + 1:2d}. {feature_names[indices[i]]:30s}: {importance_value:8.6f}")

        # Calculate importance statistics
        print(f"\nFeature Importance Statistics:")
        print(f"  Sum of all importances: {np.sum(feature_importances):.6f}")
        print(f"  Average importance: {np.mean(feature_importances):.6f}")
        print(f"  Maximum importance: {np.max(feature_importances):.6f}")
        print(f"  Minimum importance: {np.min(feature_importances):.6f}")

        # Identify zero importance features
        zero_importance_features = np.sum(feature_importances == 0)
        if zero_importance_features > 0:
            print(f"  Features with zero importance: {zero_importance_features}")
except Exception as e:
    print(f"Cannot get feature importance: {str(e)}")

# Display tree structure information
print("\n=== Tree Structure Information ===")
print(f"Number of Nodes: {best_dt.tree_.node_count}")
print(f"Number of Internal Nodes: {best_dt.tree_.n_node_samples.shape[0] - best_dt.get_n_leaves()}")
print(f"Impurity at Root Node: {best_dt.tree_.impurity[0]:.6f}")
print(f"Value at Root Node: {best_dt.tree_.value[0][0][0]:.4f}")

