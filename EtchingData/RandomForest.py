import numpy as np
import os
import joblib
# Import RandomForestRegressor from sklearn.ensemble
from sklearn.ensemble import RandomForestRegressor
# Keep GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from letsplotT import myscatterplot

# Create directory structure
os.makedirs("picture", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("img", exist_ok=True)

# Load data
df = pd.read_excel("FinalData.xlsx")
y = df['PCE']
X = df.drop(['PCE'], axis=1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

# Modify model path to reflect RandomForest model
MODEL_PATH = "models/best_randomforest_model.pkl"  # Use .pkl extension to save sklearn model
# MODEL_PATH = "models/best_randomforest_model_part.pkl"
grid_search = None  # For storing grid search object

# Check if pre-trained model exists
if os.path.exists(MODEL_PATH):
    print("Loading pre-trained RandomForest model...")
    # Use joblib to load pre-trained sklearn model
    model = joblib.load(MODEL_PATH)
else:
    print("Training new RandomForest model...")

    # Define parameter grid for RandomForestRegressor
    # Note: parameter names are different from CatBoost
    param_grid = {
        'n_estimators': [100, 200, 500],  # Number of trees
        'max_depth': [None, 10, 20, 30],  # Maximum depth of trees
        'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
        'min_samples_leaf': [1, 2, 4],  # Minimum samples required at leaf node
        'max_features': ['auto', 'sqrt'],  # Number of features to consider when looking for best split
        'bootstrap': [True, False]  # Whether to use bootstrap samples to build trees
    }

    # Create RandomForestRegressor model instance
    # Note: Removed CatBoost-specific parameters
    rf_model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1  # Use all available processors
    )

    # Use random search for hyperparameter tuning
    # Note: scoring='r2' is suitable for regression problems
    grid_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_grid,
        n_iter=30,  # Number of iterations
        cv=5,  # Cross-validation folds
        scoring='r2',
        n_jobs=-1,  # Use all available processors
        verbose=2,  # Control verbosity
        random_state=42
    )

    # Fit model
    grid_search.fit(X_train, y_train)

    # Get best model
    model = grid_search.best_estimator_

    # Save best model
    joblib.dump(model, MODEL_PATH)

    print("\n=== Best Parameters ===")
    print(grid_search.best_params_)
    print(f"Best CV R²: {grid_search.best_score_:.4f}")

# Predict results
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


# Calculate all evaluation metrics (this part remains unchanged)
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

# Feature importance
plt.figure(figsize=(12, 8))
# RandomForest model directly has feature_importances_ attribute
importances = model.feature_importances_

sorted_idx = np.argsort(importances)[::-1]
n_top = min(20, len(importances))

plt.barh(range(n_top), importances[sorted_idx[:n_top]], color='#2ecc71')  # Change color for distinction
plt.yticks(range(n_top), np.array(X.columns)[sorted_idx[:n_top]])
plt.title('RandomForest Feature Importance', fontsize=14)  # Update title
plt.xlabel('Importance Score', fontsize=12)
plt.tight_layout()
plt.savefig("picture/RF_feature_importance.png", dpi=300, bbox_inches='tight')  # Update save file name
plt.close()

# Visualize prediction results
try:
    myscatterplot(
        y_train.values,
        y_train_pred,
        y_test.values,
        y_test_pred,
        modelname="RandomForest",  # Update model name
        target="PCE",
        plot_height=8,
        savepic=True,
        picname='RF_PCE_prediction'  # Update image name prefix
    )
    print("Plot saved to: img/RF_PCE_prediction.png")
except Exception as e:
    print(f"Error using myscatterplot: {str(e)}")

# ========== Output Model Core Parameters ==========
print("\n" + "=" * 60)
print("RANDOM FOREST MODEL CORE PARAMETERS")
print("=" * 60)

# Get model parameters
model_params = model.get_params()

# Display core parameters
print("\n=== Core Hyperparameters ===")
core_params = ['n_estimators', 'criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf',
               'min_weight_fraction_leaf', 'max_features', 'max_leaf_nodes', 'min_impurity_decrease',
               'bootstrap', 'oob_score', 'n_jobs', 'random_state', 'verbose', 'warm_start',
               'ccp_alpha', 'max_samples']
for param in core_params:
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
print(f"Number of Trees: {model.n_estimators}")
print(f"Number of Features: {model.n_features_in_ if hasattr(model, 'n_features_in_') else X.shape[1]}")
print(f"Feature Names: {list(X.columns)}")

if hasattr(model, 'oob_score_'):
    print(f"Out-of-Bag Score: {model.oob_score_:.4f}")



# Display tree statistics
print("\n=== Tree Statistics ===")
print(f"Number of Trees: {len(model.estimators_)}")
if len(model.estimators_) > 0:
    tree_depths = [tree.tree_.max_depth for tree in model.estimators_]
    print(f"Average Tree Depth: {np.mean(tree_depths):.2f}")
    print(f"Maximum Tree Depth: {np.max(tree_depths)}")
    print(f"Minimum Tree Depth: {np.min(tree_depths)}")

