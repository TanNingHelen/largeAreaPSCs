import numpy as np
import os
import joblib
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from letsplotT import myscatterplot

# Ignore all warnings
import warnings

warnings.filterwarnings('ignore')

matplotlib.use('Agg')

# Create directory structure
os.makedirs("picture", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("img", exist_ok=True)

# Load data and process spaces in column names
df = pd.read_excel("FinalData.xlsx")
df.columns = [c.replace(' ', '_') for c in df.columns]  # Replace spaces with underscores
y = df['PCE']
X = df.drop(['PCE'], axis=1)  # Assuming all features are continuous variables

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

# Model path
MODEL_PATH = "models/best_lgbm_model.pkl"
# MODEL_PATH = "models/best_lgbm_model_part.pkl"
grid_search = None  # For storing grid search object

if os.path.exists(MODEL_PATH):
    print("Loading pre-trained LGBM model...")
    model = joblib.load(MODEL_PATH)
else:
    print("Training new LGBM model...")

    # Parameter distribution optimized for small dataset (prevent overfitting, improve generalization)
    param_dist = {
        'num_leaves': [10, 15, 20, 25],  # Smaller leaf count to prevent overfitting
        'max_depth': [3, 4, 5, 6],  # Shallower depth
        'learning_rate': [0.005, 0.01, 0.02, 0.05],  # Smaller learning rate
        'n_estimators': [200, 300, 400, 500],  # More trees but with small learning rate
        'min_child_samples': [5, 10, 15, 20],  # Prevent overfitting
        'min_child_weight': [0.001, 0.01, 0.1],  # Add minimum leaf weight
        'subsample': [0.6, 0.7, 0.8, 0.9],  # Lower subsampling rate
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],  # Lower column sampling rate
        'reg_alpha': [0, 0.1, 0.5, 1.0],  # L1 regularization
        'reg_lambda': [0, 0.1, 0.5, 1.0, 2.0],  # L2 regularization
        'min_split_gain': [0, 0.01, 0.05],  # Minimum gain
        'boosting_type': ['gbdt'],  # Only use gbdt, more stable
        'objective': ['regression'],  # Simplified objective function
        'bagging_freq': [1, 3, 5],  # Bagging frequency
        'bagging_fraction': [0.6, 0.7, 0.8],  # Bagging fraction
        'feature_fraction': [0.6, 0.7, 0.8],  # Feature sampling fraction
        'extra_trees': [True, False]  # Extremely randomized trees mode
    }

    lgb_estimator = lgb.LGBMRegressor(
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    # Use random search
    grid_search = RandomizedSearchCV(
        estimator=lgb_estimator,
        param_distributions=param_dist,
        n_iter=50,  # Increase iterations to find better parameters
        cv=5,  # Keep 5-fold cross-validation
        scoring='r2',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    grid_search.fit(X_train, y_train)

    # Save best model
    model = grid_search.best_estimator_
    joblib.dump(model, MODEL_PATH)

    print("\n=== Best Parameters ===")
    print(grid_search.best_params_)
    print(f"Best CV R²: {grid_search.best_score_:.4f}")

# Predict
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


# Calculate all evaluation metrics
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

# ========== Output Model Core Parameters ==========
print("\n" + "=" * 60)
print("LIGHTGBM MODEL CORE PARAMETERS")
print("=" * 60)

# Get model parameters
model_params = model.get_params()

# Display core parameters
print("\n=== Core Hyperparameters ===")
core_params = ['num_leaves', 'max_depth', 'learning_rate', 'n_estimators',
               'min_child_samples', 'min_child_weight', 'subsample', 'colsample_bytree',
               'reg_alpha', 'reg_lambda', 'min_split_gain']
for param in core_params:
    if param in model_params:
        print(f"{param}: {model_params[param]}")

print("\n=== Training Control Parameters ===")
training_params = ['boosting_type', 'objective', 'random_state', 'n_jobs',
                   'verbose', 'silent', 'importance_type']
for param in training_params:
    if param in model_params:
        print(f"{param}: {model_params[param]}")

print("\n=== Advanced Parameters ===")
advanced_params = ['bagging_freq', 'bagging_fraction', 'feature_fraction',
                   'extra_trees', 'class_weight', 'subsample_for_bin']
for param in advanced_params:
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
print(f"Number of estimators: {model.n_estimators}")
print(f"Number of features: {model.n_features_in_ if hasattr(model, 'n_features_in_') else X.shape[1]}")
print(f"Feature names: {list(X.columns)}")


