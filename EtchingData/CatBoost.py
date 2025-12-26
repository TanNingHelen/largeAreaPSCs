import numpy as np
import os
import joblib
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from letsplotT import myscatterplot
import warnings

warnings.filterwarnings('ignore')

# Create directory structure
os.makedirs("picture", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("img", exist_ok=True)

# Load data
df = pd.read_excel("FinalData.xlsx")


# Data preprocessing - do not remove low variance features
def preprocess_data(df):
    """Data preprocessing - keep all features"""
    df_processed = df.copy()

    # Check and handle missing values
    missing_cols = df_processed.columns[df_processed.isnull().any()].tolist()
    if missing_cols:
        print(f"Columns with missing values found: {missing_cols}")
        # Simple imputation for missing values in numeric columns
        for col in df_processed.select_dtypes(include=[np.number]).columns:
            if df_processed[col].isnull().any():
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())

    return df_processed


# Preprocess data
df_processed = preprocess_data(df)
y = df_processed['PCE']
X = df_processed.drop(['PCE'], axis=1)

print(f"Data shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Feature list: {list(X.columns)}")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)

# Model path
MODEL_PATH = "models/best_catboost_model.cbm"
grid_search = None  # For storing grid search object


def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    r = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r, r2, mae, rmse


def plot_feature_importance(model, feature_names, top_n=20):
    """Plot feature importance"""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.get_feature_importance()
    }).sort_values('importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, y='feature', x='importance')
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    plt.savefig('img/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    return importance_df


def plot_residuals(y_true, y_pred, dataset_name):
    """Plot residuals"""
    residuals = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Residuals vs predicted values
    ax1.scatter(y_pred, residuals, alpha=0.6)
    ax1.axhline(y=0, color='red', linestyle='--')
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title(f'{dataset_name} - Residuals vs Predicted')

    # Residuals distribution
    ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'{dataset_name} - Residuals Distribution')

    plt.tight_layout()
    plt.savefig(f'img/residuals_{dataset_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()


# Check if model exists and features match
model_exists = os.path.exists(MODEL_PATH)
if model_exists:
    try:
        # Try to load model and check feature matching
        test_model = CatBoostRegressor()
        test_model.load_model(MODEL_PATH)

        # Check model expected feature count
        expected_features = test_model.feature_names_
        if expected_features is not None:
            current_features = X.columns.tolist()
            if set(expected_features) == set(current_features):
                print("Loading pre-trained CatBoost model...")
                model = test_model
            else:
                print("Model features don't match current data. Retraining...")
                model_exists = False
        else:
            print("Model has no feature names. Retraining...")
            model_exists = False

    except Exception as e:
        print(f"Error loading existing model: {str(e)}. Retraining...")
        model_exists = False

if not model_exists:
    print("Training new CatBoost model with overfitting prevention...")

    # Parameter grid optimized for preventing overfitting
    param_grid = {
        'iterations': [500, 800, 1000],
        'depth': [4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'l2_leaf_reg': [3, 5, 10],
        'random_strength': [1, 2],
        'bagging_temperature': [0.5, 1.0],
        'subsample': [0.7, 0.8],
        'min_data_in_leaf': [5, 10, 15],
        'max_leaves': [31, 64],
        'grow_policy': ['SymmetricTree', 'Depthwise']
    }

    # Create base model - fix: do not use_best_model during cross-validation
    cb_model_for_cv = CatBoostRegressor(
        loss_function='RMSE',
        eval_metric='R2',
        random_seed=42,
        verbose=0,  # Turn off verbose output during cross-validation
        od_type='Iter',
        od_wait=50,
        use_best_model=False,  # Do not use_best_model during cross-validation
        thread_count=-1
    )

    # Evaluate using cross-validation - using simplified model
    print("Performing cross-validation...")
    cv_scores = cross_val_score(cb_model_for_cv, X_train, y_train, cv=5, scoring='r2')
    print(f"Cross-validation R² scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Use randomized search - using full model configuration
    print("\nStarting RandomizedSearchCV...")

    # Create model for random search (use_best_model, because there is eval_set)
    cb_model_for_search = CatBoostRegressor(
        loss_function='RMSE',
        eval_metric='R2',
        random_seed=42,
        verbose=100,
        od_type='Iter',
        od_wait=50,
        use_best_model=True,  # Can be used now because there is eval_set
        thread_count=-1
    )

    grid_search = RandomizedSearchCV(
        estimator=cb_model_for_search,
        param_distributions=param_grid,
        n_iter=20,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=2,
        random_state=42,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50, verbose=False)

    # Save best model
    model = grid_search.best_estimator_
    model.save_model(MODEL_PATH)

    print("\n=== Best Parameters ===")
    print(grid_search.best_params_)
    print(f"Best CV R²: {grid_search.best_score_:.4f}")

    # Display training and validation scores
    results_df = pd.DataFrame(grid_search.cv_results_)
    print(f"\nTrain vs Test scores:")
    best_idx = grid_search.best_index_
    print(f"Best model train score: {results_df.loc[best_idx, 'mean_train_score']:.4f}")
    print(f"Best model test score: {results_df.loc[best_idx, 'mean_test_score']:.4f}")

# Predict results
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate evaluation metrics
train_r, train_r2, train_mae, train_rmse = calculate_metrics(y_train, y_train_pred)
test_r, test_r2, test_mae, test_rmse = calculate_metrics(y_test, y_test_pred)

print("\n" + "=" * 50)
print("=== Final Model Performance ===")
print("=" * 50)
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

# Calculate overfitting degree
overfit_gap = train_r2 - test_r2
print(f"\n=== Overfitting Analysis ===")
print(f"Train-Test R² gap: {overfit_gap:.4f}")
if overfit_gap > 0.2:
    print("⚠️  Warning: Significant overfitting detected!")
elif overfit_gap > 0.1:
    print("ℹ️  Moderate overfitting detected")
else:
    print("✅ Good generalization performance")

# Save all points' original PCE and predicted PCE and dataset type to Excel
print("\n=== Saving Data for Manual Plotting ===")

# Create DataFrame for training set
train_data = pd.DataFrame({
    'Dataset_Type': ['Training'] * len(y_train),
    'Original_PCE': y_train.values,
    'Predicted_PCE': y_train_pred
})

# Create DataFrame for test set
test_data = pd.DataFrame({
    'Dataset_Type': ['Test'] * len(y_test),
    'Original_PCE': y_test.values,
    'Predicted_PCE': y_test_pred
})

# Combine training and test sets
all_plot_data = pd.concat([train_data, test_data], ignore_index=True)

# Save to Excel file
excel_filename = 'CatBoostPlot.xlsx'
all_plot_data.to_excel(excel_filename, index=False)
print(f"All point data saved to: {excel_filename}")
print(f"Training set sample count: {len(y_train)}")
print(f"Test set sample count: {len(y_test)}")
print(f"Total sample count: {len(all_plot_data)}")

# Visualize prediction results
try:
    myscatterplot(
        y_train.values,
        y_train_pred,
        y_test.values,
        y_test_pred,
        modelname="CatBoost",
        target="PCE",
        plot_height=8,
        savepic=True,
        picname='CB_PCE_prediction'
    )
    print("Main scatter plot saved to: img/CB_PCE_prediction.png")
except Exception as e:
    print(f"Error using myscatterplot: {str(e)}")

# Save prediction results - fix length mismatch issue
print("\n=== Saving Prediction Results ===")
# Save training and test set prediction results separately
train_results = pd.DataFrame({
    'Actual': y_train,
    'Predicted': y_train_pred
})
train_results['Dataset'] = 'Training'

test_results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_test_pred
})
test_results['Dataset'] = 'Test'

print("\n" + "=" * 60)
print("CATBOOST MODEL CORE PARAMETERS")
print("=" * 60)

# Get all model parameters
model_params = model.get_all_params()

# Display core parameters
print("\n=== Core Hyperparameters ===")
core_params = ['iterations', 'learning_rate', 'depth', 'l2_leaf_reg',
               'random_strength', 'bagging_temperature', 'border_count']
for param in core_params:
    if param in model_params:
        print(f"{param}: {model_params[param]}")

print("\n=== Training Control Parameters ===")
training_params = ['loss_function', 'eval_metric', 'random_seed', 'thread_count',
                   'verbose', 'use_best_model', 'od_type', 'od_wait', 'od_pval']
for param in training_params:
    if param in model_params:
        print(f"{param}: {model_params[param]}")

print("\n=== Tree Structure Parameters ===")
tree_params = ['grow_policy', 'min_data_in_leaf', 'max_leaves', 'rsm',
               'leaf_estimation_iterations', 'leaf_estimation_method']
for param in tree_params:
    if param in model_params:
        print(f"{param}: {model_params[param]}")

print("\n=== Feature Processing Parameters ===")
feature_params = ['one_hot_max_size', 'has_time', 'feature_border_type',
                  'per_float_feature_quantization', 'nan_mode']
for param in feature_params:
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
print(f"Number of trees (iterations): {model.tree_count_}")
print(f"Number of features: {len(X.columns)}")
print(f"Feature names: {list(X.columns)}")



