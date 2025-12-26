import numpy as np
import os
import joblib
from sklearn.neural_network import MLPRegressor
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

# Standardize features (MLP is sensitive to feature scale)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=12)

# Check if saved model exists
MODEL_PATH = "models/best_mlp_model.pkl"
SCALER_PATH = "models/mlp_scaler.pkl"
grid_search = None  # For storing grid search object

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    print("Loading pre-trained MLP model...")
    best_mlp = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    print("Training new MLP model...")

    # Use more effective parameter grid, focusing on the most important parameters
    param_grid = {
        'hidden_layer_sizes': [(100,), (100, 50), (50, 50, 25)],  # 3 options: single layer, double layer, triple layer
        'activation': ['relu'],  # Only use relu, usually performs best
        'alpha': [0.001, 0.0001],  # Regularization parameter
        'learning_rate_init': [0.001, 0.01],  # Learning rate
        'batch_size': [32, 64],  # Batch size
        'max_iter': [2000]  # Increase iterations to ensure convergence
    }

    mlp = MLPRegressor(
        random_state=42,
        solver='adam',
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        tol=1e-4,
        verbose=False
    )

    # Use 10-fold cross-validation
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=mlp,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        scoring='r2',
        verbose=1,  # Reduce detailed output
        return_train_score=True,
        refit=True  # Retrain with best parameters
    )

    total_combinations = 3 * 1 * 2 * 2 * 2 * 1
    total_training_runs = total_combinations * cv.n_splits

    print("Starting MLP model training...")
    print(f"Parameter combinations: {total_combinations}")
    print(f"Cross-validation folds: {cv.n_splits}")
    print(f"Total training runs: {total_training_runs}")

    grid_search.fit(X_train, y_train)

    best_mlp = grid_search.best_estimator_

    # Save model and scaler
    joblib.dump(best_mlp, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

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
y_train_pred = best_mlp.predict(X_train)
y_test_pred = best_mlp.predict(X_test)


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

# Calculate additional statistics
print("\n=== Prediction Statistics ===")
print(
    f"Train predictions - Min: {y_train_pred.min():.4f}, Max: {y_train_pred.max():.4f}, Mean: {y_train_pred.mean():.4f}")
print(f"Test predictions - Min: {y_test_pred.min():.4f}, Max: {y_test_pred.max():.4f}, Mean: {y_test_pred.mean():.4f}")

# Use myscatterplot for plotting
try:
    myscatterplot(
        y_train.values,
        y_train_pred,
        y_test.values,
        y_test_pred,
        modelname="MLP",
        target="PCE",
        plot_height=8,
        savepic=True,
        picname='MLP_PCE_prediction'
    )
    print("Plot saved to: img/MLP_PCE_prediction.png")
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
    plt.title('MLP: Actual vs Predicted PCE', fontsize=16)
    plt.legend(fontsize=12)

    plt.text(0.05, 0.9, f'Train R² = {train_r2:.3f}', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.05, 0.85, f'Test R² = {test_r2:.3f}', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.05, 0.8, f'Test MAE = {test_mae:.3f}', transform=plt.gca().transAxes, fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig("picture_predict/MLP_PCE_prediction.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Custom plot saved to: picture_predict/MLP_PCE_prediction.png")

# Output detailed parameters of the best model
print("\n" + "=" * 60)
print("MLP NEURAL NETWORK DETAILED PARAMETERS")
print("=" * 60)

# Get all parameters of the model
model_params = best_mlp.get_params()

# Display parameters by category
print("\n=== Network Structure Parameters ===")
structure_params = ['hidden_layer_sizes', 'activation', 'solver']
for param in structure_params:
    if param in model_params:
        print(f"{param}: {model_params[param]}")

print("\n=== Training Optimization Parameters ===")
optimization_params = ['learning_rate', 'learning_rate_init', 'max_iter', 'batch_size',
                       'shuffle', 'momentum', 'nesterovs_momentum', 'early_stopping',
                       'validation_fraction', 'n_iter_no_change']
for param in optimization_params:
    if param in model_params:
        print(f"{param}: {model_params[param]}")

print("\n=== Regularization Parameters ===")
regularization_params = ['alpha', 'lbfgs', 'epsilon', 'beta_1', 'beta_2', 'tol']
for param in regularization_params:
    if param in model_params:
        print(f"{param}: {model_params[param]}")

print("\n=== Other Parameters ===")
other_params = ['random_state', 'verbose', 'warm_start', 'power_t']
for param in other_params:
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

# Display model statistical information
print("\n=== Model Statistical Information ===")
print(f"Model Save Path: {MODEL_PATH}")
print(f"Scaler Save Path: {SCALER_PATH}")
print(f"Final Training Iterations: {best_mlp.n_iter_}")
print(f"Final Loss Value: {best_mlp.loss_:.6f}")

# Display neural network structure details
print("\n=== Neural Network Structure Information ===")
print(f"Input Layer Neuron Count: {best_mlp.n_features_in_}")
print(f"Output Layer Neuron Count: 1 (regression task)")
print(f"Hidden Layer Structure: {best_mlp.hidden_layer_sizes}")
print(f"Total Layers: {len(best_mlp.hidden_layer_sizes) + 2} (input layer + hidden layers + output layer)")

# Display neuron count per layer
print("\n=== Neuron Count per Layer ===")
layer_sizes = [best_mlp.n_features_in_] + list(best_mlp.hidden_layer_sizes) + [1]
for i, size in enumerate(layer_sizes):
    layer_type = "Input Layer" if i == 0 else "Output Layer" if i == len(layer_sizes) - 1 else f"Hidden Layer {i}"
    print(f"{layer_type}: {size} neurons")