import numpy as np
import os
import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, WhiteKernel,
                                              ConstantKernel as C, DotProduct)
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler
import pandas as pd
from letsplotT import myscatterplot
import matplotlib

matplotlib.use('Agg')


# Create output directories
os.makedirs("img", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Data loading and preprocessing
df = pd.read_excel(r"FinalDataAll.xlsx")
Y = df['PCE']
X = df.drop(['PCE'], axis=1)

# Split train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=12)


# Use RobustScaler for feature standardization, more robust to outliers
scaler_X = RobustScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Standardize target variable as well
scaler_y = RobustScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# Save scalers
joblib.dump(scaler_X, "models/gpr_scaler_X.pkl")
joblib.dump(scaler_y, "models/gpr_scaler_y.pkl")
print("Scalers saved to models/ directory")

# Model loading/training logic
MODEL_PATH = "models/best_gpr_model.pkl"

if os.path.exists(MODEL_PATH):
    print("Loading pre-trained model...")
    best_gpr = joblib.load(MODEL_PATH)
else:
    print("Training new model with optimized parameter grid...")

    # More concise but effective kernel function configurations
    kernels = [
        # Simple but effective kernel function combinations
        C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
        WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 1e1)),

        C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) +
        WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 1e1)),

        C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) +
        WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 1e1)),

        # Simpler kernel functions, reducing overfitting risk
        RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
        WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 1e1)),

        Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) +
        WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 1e1)),
    ]

    # Parameter grid more focused on regularization
    param_grid = {
        'kernel': kernels,
        'alpha': [1e-3, 1e-2, 1e-1, 0.5, 1.0],  # Increase larger alpha values to enhance regularization
        'n_restarts_optimizer': [0, 1, 2]  # Reduce optimizer restart times to lower overfitting risk
    }

    # Use stratified 10-fold cross-validation
    grid_search = GridSearchCV(
        GaussianProcessRegressor(random_state=42, normalize_y=True),  # Use normalize_y=True
        param_grid,
        cv=10,
        n_jobs=-1,
        scoring='r2',
        verbose=2
    )

    # Train with standardized data
    print("Starting optimized grid search with scaled data...")
    grid_search.fit(X_train_scaled, y_train_scaled)
    best_gpr = grid_search.best_estimator_
    joblib.dump(best_gpr, MODEL_PATH)

    print("\n=== Best Parameters ===")
    print(grid_search.best_params_)
    print(f"Best CV R²: {grid_search.best_score_:.4f}")

    # Save all cross-validation results
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results.to_csv("models/gpr_cv_results.csv", index=False)
    print("Cross-validation results saved to models/gpr_cv_results.csv")

# Prediction (need to standardize features first, then inverse-transform prediction results)
# Training set prediction
y_train_scaled_pred = best_gpr.predict(X_train_scaled)
y_train_pred = scaler_y.inverse_transform(y_train_scaled_pred.reshape(-1, 1)).ravel()

# Test set prediction
y_test_scaled_pred = best_gpr.predict(X_test_scaled)
y_test_pred = scaler_y.inverse_transform(y_test_scaled_pred.reshape(-1, 1)).ravel()


def print_metrics(y_true, y_pred, name):
    r = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\n{name} Metrics:")
    print(f"R: {r:.4f}  R²: {r2:.4f}  RMSE: {rmse:.4f}  MAE: {mae:.4f}")


print_metrics(y_train, y_train_pred, "Training Set")
print_metrics(y_test, y_test_pred, "Test Set")

# Calculate overfitting degree
overfitting_degree = (r2_score(y_train, y_train_pred) - r2_score(y_test, y_test_pred))
print(f"\nOverfitting Degree: {overfitting_degree:.4f}")

# If overfitting is severe, recommend using ensemble methods or simpler models
if overfitting_degree > 0.2:
    print("Warning: Significant overfitting detected. Consider using ensemble methods or simpler models.")

# Use original myscatterplot parameter call
try:
    y_train_values = y_train.values if hasattr(y_train, 'values') else y_train
    y_test_values = y_test.values if hasattr(y_test, 'values') else y_test

    myscatterplot(
        y_train_values, y_train_pred,
        y_test_values, y_test_pred,
        modelname="GPR",
        target="PCE",
        plot_height=8,
        savepic=True,
        picname="GPR_PCE_prediction"
    )
    print("Plot successfully saved to img/GPR_PCE_prediction.png")
except Exception as e:
    print(f"\nPlotting Error: {str(e)}")
    print("Please check letsplotT module requirements")

# Model information output
print("\n=== Model Summary ===")
print(f"Best Kernel: {best_gpr.kernel_}")
print(f"Model saved: {MODEL_PATH}")
print(f"Scaler X saved: models/gpr_scaler_X.pkl")
print(f"Scaler y saved: models/gpr_scaler_y.pkl")