import numpy as np
import os
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Create directory structure
os.makedirs("models", exist_ok=True)
os.makedirs("img", exist_ok=True)

# Load data
df = pd.read_excel("FinalData.xlsx")

# Select specific features: Cs, MA, FA, I, Br
feature_columns = ['Cs', 'MA', 'FA', 'Pb','I', 'Br','Cl']
target_column = 'Bandgap'

# Check if features exist
missing_features = [col for col in feature_columns if col not in df.columns]
if missing_features:
    print(f"❌ Missing features: {missing_features}")
    exit()

print("✅ All required features exist")

# Prepare data
X = df[feature_columns]
y = df[target_column]

print(f"Data shape: {X.shape}")
print(f"Feature list: {feature_columns}")
print(f"Target variable: {target_column}")
print(f"Bandgap statistics: min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}")

# Display feature statistics
print("\n=== Feature Statistics ===")
for col in feature_columns:
    print(f"{col}: min={X[col].min():.4f}, max={X[col].max():.4f}, mean={X[col].mean():.4f}")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

# Model path
MODEL_PATH = "models/best_catboost_bandgap.cbm"

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    r = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r, r2, mae, rmse

# Check if model exists
model_exists = os.path.exists(MODEL_PATH)
if model_exists:
    try:
        model = CatBoostRegressor()
        model.load_model(MODEL_PATH)
        print("✅ Loading pre-trained CatBoost Bandgap model...")
    except Exception as e:
        print(f"❌ Failed to load existing model: {str(e)}, retraining...")
        model_exists = False

if not model_exists:
    print("🚀 Training new CatBoost Bandgap model...")

    # Simplified parameter grid (few features, no need for complex parameters)
    param_grid = {
        'iterations': [300, 500, 800],
        'depth': [4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5],
        'random_strength': [1, 2],
    }

    # Use cross-validation to find best parameters
    best_score = -np.inf
    best_params = None
    best_model = None

    print("🔍 Performing parameter search...")

    for iterations in param_grid['iterations']:
        for depth in param_grid['depth']:
            for lr in param_grid['learning_rate']:
                for l2 in param_grid['l2_leaf_reg']:
                    for random_strength in param_grid['random_strength']:

                        model = CatBoostRegressor(
                            iterations=iterations,
                            depth=depth,
                            learning_rate=lr,
                            l2_leaf_reg=l2,
                            random_strength=random_strength,
                            loss_function='RMSE',
                            eval_metric='R2',
                            random_seed=42,
                            verbose=False,
                            thread_count=-1
                        )

                        # Evaluate using cross-validation
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                        mean_score = cv_scores.mean()

                        if mean_score > best_score:
                            best_score = mean_score
                            best_params = {
                                'iterations': iterations,
                                'depth': depth,
                                'learning_rate': lr,
                                'l2_leaf_reg': l2,
                                'random_strength': random_strength
                            }
                            best_model = model

    # Train final model with best parameters
    print(f"🎯 Best parameters: {best_params}")
    print(f"Best cross-validation R²: {best_score:.4f}")

    model = CatBoostRegressor(**best_params, random_seed=42, verbose=100)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50, verbose=False)

    # Save model
    model.save_model(MODEL_PATH)
    print("✅ Model saved successfully")

# Predict results
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate evaluation metrics
train_r, train_r2, train_mae, train_rmse = calculate_metrics(y_train, y_train_pred)
test_r, test_r2, test_mae, test_rmse = calculate_metrics(y_test, y_test_pred)

print("\n" + "=" * 50)
print("=== Final Model Performance - Bandgap Prediction ===")
print("=" * 50)
print("\n=== Training Set Metrics ===")
print(f"R: {train_r:.4f}")
print(f"R²: {train_r2:.4f}")
print(f"MAE: {train_mae:.4f} eV")
print(f"RMSE: {train_rmse:.4f} eV")

print("\n=== Test Set Metrics ===")
print(f"R: {test_r:.4f}")
print(f"R²: {test_r2:.4f}")
print(f"MAE: {test_mae:.4f} eV")
print(f"RMSE: {test_rmse:.4f} eV")

# Calculate overfitting degree
overfit_gap = train_r2 - test_r2
print(f"\n=== Overfitting Analysis ===")
print(f"Training-Test R² gap: {overfit_gap:.4f}")
if overfit_gap > 0.2:
    print("⚠️  Significant overfitting detected!")
elif overfit_gap > 0.1:
    print("ℹ️  Moderate overfitting")
else:
    print("✅ Good generalization performance")

# Save prediction results
print("\n=== Saving Prediction Results ===")
results_df = pd.DataFrame({
    'Actual_Bandgap': pd.concat([y_train, y_test]),
    'Predicted_Bandgap': np.concatenate([y_train_pred, y_test_pred]),
    'Dataset': ['Training'] * len(y_train) + ['Test'] * len(y_test)
})

# Save model performance information
model_info = {
    'train_r2': train_r2,
    'test_r2': test_r2,
    'train_mae': train_mae,
    'test_mae': test_mae,
    'train_rmse': train_rmse,
    'test_rmse': test_rmse,
    'overfit_gap': overfit_gap,
    'features': ','.join(feature_columns)
}


print("\n=== Model Information ===")
print(f"Bandgap model saved to: {MODEL_PATH}")
print(f"Training set sample count: {len(y_train)}")
print(f"Test set sample count: {len(y_test)}")

# Additional statistics
print(f"\n=== Bandgap Prediction Statistics ===")
print(f"Actual Bandgap range: {y.min():.4f} - {y.max():.4f} eV")
print(f"Predicted Bandgap range: {results_df['Predicted_Bandgap'].min():.4f} - {results_df['Predicted_Bandgap'].max():.4f} eV")
print(f"Test set MAE relative error: {test_mae / y.mean() * 100:.2f}%")

print("\n🎉 Bandgap prediction model training completed!")