import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl

# ============== Configuration Section ==============
rcParams['font.family'] = 'Times New Roman'
rcParams['axes.unicode_minus'] = False
rcParams.update({'font.size': 12})

# Feature name mapping dictionary
FEATURE_ALIASES = {
    'FA': 'FA ratio',
    'MA': 'MA ratio',
    'Precursor_Solution_Addictive': 'Pre-Sol-Add',
    'Annealing_Temperature1': 'Annealing Tp1',
    'Precursor_Solution': 'Pre-Sol',
    'HTL_Passivator': 'HTL-Psvt',
    'Br': 'Br ratio',
    'Annealing_Time1': 'Annealing T1',
    'submodule_number': 'sub PSMs Num',
    'total_scribing_line_width(μm)': 'total Width',
    'Active_Area': 'Active-Area',
    'ETL_Passivator': 'ETL-Psvt',
    'Cs':'Cs ratio'

}

# Model weight configuration (based on R² performance)
MODEL_WEIGHTS = {
    'lgbm': 0.7446,
    'rf': 0.6892,
    'catboost': 0.6762,
    'xgboost': 0.7630
}

# Normalize weights
total_weight = sum(MODEL_WEIGHTS.values())
for model in MODEL_WEIGHTS:
    MODEL_WEIGHTS[model] /= total_weight

print("Model weight configuration:")
for model, weight in MODEL_WEIGHTS.items():
    print(f"  {model}: {weight:.4f}")

# ============== Read data and mapping file ==============
df = pd.read_excel("FinalDataAll.xlsx")
mapping_df = pd.read_csv("label_mappings/full_mapping_summary.csv")

# Get deposition method mapping relationship
deposition_mapping_df = mapping_df[mapping_df['Feature'] == 'Deposition_Method']
method_mapping = dict(zip(deposition_mapping_df['Original'], deposition_mapping_df['Encoded']))
# Create reverse mapping dictionary (encoded value to original name)
reverse_method_mapping = dict(zip(deposition_mapping_df['Encoded'], deposition_mapping_df['Original']))

# Define three deposition methods to analyze
target_methods = ['slot die-coating', 'spin-coating', 'blade-coating']
target_encoded = [method_mapping[method] for method in target_methods]

# Modification: Filter data with Active_Area < 10
filtered_df = df[df['Active_Area'] < 10].copy()
print(f"Filtered data size: {len(filtered_df)} records (Active_Area < 10)")

# ============== Calculate counts for each deposition method ==============
print("Calculating counts for each deposition method...")

# Count number of each deposition method
deposition_counts = filtered_df['Deposition_Method'].value_counts().reset_index()
deposition_counts.columns = ['Encoded_Method', 'Count']

# Map encoded values back to original names
deposition_counts['Method_Name'] = deposition_counts['Encoded_Method'].map(reverse_method_mapping)

# Print statistical results
print("\n=== Counts of each deposition method (Active_Area < 10) ===")
for _, row in deposition_counts.iterrows():
    print(f"{row['Method_Name']} (code: {row['Encoded_Method']}): {row['Count']} records")

# Save statistical results to CSV
deposition_counts.to_csv('deposition_methods_count_active_area_lt_10.csv', index=False)
print("\nDeposition method count statistics saved to: deposition_methods_count_active_area_lt_10.csv")

# ============== Prepare data ==============
print("\nPreparing data...")

# Prepare complete features and target variable
X_full = filtered_df.drop(['PCE'], axis=1)
y_full = filtered_df['PCE']

# ============== Load pre-trained models ==============
print("Loading pre-trained ensemble models...")

# Define model paths
model_paths = {
    'xgboost': "models/best_xgb_model.pkl",
    'lgbm': "models/best_lgbm_model.pkl",
    'rf': "models/best_rf_model.pkl",
    'catboost': "models/best_catboost_model.pkl"  # Based on your provided path
}

# Load all models
models = {}
for model_name, model_path in model_paths.items():
    if not os.path.exists(model_path):
        print(f"Warning: Cannot find model file: {model_path}")
        continue

    try:
        model = joblib.load(model_path)
        models[model_name] = model
        print(f"✅ {model_name} model loaded successfully: {model_path}")
    except Exception as e:
        print(f"❌ Error loading {model_name} model: {e}")

if not models:
    raise FileNotFoundError("No models loaded successfully!")


# ============== SHAP calculation ==============
def calculate_shap_values(model, X, model_type):
    """Calculate SHAP values for a single model"""
    try:
        print(f"Calculating SHAP for {model_type}...")

        # Choose appropriate explainer based on model type
        if model_type in ['xgboost', 'lgbm', 'rf', 'catboost']:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model)

        # Calculate SHAP values
        shap_values_obj = explainer(X)
        shap_values = shap_values_obj.values

        # Handle multi-dimensional SHAP values
        if shap_values.ndim == 3:
            if shap_values.shape[2] == 1:
                shap_values = shap_values[:, :, 0]
            else:
                print(f"Warning: SHAP values are three-dimensional (shape: {shap_values.shape}), taking first dimension.")
                shap_values = shap_values[:, :, 0]

        return shap_values

    except Exception as e:
        print(f"{model_type} SHAP calculation failed: {str(e)}")
        return None


def calculate_ensemble_shap(X_method, feature_names):
    """Calculate weighted SHAP values for ensemble model"""
    ensemble_shap_values = None
    total_weight = 0

    for model_name, model in models.items():
        if model_name not in MODEL_WEIGHTS:
            continue

        # Ensure data column order matches model expectations
        try:
            if hasattr(model, 'get_booster'):  # XGBoost
                expected_features = model.get_booster().feature_names
            elif hasattr(model, 'feature_name_'):  # LightGBM
                expected_features = model.feature_name_
            else:
                expected_features = feature_names

            if expected_features:
                X_aligned = X_method.reindex(columns=expected_features)
            else:
                X_aligned = X_method.reindex(columns=feature_names)
        except:
            X_aligned = X_method.reindex(columns=feature_names)

        # Calculate SHAP values for current model
        shap_values = calculate_shap_values(model, X_aligned, model_name)

        if shap_values is not None:
            weight = MODEL_WEIGHTS[model_name]

            if ensemble_shap_values is None:
                ensemble_shap_values = shap_values * weight
            else:
                ensemble_shap_values += shap_values * weight

            total_weight += weight

    if ensemble_shap_values is not None and total_weight > 0:
        # Normalize
        ensemble_shap_values /= total_weight
        return ensemble_shap_values
    else:
        return None


# ============== Main program ==============
print("Starting SHAP importance analysis for three deposition methods (using ensemble model, Active_Area < 10)...")

# Get feature names
feature_names = X_full.columns.tolist()
print(f"Number of features used: {len(feature_names)}")

# Calculate SHAP values for each deposition method
shap_results = {}

for method_name, encoded_value in zip(target_methods, target_encoded):
    print(f"\nAnalyzing {method_name} (encoded value: {encoded_value})...")

    # Filter data for current method
    method_data = filtered_df[filtered_df['Deposition_Method'] == encoded_value].copy()

    if len(method_data) < 5:
        print(f"Insufficient data ({len(method_data)} records), skipping {method_name}")
        continue

    X_method = method_data.drop(['PCE'], axis=1)

    print(f"  Data shape for SHAP calculation: {X_method.shape}")

    try:
        # Calculate ensemble SHAP values
        ensemble_shap_values = calculate_ensemble_shap(X_method, feature_names)

        if ensemble_shap_values is not None:
            mean_abs_shap = np.abs(ensemble_shap_values).mean(axis=0)

            shap_results[method_name] = {
                'shap_values': ensemble_shap_values,
                'mean_abs_shap': mean_abs_shap,
                'data_size': len(method_data),
                'features': X_method.columns.tolist()
            }

            print(f"{method_name}: data_size={len(method_data)}, ensemble SHAP calculation completed")
        else:
            print(f"{method_name}: SHAP calculation failed")

    except Exception as e:
        print(f"{method_name} SHAP calculation error: {e}")
        continue

# Create SHAP result table
shap_table_data = []
for method_name, result in shap_results.items():
    features = result['features']
    importances = result['mean_abs_shap']

    if len(features) != len(importances):
        min_len = min(len(features), len(importances))
        features = features[:min_len]
        importances = importances[:min_len]

    for feature, importance in zip(features, importances):
        shap_table_data.append({
            'Deposition_Method': method_name,
            'Feature': feature,
            'SHAP_Importance': importance,
            'Data_Size': result['data_size']
        })

if shap_table_data:
    shap_df = pd.DataFrame(shap_table_data)
    output_csv_file = 'deposition_methods_ensemble_shap_active_area_lt_10.csv'
    shap_df.to_csv(output_csv_file, index=False, encoding='utf-8-sig')
    print(f"\nEnsemble model SHAP analysis results saved to: {output_csv_file}")
else:
    print("\nNo SHAP analysis results generated.")

# ============== Plotting section ==============
if shap_results:
    num_methods = len(shap_results)

    # Dynamically adjust image size
    fig_width = 6 * num_methods  # 6 inches width per subplot
    fig_height = 8  # Fixed height of 8 inches

    # Create figure
    fig, axes = plt.subplots(1, num_methods, figsize=(fig_width, fig_height))

    if num_methods == 1:
        axes = [axes]

    # Modification: Set color to #4d8f74 (green)
    bar_color = '#4d8f74'  # Green

    for i, (method_name, result) in enumerate(shap_results.items()):
        features = result['features']
        importances = result['mean_abs_shap']

        if len(features) != len(importances):
            min_len = min(len(features), len(importances))
            features = features[:min_len]
            importances = importances[:min_len]

        # Remove 'Deposition_Method' feature
        non_depo_method_mask = np.array(features) != 'Deposition_Method'

        if not np.any(non_depo_method_mask):
            print(f"Warning: After removing 'Deposition_Method', {method_name} has no remaining features for plotting.")
            continue

        # Filter features and importance
        filtered_features = np.array(features)[non_depo_method_mask].tolist()
        filtered_importances = importances[non_depo_method_mask]

        # Replace feature names with aliases
        aliased_features = []
        for feature in filtered_features:
            aliased_features.append(FEATURE_ALIASES.get(feature, feature))

        # Get top 15 most important features
        num_top_features = min(15, len(aliased_features))
        importance_df = pd.DataFrame({
            'Feature': aliased_features,
            'Importance': filtered_importances,
            'Original_Feature': filtered_features
        }).sort_values('Importance', ascending=False).head(num_top_features)

        if importance_df.empty:
            print(f"Warning: {method_name} has insufficient features for plotting.")
            continue

        # Draw horizontal bar chart - modification: add 0.8pt border to each bar
        ax = axes[i]
        bars = ax.barh(range(len(importance_df)), importance_df['Importance'],
                       color=bar_color, alpha=0.8, edgecolor='black', linewidth=0.8)

        # Set y-axis labels (using aliases) - modification: increase font size
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['Feature'], fontsize=12)

        # Display x-axis labels and tick values below all subplots - modification: increase font size
        ax.set_xlabel('mean(|SHAP value|)', fontsize=13)

        # Set axis tick direction inward - modification: tick line width 0.8pt
        ax.tick_params(axis='x', direction='in', labelsize=11, width=0.8)
        ax.tick_params(axis='y', direction='in', labelsize=11, width=0.8)

        # Ensure x-axis tick values are displayed below all subplots
        ax.tick_params(axis='x', bottom=True, labelbottom=True)

        # Set subplot title, display method name and data size information - modification: increase font size
        ax.set_title(f'{method_name} (n={result["data_size"]})',
                     fontsize=13, pad=10)

        ax.invert_yaxis()

        # Remove grid dashed lines display
        ax.grid(False)

        # Modification: Thicken axis borders to 0.8 points
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    plt.tight_layout()

    # Check if at least one subplot is drawn
    if any(ax.has_data() for ax in axes):
        output_plot_file = 'deposition_methods_ensemble_shap_importance_active_area_lt_10_no_depo.tif'
        plt.savefig(output_plot_file, dpi=300, bbox_inches='tight', format='tiff')
        print(f"Ensemble model SHAP importance chart saved to: {output_plot_file}")
    else:
        print("No plottable SHAP results (all subplots have no data).")
    plt.close(fig)
else:
    print("No plottable SHAP results.")

# Display most important features for each method
print("\n=== Most important features by method (using ensemble model, Active_Area < 10) ===")
for method_name, result in shap_results.items():
    features = result['features']
    importances = result['mean_abs_shap']

    if len(features) != len(importances):
        min_len = min(len(features), len(importances))
        features = features[:min_len]
        importances = importances[:min_len]

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(10)

    print(f"\n{method_name} (n={result['data_size']}) (using ensemble model):")
    for _, row in importance_df.iterrows():
        feature_alias = FEATURE_ALIASES.get(row['Feature'], row['Feature'])
        print(f"  {feature_alias}: {row['Importance']:.4f}")

print("\nAnalysis completed! (using ensemble model, Active_Area < 10)")