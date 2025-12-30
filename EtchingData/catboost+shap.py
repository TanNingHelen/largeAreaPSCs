import os
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import shap
import matplotlib.pyplot as plt
import warnings
from matplotlib import rcParams


warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
rcParams.update({'font.size': 10})

# Create output directory
os.makedirs("picture", exist_ok=True)

# Load data
df = pd.read_excel("FinalData.xlsx")
y = df['PCE']
X = df.drop('PCE', axis=1)

# Save original column names (for CatBoost)
original_columns = X.columns.tolist()

# Unify column name format (for other processing)
X.columns = [col.replace(' ', '_') for col in X.columns]

# Load CatBoost model
print("\nLoading CatBoost model...")
catboost_model = CatBoostRegressor()
catboost_model.load_model("models/best_catboost_model.cbm")
print("CatBoost model loaded successfully!")


def calculate_shap_values(X_data, use_original_columns=False):
    """Calculate SHAP values for CatBoost model"""
    if use_original_columns:
        # For CatBoost, use original column names
        X_data_catboost = X_data.copy()
        X_data_catboost.columns = original_columns
        explainer = shap.TreeExplainer(catboost_model)
        shap_values = explainer.shap_values(Pool(X_data_catboost))
    else:
        explainer = shap.TreeExplainer(catboost_model)
        shap_values = explainer.shap_values(Pool(X_data))
    return shap_values


def format_feature_name(feature_name):
    """Format feature names, replacing specific features with more friendly display names"""
    replacements = {
        'FA': 'FA ratio',
        'MA': 'MA ratio',
        'Br': 'Br ratio',
        'Precursor_Solution': 'Pre_Sol',
        'P2etching_Power(W)': 'P2 Power',
        'P1etching_Power(W)': 'P1 Power',
        'P3etching_Power(W)': 'P3 Power',
        'HTL_Passivator': 'HTL-Psvt',
        'HTL-Addictive': 'HTL-Add',
        'P2etching_Power_percentage(%)': 'P2 Power per',
        'total_scribing_line_width(μm)': 'total Width',
        'P2Width(μm)': 'P2 Width',
        'P1Width(μm)': 'P1 Width',
        'P3Width(μm)': 'P3 Width',
        'P1Wavelength(nm)': 'P1 Wavelength',
        'P2Wavelength(nm)': 'P2 Wavelength',
        'P3Wavelength(nm)': 'P3 Wavelength',
        'P1etching_frequency(kHz)': 'P1 Frequency',
        'P2etching_frequency(kHz)': 'P2 Frequency',
        'P3etching_frequency(kHz)': 'P3 Frequency',
        'P1_P2Scribing_Spacing(μm)': 'P1P2 Spacing',
        'P2_P3Scribing_Spacing(μm)': 'P2P3 Spacing',
        'P1etching_Power_percentage(%)': 'P1 Power per',
        'P3etching_Power_percentage(%)': 'P3 Power per',
        'P2Scan_Velocity': 'P2 Velocity',
        'P3Scan_Velocity': 'P3 Velocity',
        'P1Scan_Velocity(mm/s)': 'P1 Velocity',
        'P1Spot_Size(μm)': 'P1 Spot Size',
        'P2Spot_Size(μm)': 'P2 Spot Size',
        'P3Spot_Size(μm)': 'P3 Spot Size'

    }

    # Check if it's an exact match
    if feature_name in replacements:
        return replacements[feature_name]

    # Check if it contains these keywords
    for key, replacement in replacements.items():
        if key in feature_name:
            return feature_name.replace(key, replacement)

    return feature_name


def plot_importance_bar(shap_values, X_data, filename, color):
    """Plot importance bar chart (with custom color)"""
    # Set figure size (8cm wide, height adjusted according to number of features)
    fig_width_cm = 8
    fig_height_cm = 15  # Increase height to accommodate more features and avoid truncated labels
    fig_width_inch = fig_width_cm / 2.54
    fig_height_inch = fig_height_cm / 2.54

    # Create figure and subplot, adjust layout
    fig, ax = plt.subplots(figsize=(fig_width_inch, fig_height_inch))

    # Calculate mean absolute SHAP values
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    features = X_data.columns
    sorted_idx = np.argsort(mean_abs_shap)[-20:]  # Take top20

    # Format feature names
    formatted_features = [format_feature_name(features[i]) for i in sorted_idx]

    # Plot bar chart (using specified color)
    bars = ax.barh(range(len(sorted_idx)), mean_abs_shap[sorted_idx], color=color, height=0.7)
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels(formatted_features, fontname='Times New Roman')
    ax.set_xlabel('mean absolute SHAP value', fontname='Times New Roman', fontsize=10)

    # Set x-axis tick font
    ax.tick_params(axis='x', labelsize=9, width=0.5)
    ax.tick_params(axis='y', labelsize=9, width=0.5)

    # Set axis thickness to 0.5pt
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # Adjust layout to ensure all elements are displayed
    plt.tight_layout(pad=2.0)  # Increase padding

    # Save image as TIFF format, ensure all elements are in the image
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1, format='tiff')
    plt.close()
    print(f"Successfully saved: {filename}")

    # Return feature importance data for subsequent analysis
    importance_df = pd.DataFrame({
        'feature': features,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)

    return importance_df


def plot_laser_scribing_importance(shap_values, X_data, filename):
    """Plot laser scribing parameter importance sorted bar chart"""
    # Define laser scribing feature list - modified: use underscore format feature names
    laser_scribing_features = [
        'P1Wavelength(nm)',
        'P2Wavelength(nm)',
        'P3Wavelength(nm)',
        'total_scribing_line_width(μm)',
        'P1Width(μm)',
        'P2Width(μm)',
        'P3Width(μm)',
        'GFF',
        'P1Scan_Velocity(mm/s)',
        'P1Spot_Size(μm)',  # Changed to underscore
        'P1etching_frequency(kHz)',
        'P1etching_Power(W)',
        'P1etching_Power_percentage(%)',
        'P2Scan_Velocity',
        'P2Spot_Size(μm)',  # Changed to underscore
        'P2etching_frequency(kHz)',
        'P2etching_Power(W)',
        'P2etching_Power_percentage(%)',
        'P3Scan_Velocity',
        'P3Spot_Size(μm)',  # Changed to underscore
        'P3etching_frequency(kHz)',
        'P3etching_Power(W)',
        'P3etching_Power_percentage(%)',
        'P1_P2Scribing_Spacing(μm)',
        'P2_P3Scribing_Spacing(μm)',
        'brand',
        'subsubmodule_number',
        'Type'
    ]

    # Add debugging information: print all laser scribing features and actual features in data
    print("\n=== Laser Scribing Feature Matching Debug ===")
    print("Laser scribing feature list:")
    for feature in laser_scribing_features:
        print(f"  - {feature}")

    print("\nFeatures in data (first 30):")
    for i, feature in enumerate(X_data.columns[:30]):
        print(f"  {i + 1}. {feature}")

    # Find all features related to laser scribing
    print("\nFinding features containing the following keywords:")
    laser_keywords = ['P1', 'P2', 'P3', 'Spot', 'Size', 'Width', 'Wavelength', 'Scan', 'Velocity', 'etching', 'Power',
                      'frequency', 'Spacing']
    for keyword in laser_keywords:
        matching_features = [f for f in X_data.columns if keyword.lower() in f.lower()]
        if matching_features:
            print(f"Keyword '{keyword}': {matching_features}")

    # Filter laser scribing parameters that exist in the data
    available_laser_features = [f for f in laser_scribing_features if f in X_data.columns]
    print(f"\nFound {len(available_laser_features)} laser scribing parameters:")
    for f in available_laser_features:
        print(f"  - {f}")

    # If some features are not found, try using a more flexible search method
    missing_features = [f for f in laser_scribing_features if f not in X_data.columns]
    if missing_features:
        print(f"\nFeatures not directly found: {missing_features}")
        print("Trying partial matching...")
        for missing_feature in missing_features:
            # Remove unit part for matching
            base_name = missing_feature.split('(')[0] if '(' in missing_feature else missing_feature
            matching = [f for f in X_data.columns if base_name.lower() in f.lower()]
            if matching:
                print(f"  For '{missing_feature}', found possible matching features: {matching}")
                # Use first match
                available_laser_features.extend(matching[:1])

    # Set figure size
    fig_width_cm = 10
    fig_height_cm = 12
    fig_width_inch = fig_width_cm / 2.54
    fig_height_inch = fig_height_cm / 2.54

    # Create figure and subplot
    fig, ax = plt.subplots(figsize=(fig_width_inch, fig_height_inch))

    # Calculate mean absolute SHAP values
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    features = X_data.columns

    # Get importance of laser scribing parameters
    laser_importance = []
    for feature in available_laser_features:
        if feature in features:
            idx = list(features).index(feature)
            laser_importance.append((feature, mean_abs_shap[idx]))

    # Sort by importance
    laser_importance.sort(key=lambda x: x[1], reverse=True)

    # Prepare plot data
    laser_features = [item[0] for item in laser_importance]
    laser_values = [item[1] for item in laser_importance]

    # Format feature names
    formatted_laser_features = [format_feature_name(feature) for feature in laser_features]

    # Plot bar chart - modified: add border and adjust style
    bars = ax.barh(range(len(laser_features)), laser_values,
                   color='#1f77b4', height=0.7, edgecolor='black', linewidth=0.8)

    ax.set_yticks(range(len(laser_features)))
    ax.set_yticklabels(formatted_laser_features, fontname='Times New Roman')
    ax.set_xlabel('mean(|SHAP value|)', fontname='Times New Roman', fontsize=10)
    # ax.set_title('Laser Scribing Parameters Importance', fontname='Times New Roman', fontsize=12)

    # Modified: Set axis tick direction inward, width 0.8pt
    ax.tick_params(axis='x', labelsize=9, direction='in', width=0.8)
    ax.tick_params(axis='y', labelsize=9, direction='in', width=0.8)

    # Modified: Set axis border to 0.8pt
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # Adjust layout
    plt.tight_layout(pad=2.0)

    # Save as TIFF format
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1, format='tiff')
    plt.close()
    print(f"Successfully saved laser scribing parameter importance chart: {filename}")

    # Return laser scribing parameter importance data
    laser_importance_df = pd.DataFrame({
        'feature': laser_features,
        'mean_abs_shap': laser_values
    }).sort_values('mean_abs_shap', ascending=False)

    return laser_importance_df


# Color configuration
COLOR_CONFIG = {
    "all": '#046B38',  # Green
    "small_area": '#1f77b4',  # Blue
    "medium_area": '#ff7f0e',  # Orange
    "large_area": '#d62728'  # Red
}

# Analyze all data
print("\nAnalyzing all data...")
# For CatBoost, use original column names
shap_values_all = calculate_shap_values(X, use_original_columns=True)
importance_df_all = plot_importance_bar(
    shap_values_all,
    X,  # Here use modified column names
    "picture/shap_top20_all_catboost.tif",  # Changed to TIFF format
    COLOR_CONFIG["all"]
)

# Plot laser scribing parameter importance chart
print("\nPlotting laser scribing parameter importance chart...")
laser_importance_df = plot_laser_scribing_importance(
    shap_values_all,
    X,
    "picture/laser_scribing_importance.tif"
)

# Group by Active_Area
active_area_col = df.columns[df.columns.str.contains('Active_Area', case=False)][0]  # Find Active_Area column name
print(f"\nActive_Area column name: {active_area_col}")

# Grouping conditions
small_area_mask = (df[active_area_col] >= 1) & (df[active_area_col] < 10)
medium_area_mask = (df[active_area_col] >= 10) & (df[active_area_col] < 100)
large_area_mask = df[active_area_col] >= 100

print(f"Small area group (1-10): {sum(small_area_mask)} samples")
print(f"Medium area group (10-100): {sum(medium_area_mask)} samples")
print(f"Large area group (>=100): {sum(large_area_mask)} samples")

# Analyze small area group (1 <= Active_Area < 10)
if sum(small_area_mask) > 0:
    print("\nAnalyzing small area group (1 <= Active_Area < 10)...")
    X_small = X[small_area_mask]
    y_small = y[small_area_mask]

    # Rebuild Pool data to ensure column name matching
    X_small_original = X_small.copy()
    X_small_original.columns = original_columns
    shap_values_small = calculate_shap_values(X_small_original, use_original_columns=True)

    importance_df_small = plot_importance_bar(
        shap_values_small,
        X_small,
        "picture/shap_top20_small_area_catboost.tif",  # Changed to TIFF format
        COLOR_CONFIG["small_area"]
    )
else:
    print("\nSmall area group (1 <= Active_Area < 10) has no data")
    importance_df_small = None

# Analyze medium area group (10 <= Active_Area < 100)
if sum(medium_area_mask) > 0:
    print("\nAnalyzing medium area group (10 <= Active_Area < 100)...")
    X_medium = X[medium_area_mask]
    y_medium = y[medium_area_mask]

    # Rebuild Pool data to ensure column name matching
    X_medium_original = X_medium.copy()
    X_medium_original.columns = original_columns
    shap_values_medium = calculate_shap_values(X_medium_original, use_original_columns=True)

    importance_df_medium = plot_importance_bar(
        shap_values_medium,
        X_medium,
        "picture/shap_top20_medium_area_catboost.tif",  # Changed to TIFF format
        COLOR_CONFIG["medium_area"]
    )
else:
    print("\nMedium area group (10 <= Active_Area < 100) has no data")
    importance_df_medium = None

# Analyze large area group (Active_Area >= 100)
if sum(large_area_mask) > 0:
    print("\nAnalyzing large area group (Active_Area >= 100)...")
    X_large = X[large_area_mask]
    y_large = y[large_area_mask]

    # Rebuild Pool data to ensure column name matching
    X_large_original = X_large.copy()
    X_large_original.columns = original_columns
    shap_values_large = calculate_shap_values(X_large_original, use_original_columns=True)

    importance_df_large = plot_importance_bar(
        shap_values_large,
        X_large,
        "picture/shap_top20_large_area_catboost.tif",  # Changed to TIFF format
        COLOR_CONFIG["large_area"]
    )
else:
    print("\nLarge area group (Active_Area >= 100) has no data")
    importance_df_large = None

# Save all feature importance data to CSV
importance_df_all.to_csv("picture/feature_importance_all_catboost.csv", index=False)
print("All data feature importance saved to: picture/feature_importance_all_catboost.csv")

# Save laser scribing parameter importance data to CSV
laser_importance_df.to_csv("picture/laser_scribing_importance.csv", index=False)
print("Laser scribing parameter importance saved to: picture/laser_scribing_importance.csv")

if importance_df_small is not None:
    importance_df_small.to_csv("picture/feature_importance_small_area_catboost.csv", index=False)
    print("Small area group feature importance saved to: picture/feature_importance_small_area_catboost.csv")

if importance_df_medium is not None:
    importance_df_medium.to_csv("picture/feature_importance_medium_area_catboost.csv", index=False)
    print("Medium area group feature importance saved to: picture/feature_importance_medium_area_catboost.csv")

if importance_df_large is not None:
    importance_df_large.to_csv("picture/feature_importance_large_area_catboost.csv", index=False)
    print("Large area group feature importance saved to: picture/feature_importance_large_area_catboost.csv")

# Print top 20 most important features for each group
print("\n=== All Data Top 20 Most Important Features ===")
for i, (feature, importance) in enumerate(
        zip(importance_df_all['feature'][:20], importance_df_all['mean_abs_shap'][:20]), 1):
    print(f"{i}. {feature}: {importance:.6f}")

# Print laser scribing parameter importance
print("\n=== Laser Scribing Parameter Importance Ranking ===")
for i, (feature, importance) in enumerate(
        zip(laser_importance_df['feature'], laser_importance_df['mean_abs_shap']), 1):
    print(f"{i}. {feature}: {importance:.6f}")

if importance_df_small is not None:
    print("\n=== Small Area Group (1-10) Top 20 Most Important Features ===")
    for i, (feature, importance) in enumerate(
            zip(importance_df_small['feature'][:20], importance_df_small['mean_abs_shap'][:20]), 1):
        print(f"{i}. {feature}: {importance:.6f}")

if importance_df_medium is not None:
    print("\n=== Medium Area Group (10-100) Top 20 Most Important Features ===")
    for i, (feature, importance) in enumerate(
            zip(importance_df_medium['feature'][:20], importance_df_medium['mean_abs_shap'][:20]), 1):
        print(f"{i}. {feature}: {importance:.6f}")

if importance_df_large is not None:
    print("\n=== Large Area Group (>=100) Top 20 Most Important Features ===")
    for i, (feature, importance) in enumerate(
            zip(importance_df_large['feature'][:20], importance_df_large['mean_abs_shap'][:20]), 1):
        print(f"{i}. {feature}: {importance:.6f}")

print("\n=== Analysis Completed ===")
print("Generated TIFF image files:")
print("- picture/shap_top20_all_catboost.tif")
print("- picture/laser_scribing_importance.tif")
print("- picture/shap_top20_small_area_catboost.tif")
print("- picture/shap_top20_medium_area_catboost.tif")
print("- picture/shap_top20_large_area_catboost.tif")
print("Generated data files:")
print("- picture/feature_importance_all_catboost.csv")
print("- picture/laser_scribing_importance.csv")
print("- picture/feature_importance_small_area_catboost.csv")
print("- picture/feature_importance_medium_area_catboost.csv")
print("- picture/feature_importance_large_area_catboost.csv")