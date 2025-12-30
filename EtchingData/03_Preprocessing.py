import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
from sklearn import preprocessing

warnings.filterwarnings('ignore')


def enhanced_preprocessing(df):
    """Improved data preprocessing"""
    df1 = df.copy()
    df1['Original_GFF'] = df1['GFF']
    df1['Filled_GFF'] = (df1['GFF'] == 0).astype(int)

    print(f"Data statistics:")
    print(f"Total samples: {len(df1)}")
    print(f"Samples with GFF not 0: {((df1['GFF'] != 0) & (df1['GFF'].notnull())).sum()}")
    print(f"Samples with GFF equal to 0: {(df1['GFF'] == 0).sum()}")

    # Fix column names
    df1 = df1.rename(columns={
        'HTL-Additive': 'HTL-Addictive',
        'ETL-Additive': 'ETL-Addictive',
        'Precursor_Solution_Additive': 'Precursor_Solution_Addictive'
    })

    # Create label_mappings directory
    os.makedirs("label_mappings", exist_ok=True)

    # Label encoding
    categorical_cols = ['Structure', 'HTL', 'HTL-2', 'HTL_Passivator', 'HTL-Addictive',
                        'ETL', 'ETL-2', 'ETL_Passivator', 'ETL-Addictive',
                        'Metal_Electrode', 'Glass', 'Precursor_Solution',
                        'Precursor_Solution_Addictive', 'Deposition_Method',
                        'Antisolvent', 'Type', 'brand']

    # Store all mapping relationships
    label_mappings = {}

    for col in categorical_cols:
        if col in df1.columns:
            lbl = preprocessing.LabelEncoder()
            df1[col] = lbl.fit_transform(df1[col].astype('str'))

            # Save mapping
            mapping = pd.DataFrame({
                'Original': lbl.classes_,
                'Encoded': range(len(lbl.classes_))
            })
            label_mappings[col] = mapping

            # Save to CSV
            mapping.to_csv(f"label_mappings/{col}_mapping.csv", index=False)
            print(f"Saved mapping for {col} to: label_mappings/{col}_mapping.csv")

    # Save summary of all mappings
    mapping_summary = []
    for col, mapping_df in label_mappings.items():
        temp_df = mapping_df.copy()
        temp_df['Feature'] = col
        mapping_summary.append(temp_df)

    full_mapping = pd.concat(mapping_summary, ignore_index=True)
    full_mapping = full_mapping[['Feature', 'Original', 'Encoded']]  # Adjust column order
    full_mapping.to_csv("label_mappings/full_mapping_summary.csv", index=False)

    print(f"\n=== Mapping file generation completed ===")
    print(f"Total mapped features: {len(label_mappings)}")
    print(f"Total mapping entries: {len(full_mapping)}")
    print(f"Full mapping summary saved to: label_mappings/full_mapping_summary.csv")

    # Print mapping details for Deposition_Method
    if 'Deposition_Method' in label_mappings:
        depo_mapping = label_mappings['Deposition_Method']
        print(f"\nDeposition_Method mapping details:")
        print(f"Unique values: {len(depo_mapping)}")
        for _, row in depo_mapping.iterrows():
            print(f"  {row['Original']} -> {row['Encoded']}")

    return df1


def analyze_feature_importance(df):
    """Analyze feature importance"""
    valid_data = df[(df['GFF'] != 0) & (df['GFF'].notnull())].copy()

    if len(valid_data) < 10:
        return []

    X = valid_data.drop(columns=['GFF', 'Original_GFF', 'Filled_GFF'], errors='ignore')
    y = valid_data['GFF']

    # Select only numeric features for correlation analysis
    numeric_cols = X.select_dtypes(include=[np.number]).columns

    correlations = []
    for col in numeric_cols:
        corr = np.corrcoef(X[col], y)[0, 1]
        if not np.isnan(corr):
            correlations.append((col, abs(corr)))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 15 features correlated with GFF:")
    for i, (col, corr) in enumerate(correlations[:15], 1):
        print(f"  {i:2d}. {col:30} | Correlation: {corr:.4f}")

    # Return top features
    return [col for col, _ in correlations[:15]]


def robust_cross_validation(X, y, model, n_splits=5):
    """Robust cross-validation"""
    try:
        kf = KFold(n_splits=min(n_splits, len(y)), shuffle=True, random_state=42)
        mae_scores = []
        r2_scores = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            model.fit(X_train_scaled, y_train)

            # Predict
            y_pred = model.predict(X_test_scaled)

            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            mae_scores.append(mae)
            r2_scores.append(r2)

        return np.mean(mae_scores), np.std(mae_scores), np.mean(r2_scores), np.std(r2_scores)

    except Exception as e:
        print(f"Cross-validation error: {e}")
        return np.inf, 0, -np.inf, 0


def evaluate_regression_models(df, selected_features):
    """Evaluate regression model performance"""
    valid_data = df[(df['GFF'] != 0) & (df['GFF'].notnull())].copy()

    if len(valid_data) < 20:
        print("Insufficient valid data for reliable evaluation")
        return None, None

    X = valid_data[selected_features]
    y = valid_data['GFF']

    # Define models
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5),
        'SVR': SVR(kernel='rbf', C=1.0),
        'KNN': KNeighborsRegressor(n_neighbors=5)
    }

    results = {}

    for name, model in models.items():
        print(f"Evaluating {name}...")
        mae, mae_std, r2, r2_std = robust_cross_validation(X, y, model)
        results[name] = {
            'MAE': mae,
            'MAE_std': mae_std,
            'R2': r2,
            'R2_std': r2_std,
            'model': model
        }
        print(f"  {name}: MAE = {mae:.3f} ± {mae_std:.3f}, R² = {r2:.3f} ± {r2_std:.3f}")

    # Select best model
    if results:
        best_model_name = max(results.keys(), key=lambda x: results[x]['R2'])
        best_result = results[best_model_name]

        print(f"\nBest model: {best_model_name}")
        print(f"MAE: {best_result['MAE']:.3f} ± {best_result['MAE_std']:.3f}")
        print(f"R²: {best_result['R2']:.3f} ± {best_result['R2_std']:.3f}")

        return best_model_name, results[best_model_name]['model']

    return None, None


def impute_with_best_model(df, selected_features, best_model):
    """Impute missing values using the best model"""
    df_result = df.copy()

    # Separate known and unknown data
    known_data = df_result[(df_result['GFF'] != 0) & (df_result['GFF'].notnull())].copy()
    unknown_data = df_result[df_result['GFF'] == 0].copy()

    if len(unknown_data) == 0:
        print("No data to impute")
        return df_result

    # Prepare training data
    X_train = known_data[selected_features]
    y_train = known_data['GFF']

    # Prepare prediction data
    X_pred = unknown_data[selected_features]

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_pred_scaled = scaler.transform(X_pred)

    # Train model
    best_model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = best_model.predict(X_pred_scaled)

    # Update GFF values
    df_result.loc[df_result['GFF'] == 0, 'GFF'] = y_pred

    print(f"Successfully imputed {len(unknown_data)} GFF values that were 0")
    print(f"Imputed value statistics: min={y_pred.min():.2f}, max={y_pred.max():.2f}, mean={y_pred.mean():.2f}")

    return df_result


def main():
    # Read data
    df = pd.read_excel(r"BandgapDone.xlsx")

    # Preprocessing
    df_processed = enhanced_preprocessing(df)

    # Analyze feature importance
    selected_features = analyze_feature_importance(df_processed)

    if not selected_features:
        print("Cannot select valid features, using simple imputation")
        # Use median imputation
        valid_gff = df_processed[df_processed['GFF'] != 0]['GFF']
        median_gff = valid_gff.median()
        df_processed.loc[df_processed['GFF'] == 0, 'GFF'] = median_gff
        print(f"Using median imputation: {median_gff:.2f}")
    else:
        # Evaluate models
        best_model_name, best_model = evaluate_regression_models(df_processed, selected_features)

        if best_model is not None:
            # Impute with best model
            df_processed = impute_with_best_model(df_processed, selected_features, best_model)
        else:
            # Fallback to median imputation
            valid_gff = df_processed[df_processed['GFF'] != 0]['GFF']
            median_gff = valid_gff.median()
            df_processed.loc[df_processed['GFF'] == 0, 'GFF'] = median_gff
            print(f"Model evaluation failed, using median imputation: {median_gff:.2f}")

    # Post-processing
    cols_to_drop = ['Sn', 'Rb']
    cols_to_drop = [col for col in cols_to_drop if col in df_processed.columns]
    df_processed.drop(columns=cols_to_drop, inplace=True)

    # Remove unnecessary columns: Original_GFF and Filled_GFF
    columns_to_remove = ['Original_GFF', 'Filled_GFF']
    for col in columns_to_remove:
        if col in df_processed.columns:
            df_processed.drop(columns=[col], inplace=True)
            print(f"Removed column: {col}")

    # Save data without removing rows
    df_processed.to_excel(r"FinalDataAll1.xlsx", index=False)
    print("Data without row removal saved to FinalDataAll1.xlsx")

    # Filter rows where all laser features are 0
    laser_cols = ['P1Wavelength(nm)', 'P2Wavelength(nm)', 'P3Wavelength(nm)',
                  'total_scribing_line_width(μm)',
                  'P1Width(μm)', 'P2Width(μm)', 'P3Width(μm)']
    laser_cols = [col for col in laser_cols if col in df_processed.columns]

    # Convert these columns to numeric first
    for col in laser_cols:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

    if laser_cols:
        all_zero_mask = (df_processed[laser_cols] == 0).all(axis=1)

        df_filtered = df_processed[~all_zero_mask]
        removed_count = all_zero_mask.sum()
        print(f"Filtered out {removed_count} rows with all laser features equal to 0")

        # Save filtered data
        df_filtered.to_excel(r"FinalData1.xlsx", index=False)
        print(f"Filtered data saved to FinalData1.xlsx, remaining {len(df_filtered)} rows")
    else:
        # If no laser feature columns, save directly
        df_processed.to_excel(r"FinalData1.xlsx", index=False)
        print("No laser feature columns, data saved to FinalData1.xlsx")

    print("\nData processing completed!")
    print(f"- Full data (without row removal): FinalDataAll1.xlsx")
    print(f"- Filtered data: FinalData1.xlsx")
# The suffix '1' is used to avoid overwriting source files and better reproduce results

if __name__ == "__main__":
    main()