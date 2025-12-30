import pickle
import pandas as pd
import numpy as np
import re
from collections import defaultdict
import warnings
import joblib


warnings.filterwarnings('ignore')


def prepare_sample_data(sample_data, mapping_df, historical_data, fixed_bandgap=1.6095):
    """
    Prepare sample data and perform preprocessing

    Parameters:
    - sample_data: sample data dictionary
    - mapping_df: mapping dataframe
    - historical_data: historical data
    - fixed_bandgap: fixed Bandgap value
    """
    # Use fixed Bandgap value
    sample_data['Bandgap'] = fixed_bandgap
    print(f"✅ Using fixed Bandgap: {sample_data['Bandgap']:.4f} eV")

    # Create DataFrame for new data
    new_sample = pd.DataFrame([sample_data])

    # Remove Perovskite column (since element ratios and Bandgap are already present)
    if 'Perovskite' in new_sample.columns:
        new_sample = new_sample.drop('Perovskite', axis=1)
        print("✅ Removed Perovskite column, keeping element ratios and Bandgap features")

    # Apply numerical mapping
    categorical_features = [
        'Structure', 'HTL', 'HTL-2', 'HTL_Passivator', 'HTL-Addictive',
        'ETL', 'ETL-2', 'ETL_Passivator', 'ETL-Addictive',
        'Metal_Electrode', 'Glass', 'Precursor_Solution',
        'Precursor_Solution_Addictive', 'Deposition_Method',
        'Antisolvent', 'Type', 'brand'
    ]

    print("\n🔧 Starting feature encoding...")

    for feature in categorical_features:
        if feature in new_sample.columns:
            # Get mapping relationship for this feature
            feature_mapping = mapping_df[mapping_df['Feature'] == feature]

            if len(feature_mapping) > 0:
                # Create mapping dictionary
                mapping_dict = dict(zip(feature_mapping['Original'], feature_mapping['Encoded']))

                # Apply mapping
                original_value = new_sample[feature].iloc[0]

                # Handle null values
                if original_value == '' or pd.isna(original_value):
                    # Find mapping for null values
                    empty_mapping = feature_mapping[feature_mapping['Original'].isna()]
                    if len(empty_mapping) > 0:
                        encoded_value = empty_mapping['Encoded'].iloc[0]
                    else:
                        # If no null mapping, use 0
                        encoded_value = 0
                else:
                    # Normal mapping
                    encoded_value = mapping_dict.get(original_value, 0)

                new_sample[feature] = encoded_value
                print(f"   {feature}: '{original_value}' -> {encoded_value}")
            else:
                print(f"   ⚠️  Feature '{feature}' not found in mapping file, using default value 0")
                new_sample[feature] = 0

    # Ensure all columns are numeric type
    for col in new_sample.columns:
        if new_sample[col].dtype == 'object':
            try:
                new_sample[col] = pd.to_numeric(new_sample[col])
            except:
                print(f"   ⚠️  Cannot convert column '{col}' to numeric type, using 0")
                new_sample[col] = 0

    # Ensure feature order matches training
    try:
        # Get feature order from historical data (excluding target variable PCE)
        expected_features = [col for col in historical_data.columns if col != 'PCE']

        print(f"\n📋 Expected feature count: {len(expected_features)}")

        # Check for missing and extra features
        missing_features = set(expected_features) - set(new_sample.columns)
        extra_features = set(new_sample.columns) - set(expected_features)

        print(f"🔍 Feature matching check:")
        print(f"   Missing features: {missing_features}")
        print(f"   Extra features: {extra_features}")

        # Add missing features
        for feature in missing_features:
            print(f"   ➕ Adding missing feature: {feature} = 0")
            new_sample[feature] = 0

        # Remove extra features
        if extra_features:
            print(f"   ➖ Removing extra features: {extra_features}")
            new_sample = new_sample.drop(columns=list(extra_features))

        # Reorder columns
        new_sample = new_sample[expected_features]
        print(f"   ✅ Feature order adjusted, current feature count: {len(new_sample.columns)}")

    except Exception as e:
        print(f"⚠️  Feature order adjustment failed: {e}")

    return new_sample


def predict_pce_for_first_sample():
    """
    Use three different models to predict PCE for the first raw dataset separately
    """
    # 1. Load three PCE prediction models
    models = {}
    try:
        # Load Random Forest model
        rf_model = joblib.load('models/best_randomforest_model.pkl')
        models['Random Forest'] = rf_model
        print("✅ Random Forest model loaded successfully")
    except Exception as e:
        print(f"❌ Random Forest model loading failed: {e}")
        return None

    try:
        # Load LightGBM model
        lgb_model = joblib.load('models/best_lgbm_model.pkl')
        models['LightGBM'] = lgb_model
        print("✅ LightGBM model loaded successfully")
    except Exception as e:
        print(f"❌ LightGBM model loading failed: {e}")
        return None

    try:
        # Load XGBoost model
        xgb_model = joblib.load('models/best_xgboost_model.pkl')
        models['XGBoost'] = xgb_model
        print("✅ XGBoost model loaded successfully")
    except Exception as e:
        print(f"❌ XGBoost model loading failed: {e}")
        return None

    print(f"📋 Loaded {len(models)} models")

    # 2. Load historical data to get feature structure
    try:
        historical_data = pd.read_excel('FinalData.xlsx')
        print("✅ Historical data loaded successfully")
        print(f"Historical data feature count: {len(historical_data.columns)}")
    except Exception as e:
        print(f"❌ Historical data loading failed: {e}")
        return None

    # 3. Load mapping file
    try:
        mapping_df = pd.read_csv('label_mappings/full_mapping_summary.csv')
        print("✅ Mapping file loaded successfully")
    except Exception as e:
        print(f"❌ Mapping file loading failed: {e}")
        return None

    # 4. Prepare first dataset (raw data)
    sample1_data = {
        'Structure': 'p-i-n',
        'HTL': 'NiOx',
        'HTL-2': 'Me-4PACz',
        'HTL_Passivator': '',
        'HTL-Addictive': 'DMPU+PEAI',
        'ETL': 'C60',
        'ETL-2': 'SnO2',
        'ETL_Passivator': '',
        'ETL-Addictive': '',
        'Metal_Electrode': 'Cu',
        'Glass': 'FTO',
        'Perovskite': '(FA0.98MA0.02)0.95Cs0.05Pb(l0.98Br0.02)3',
        'Active_Area': 12.96,
        'Precursor_Solution': 'DMF:NMP (7:1)',
        'Precursor_Solution_Addictive': '',
        'Deposition_Method': 'blade-coating',
        'Antisolvent': '',
        'Annealing_Temperature1': 120,
        'Annealing_Time1': 25,
        'Annealing_Temperature2': 0,
        'Annealing_Time2': 0,
        'P1Wavelength(nm)': 532,
        'P2Wavelength(nm)': 532,
        'P3Wavelength(nm)': 532,
        'total_scribing_line_width(μm)': 235,
        'P1Width(μm)': 40,
        'P2Width(μm)': 65,
        'P3Width(μm)': 40,
        'GFF': 95.36,
        'Type': 'Series',
        'submodule_number': 6,
        'P1Scan_Velocity(mm/s)': 4000,
        'P1etching_frequency(kHz)': 500,
        'P1Spot Size(μm)': 40,
        'P1etching_Power(W)': 0,
        'P1etching_Power_percentage(%)': 40,
        'P2Scan_Velocity': 2000,
        'P2etching_frequency(kHz)': 500,
        'P2Spot Size(μm)': 40,
        'P2etching_Power(W)': 0,
        'P2etching_Power_percentage(%)': 10,
        'P3Scan_Velocity': 2000,
        'P3etching_frequency(kHz)': 500,
        'P3Spot Size(μm)': 40,
        'P3etching_Power(W)': 0,
        'P3etching_Power_percentage(%)': 9,
        'P1_P2Scribing_Spacing(μm)': 45,
        'P2_P3Scribing_Spacing(μm)': 45,
        'brand': '',
        'Cs': 0.05,
        'MA': 0.02,
        'FA': 0.93,
        'I': 2.94,
        'Br': 0.96,
        'Pb': 1.0,
        'Cl': 0,
        'Bandgap': 1.5284  # Fixed Bandgap value
    }

    # Store all prediction results
    all_predictions = {}

    print("=" * 60)
    print("🎯 First dataset prediction (original configuration)")
    print("=" * 60)
    print("Configuration: HTL = NiOx, HTL-2 = Me-4PACz, HTL-Addictive = DMPU, ETL_Passivator = null")
    print(f"Using fixed Bandgap value: 1.6095 eV")

    # Prepare first dataset
    sample1_processed = prepare_sample_data(sample1_data, mapping_df, historical_data, fixed_bandgap=1.6095)

    # Predict with three models separately
    for model_name, model in models.items():
        try:
            pce_prediction = model.predict(sample1_processed)[0]
            all_predictions[model_name] = pce_prediction
            print(f"\n📊 {model_name} prediction results:")
            print(f"   Predicted PCE: {pce_prediction:.2f} %")

            # Provide performance evaluation
            if pce_prediction > 20:
                print("   ⭐ Excellent performance!")
            elif pce_prediction > 18:
                print("   👍 Good performance!")
            else:
                print("   💡 Suggested to further optimize process parameters!")
        except Exception as e:
            print(f"\n❌ {model_name} prediction failed: {e}")
            all_predictions[model_name] = None

    return all_predictions


# Main function
if __name__ == "__main__":
    print("=== Perovskite Solar Cell PCE Prediction System ===")
    print("Using three models to predict first raw dataset separately")
    print("Configuration: HTL = NiOx, HTL-2 = Me-4PACz, HTL-Addictive = DMPU, ETL_Passivator = null")
    print("Using fixed Bandgap value: 1.5284 eV")
    print("Prediction models: Random Forest, LightGBM, XGBoost")
    print("=" * 60)

    # Predict PCE for first dataset
    predictions = predict_pce_for_first_sample()

    if predictions:
        print("\n" + "=" * 60)
        print("📊 All model prediction results summary")
        print("=" * 60)

        for model_name, pce in predictions.items():
            if pce is not None:
                print(f"{model_name}: {pce:.2f} %")
            else:
                print(f"{model_name}: Prediction failed")

        print("\n" + "=" * 60)
        print("📈 Prediction results statistics")
        print("=" * 60)

        # Calculate statistics
        valid_predictions = [p for p in predictions.values() if p is not None]
        if valid_predictions:
            print(f"Number of prediction models: {len(valid_predictions)}")
            print(f"Average predicted PCE: {np.mean(valid_predictions):.2f} %")
            print(f"Highest predicted PCE: {max(valid_predictions):.2f} %")
            print(f"Lowest predicted PCE: {min(valid_predictions):.2f} %")
            print(f"Predicted PCE range: {max(valid_predictions) - min(valid_predictions):.2f} %")
        else:
            print("All model predictions failed")
    else:
        print("Prediction failed, please check models and data")