import pandas as pd
import warnings
from catboost import CatBoostRegressor
warnings.filterwarnings('ignore')

def predict_bandgap(element_ratios):
    """
    Predict perovskite Bandgap using trained CatBoost model
    element_ratios: dictionary containing element ratios
    """
    try:
        # Load Bandgap prediction model
        bandgap_model = CatBoostRegressor()
        bandgap_model.load_model('models/best_catboost_bandgap.cbm')
        print("✅ Bandgap model loaded successfully")
    except Exception as e:
        print(f"❌ Bandgap model loading failed: {e}")
        return None

    # Prepare Bandgap prediction features
    bandgap_features = pd.DataFrame({
        'FA': [element_ratios['FA']],
        'MA': [element_ratios['MA']],
        'Cs': [element_ratios['Cs']],
        'I': [element_ratios['I']],
        'Br': [element_ratios['Br']],
        'Cl': [element_ratios['Cl']],
        'Pb': [element_ratios['Pb']]
    })

    print("🔬 Predicting Bandgap using element ratios:")
    print(f"   FA: {element_ratios['FA']:.4f}, MA: {element_ratios['MA']:.4f}, Cs: {element_ratios['Cs']:.4f}")
    print(
        f"   I: {element_ratios['I']:.4f}, Br: {element_ratios['Br']:.4f}, Cl: {element_ratios['Cl']:.4f}, Pb: {element_ratios['Pb']:.4f}")

    # Predict Bandgap
    try:
        predicted_bandgap = bandgap_model.predict(bandgap_features)[0]
        print(f"   📊 Predicted Bandgap: {predicted_bandgap:.4f} eV")
        return predicted_bandgap
    except Exception as e:
        print(f"❌ Bandgap prediction failed: {e}")
        return None


def prepare_sample_data(sample_data, mapping_df, historical_data, use_predicted_bandgap=True, predicted_bandgap=None):
    if use_predicted_bandgap and predicted_bandgap is not None:
        # Use predicted Bandgap value
        sample_data['Bandgap'] = predicted_bandgap
        print(f"✅ Using predicted Bandgap: {sample_data['Bandgap']:.4f} eV")
    elif use_predicted_bandgap:
        # Predict Bandgap (using existing element ratios)
        print("\n🔬 Starting Bandgap prediction...")
        element_ratios = {
            'FA': sample_data['FA'],
            'MA': sample_data['MA'],
            'Cs': sample_data['Cs'],
            'I': sample_data['I'],
            'Br': sample_data['Br'],
            'Cl': sample_data['Cl'],
            'Pb': sample_data['Pb']
        }

        predicted_bandgap = predict_bandgap(element_ratios)

        if predicted_bandgap is not None:
            # Add predicted Bandgap to features
            sample_data['Bandgap'] = predicted_bandgap
            print(f"✅ Added predicted Bandgap: {predicted_bandgap:.4f} eV")
        else:
            # If Bandgap prediction fails, use sum of element ratios as alternative
            element_cols = ['Cs', 'MA', 'FA', 'I', 'Br', 'Cl', 'Pb']
            sample_data['Bandgap'] = sum(sample_data[col] for col in element_cols)
            print(f"⚠️  Bandgap prediction failed, using sum of element ratios: {sample_data['Bandgap']:.4f}")
    else:
        # Use given Bandgap value
        sample_data['Bandgap'] = 1.6039
        print(f"✅ Using given Bandgap: {sample_data['Bandgap']:.4f} eV")

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


def predict_pce_for_new_samples():
    """
    Predict PCE for new experimental data using trained CatBoost model
    """
    # 1. Load PCE prediction model
    try:
        # Use CatBoost's load_model method to load .cbm file
        model = CatBoostRegressor()
        model.load_model('models/best_catboost_model.cbm')
        print("✅ CatBoost PCE model loaded successfully")

        # Print model information
        print(f"📋 Model feature count: {model.feature_count_ if hasattr(model, 'feature_count_') else 'unknown'}")

    except Exception as e:
        print(f"❌ CatBoost PCE model loading failed: {e}")
        return None

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

    # 4. Prepare raw data for prediction
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
        'Perovskite': '(FA0.98MA0.02)0.95Cs0.05Pb(I0.98Br0.02)3',
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
        'Br': 0.06,
        'Pb': 1.0,
        'Cl': 0
    }

    # Store prediction results
    all_results = {}

    print("=" * 60)
    print("🎯 First dataset prediction (baseline configuration)")
    print("=" * 60)

    # Prepare first dataset (with bandgap prediction)
    sample1_processed = prepare_sample_data(sample1_data, mapping_df, historical_data, use_predicted_bandgap=True)

    # Predict PCE
    try:
        pce_prediction1 = model.predict(sample1_processed)[0]
        predicted_bandgap = sample1_data.get('Bandgap', None)

        print(f"\n🎯 Prediction results:")
        print(f"   Predicted PCE: {pce_prediction1:.2f} %")
        if predicted_bandgap is not None:
            print(f"   Predicted Bandgap: {predicted_bandgap:.4f} eV")
        all_results['sample1'] = {'pce': pce_prediction1, 'bandgap': predicted_bandgap}
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        all_results['sample1'] = {'pce': None, 'bandgap': None}

    return all_results


# Main function
if __name__ == "__main__":
    print("=== Perovskite Solar Cell PCE Prediction System (CatBoost) ===\n")
    results = predict_pce_for_new_samples()

    if results:
        print("\n" + "=" * 60)
        print("=" * 60)

        config_names = {
            'sample1': "HTL = NiOx, HTL-2 = Me-4PACz, HTL-Addictive = DMPU, ETL_Passivator = null",
        }

        for sample_key, result in results.items():
            if result['pce'] is not None:
                config_name = config_names.get(sample_key, "Baseline configuration")
                print(f"Configuration details:")
                print(f"  {config_name}")
                print(f"  PCE = {result['pce']:.2f} %")
                print(f"  Bandgap = {result['bandgap']:.4f} eV")

                # Provide performance evaluation
                if result['pce'] > 20:
                    print("  ⭐ Excellent performance!")
                elif result['pce'] > 18:
                    print("  👍 Good performance!")
                else:
                    print("  💡 Suggested to further optimize process parameters!")
            else:
                print(f"Prediction failed")

            print()

        # Bandgap reference information
        if results['sample1']['bandgap'] is not None:
            print(f"🔬 Bandgap information: {results['sample1']['bandgap']:.4f} eV")
            if results['sample1']['bandgap'] < 1.5:
                print("   💡 Low Bandgap, potentially suitable for tandem cell applications")
            elif results['sample1']['bandgap'] > 1.7:
                print("   💡 High Bandgap, may achieve higher open-circuit voltage")
            else:
                print("   💡 Moderate Bandgap, suitable for single-junction cell applications")