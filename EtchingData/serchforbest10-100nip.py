import pandas as pd
import os


class PCEAnalyzer:
    def __init__(self, data_path="FinalData0711.xlsx", mapping_path="label_mappings/full_mapping_summary.csv"):
        self.data_path = data_path
        self.mapping_path = mapping_path
        self.target_conditions = {
            'HTL': 'Spiro-OMeTAD',
            'ETL': 'SnO2',
            'Metal_Electrode': 'Au',  # Allow multiple values for Metal_Electrode
            'Glass': 'FTO'  # Allow multiple values for Glass
        }
        self.df = None
        self.all_mappings = {}  # Store all field mappings

        self.load_data()
        self.load_all_mappings()  # Load all field mappings

    def load_data(self):
        """Load data (ensure all features are numeric)"""
        self.df = pd.read_excel(self.data_path)
        print(f"Data loaded successfully, total records: {len(self.df)}")
        print("Fields:", list(self.df.columns))

    def load_all_mappings(self):
        """Load mappings for all fields (not just target features)"""
        if not os.path.exists(self.mapping_path):
            raise FileNotFoundError(f"Mapping file not found: {self.mapping_path}")

        mapping_df = pd.read_csv(self.mapping_path)
        for feature in mapping_df['Feature'].unique():
            self.all_mappings[feature] = {
                'original_to_encoded': {},
                'encoded_to_original': {}
            }
            feature_data = mapping_df[mapping_df['Feature'] == feature]

            for _, row in feature_data.iterrows():
                # Bidirectional mapping
                self.all_mappings[feature]['original_to_encoded'][row['Original']] = row['Encoded']
                self.all_mappings[feature]['encoded_to_original'][row['Encoded']] = row['Original']
        print("All field mappings loaded successfully")

    def decode_feature(self, feature_name, encoded_value):
        """Convert encoded value back to original value"""
        if feature_name in self.all_mappings:
            return self.all_mappings[feature_name]['encoded_to_original'].get(encoded_value, str(encoded_value))
        return encoded_value  # If no mapping, return value directly

    def find_top_records(self):
        """Find top 5 records that match the target conditions"""
        # Build query conditions (using encoded values)
        query_parts = []
        for feature, target in self.target_conditions.items():
            if isinstance(target, list):  # If the target is a list (multiple options)
                encoded_vals = [self.all_mappings[feature]['original_to_encoded'].get(val) for val in target]
                if None in encoded_vals:
                    raise ValueError(f"Field {feature} does not have mappings for one or more of the target values {target}")
                query_parts.append(f"{feature} in {encoded_vals}")
            else:
                encoded_val = self.all_mappings[feature]['original_to_encoded'].get(target)
                if encoded_val is None:
                    raise ValueError(f"Field {feature} does not have {target} as a mapped value")
                query_parts.append(f"{feature} == {encoded_val}")

        # Execute query
        query_str = " & ".join(query_parts)
        matched = self.df.query(query_str)

        # Filter Active_Area and take Top 5
        result = matched[
            (matched['Active_Area'] >= 10) &
            (matched['Active_Area'] < 20)
            ].nlargest(5, 'PCE').copy()

        return result

    def print_full_results(self, result_df):
        """Print full results (all fields in original values)"""
        if result_df.empty:
            print("No matching records found")
            return

        print(f"\n=== Matching Conditions ===")
        print(" | ".join([f"{k}={v}" for k, v in self.target_conditions.items()]))
        print(f"Active_Area range: [10, 20)")
        print(f"Found {len(result_df)} matching records, sorted by PCE in descending order:\n")

        for idx, row in result_df.iterrows():
            print(f"🔷 Record {idx} (PCE: {row['PCE']:.2f}%, Active_Area: {row['Active_Area']:.2f})")

            # Print all fields (show original values)
            for col in result_df.columns:
                if col in self.all_mappings:  # Fields with mappings
                    original_val = self.decode_feature(col, row[col])
                    print(f"  {col}: {original_val} (Encoded Value: {row[col]})")
                else:  # Numeric fields
                    print(f"  {col}: {row[col]}")

            print("─" * 50)


if __name__ == "__main__":
    try:
        analyzer = PCEAnalyzer()
        top_records = analyzer.find_top_records()
        analyzer.print_full_results(top_records)


        if not top_records.empty:
            output_df = top_records.copy()
            for col in output_df.columns:
                if col in analyzer.all_mappings:
                    output_df[f"{col}_原始值"] = output_df[col].apply(
                        lambda x: analyzer.decode_feature(col, x)
                    )
            output_df.to_excel("pce_predict/top_matches_with_original_values[10-20)nip.xlsx", index=False)
            print("\nResults have been saved to pce_predict/top_matches_with_original_values[10-20)nip.xlsx")
    except Exception as e:
        print(f"Error: {str(e)}")
