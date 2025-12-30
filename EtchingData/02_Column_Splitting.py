import pandas as pd
import re
import os

def preprocess_formula(formula):
    """Preprocess chemical formula, handle common formatting issues"""
    if pd.isna(formula):
        return formula

    formula = str(formula).strip()

    # Remove invisible characters and special whitespace characters
    formula = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', formula)
    formula = re.sub(r'\s+', '', formula)  # Remove all spaces

    # Handle newline characters
    formula = formula.replace('\n', '').replace('\r', '')

    # Handle superscript numbers
    subscript_map = {
        '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
        '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'
    }
    for sub, normal in subscript_map.items():
        formula = formula.replace(sub, normal)

    # Handle mathematical dots
    formula = formula.replace('⋅', '.')

    # Fix: Handle lowercase l more safely to avoid affecting Cl
    # Replace only when l is not preceded by C and is followed by a number
    formula = re.sub(r'(?<![Cc])l(?=\d)', 'I', formula)

    # Handle square brackets (convert to parentheses)
    formula = formula.replace('[', '(').replace(']', ')')

    # Handle full names of organic cations
    formula = re.sub(r'\[CH\(NH2\)2\]', 'FA', formula)
    formula = re.sub(r'CH\(NH2\)2', 'FA', formula)
    formula = re.sub(r'\[CH3NH3\]', 'MA', formula)
    formula = re.sub(r'CH3NH3', 'MA', formula)

    return formula


def parse_simple_formula(formula):
    """Parse simple chemical formula (no parentheses)"""
    elements = ['Cs', 'MA', 'FA', 'Rb', 'Pb', 'Sn', 'I', 'Br', 'Cl']
    ratios = {ele: 0.0 for ele in elements}

    # Sort elements by length in descending order to match longer names first (like MA, FA)
    sorted_elements = sorted(elements, key=len, reverse=True)

    remaining_formula = formula

    for ele in sorted_elements:
        # Match element and the number following it
        pattern = re.escape(ele) + r'(\d*\.?\d*)'
        match = re.search(pattern, remaining_formula)

        if match:
            num_str = match.group(1)
            if not num_str:
                ratios[ele] = 1.0
            else:
                ratios[ele] = float(num_str)

            # Remove matched part from remaining string
            start, end = match.span()
            remaining_formula = remaining_formula[:start] + remaining_formula[end:]

    return ratios, remaining_formula


def get_element_ratio(composition):
    elements = ['Cs', 'MA', 'FA', 'Rb', 'Pb', 'Sn', 'I', 'Br', 'Cl']
    element_ratio = {ele: 0.0 for ele in elements}

    # Preprocess chemical formula
    composition = preprocess_formula(composition)

    # Handle parentheses content
    if '(' in composition:
        # Match parentheses content and multiplier
        pattern = r"\(([^()]+)\)(\d*\.?\d*)"
        matches = re.findall(pattern, composition)

        for bracket_content, multiplier_str in matches:
            if not multiplier_str:
                multiplier = 1.0
            else:
                multiplier = float(multiplier_str)

            # Parse elements inside parentheses
            bracket_ratios, _ = parse_simple_formula(bracket_content)

            # Multiply ratios inside parentheses by multiplier and accumulate
            for ele in elements:
                element_ratio[ele] += bracket_ratios[ele] * multiplier

            # Remove processed parentheses part from original string
            bracket_pattern = re.escape(f"({bracket_content}){multiplier_str}")
            composition = re.sub(bracket_pattern, '', composition)

    # Parse remaining part (no parentheses)
    remaining_ratios, _ = parse_simple_formula(composition)
    for ele in elements:
        element_ratio[ele] += remaining_ratios[ele]

    # Handle special case: if Pb exists but no explicit ratio, set to 1
    if any(element_ratio[ele] > 0 for ele in ['Cs', 'MA', 'FA', 'Rb']) and \
            any(element_ratio[ele] > 0 for ele in ['I', 'Br', 'Cl']) and \
            element_ratio['Pb'] == 0 and element_ratio['Sn'] == 0:
        element_ratio['Pb'] = 1.0

    return element_ratio


# Process Excel file
def process_perovskite_column(file_path):
    try:
        # Check if input file exists
        if not os.path.exists(file_path):
            print(f"Error: Input file {file_path} does not exist!")
            print(f"Current working directory: {os.getcwd()}")
            return None

        df = pd.read_excel(file_path)
        print(f"Successfully read Excel file: {file_path}")
        print(f"File contains {len(df)} rows of data")

        if 'Perovskite' not in df.columns:
            print("Error: 'Perovskite' column not found in file")
            print("Available columns:", df.columns.tolist())
            return None

        element_data = []
        valid_rows = []
        invalid_formulas = []

        for i, formula in enumerate(df['Perovskite']):
            if pd.isna(formula):
                invalid_formulas.append((i, "Blank value"))
                continue

            formula_str = str(formula).strip()
            try:
                ratio = get_element_ratio(formula_str)
                element_data.append(ratio)
                valid_rows.append(i)
            except Exception as e:
                invalid_formulas.append((i, f"{formula_str} (Error: {str(e)})"))

        # Check if there is valid data
        if len(element_data) == 0:
            print("Error: No chemical formulas successfully parsed!")
            return None

        ratio_df = pd.DataFrame(element_data)

        # Ensure all necessary columns exist
        required_columns = ['Cs', 'MA', 'FA', 'Rb', 'Pb', 'Sn', 'I', 'Br', 'Cl']
        for col in required_columns:
            if col not in ratio_df.columns:
                ratio_df[col] = 0.0

        # Add original data
        result_df = pd.concat([df.iloc[valid_rows].reset_index(drop=True), ratio_df], axis=1)

        print(f"\nProcessing Summary:")
        print(f"Total rows: {len(df)}")
        print(f"Successfully parsed: {len(valid_rows)}")
        print(f"Failed to parse: {len(invalid_formulas)}")
        print(f"Success rate: {len(valid_rows) / len(df) * 100:.1f}%")

        # Special statistics for Cl
        cl_count = (ratio_df['Cl'] > 0).sum()
        print(f"Samples containing Cl element: {cl_count}")
        if cl_count > 0:
            print(f"Cl ratio range: [{ratio_df['Cl'].min():.4f}, {ratio_df['Cl'].max():.4f}]")

        if invalid_formulas:
            print("\nInvalid or unparsable chemical formulas:")
            for row, formula in invalid_formulas[:10]:  # Show only first 10 errors
                print(f"Row {row + 1}: {formula}")
            if len(invalid_formulas) > 10:
                print(f"... and {len(invalid_formulas) - 10} more errors not shown")

        # Calculate ratio sum for verification (not added to final table)
        element_cols = ['Cs', 'MA', 'FA', 'Rb', 'Pb', 'Sn', 'I', 'Br', 'Cl']
        total_ratio_series = result_df[element_cols].sum(axis=1)

        print("\nRatio Sum Verification:")
        print(total_ratio_series.describe().round(4))
        print("\nNote: 'Total_Ratio' is used to verify the original sum of element ratios, not normalized, and will not appear in the final table")

        # Move PCE column to last (if exists)
        if 'PCE' in result_df.columns:
            # Get all columns except PCE
            other_columns = [col for col in result_df.columns if col != 'PCE']
            # Reorder columns, PCE at the end
            result_df = result_df[other_columns + ['PCE']]
            print("\nNote: PCE column has been moved to the last column")

        # Save result (without Total_Ratio column)
        output_file = "perovskite_element_ratios.xlsx"
        result_df.to_excel(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

        # Verify file creation success
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"File created successfully! File size: {file_size} bytes")
            print(f"File location: {os.path.abspath(output_file)}")
        else:
            print("Error: File not found after saving!")
            return None

        return result_df, total_ratio_series

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


# Main execution program
if __name__ == "__main__":
    # File path to be parsed
    file_path = r"GeneralNullFilled.xlsx"

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist!")
        print(f"Current working directory: {os.getcwd()}")
        print("Please ensure the file exists in the current directory")
        exit(1)

    print("Starting perovskite chemical composition analysis...")
    print("=" * 50)

    result_df, total_ratio_series = process_perovskite_column(file_path)

    if result_df is not None:
        print("\nPreview of first 5 rows:")
        # Show only element ratio related columns
        element_cols = ['Cs', 'MA', 'FA', 'Rb', 'Pb', 'Sn', 'I', 'Br', 'Cl']
        display_cols = [col for col in element_cols if col in result_df.columns]
        if display_cols:
            # If PCE column exists, also show it
            if 'PCE' in result_df.columns:
                display_cols.append('PCE')
            print(result_df[display_cols].head().round(4))
        else:
            print(result_df.head())

        # Show basic statistical information
        print("\nElement ratio statistics:")
        element_cols = ['Cs', 'MA', 'FA', 'Rb', 'Pb', 'Sn', 'I', 'Br', 'Cl']
        available_cols = [col for col in element_cols if col in result_df.columns]
        if available_cols:
            stats = result_df[available_cols].describe().round(4)
            print(stats)

        # Verify column order
        print("\nFinal table column order:")
        print(result_df.columns.tolist())

        print("\nProcessing completed!")
    else:
        print("\nProcessing failed, no results generated!")