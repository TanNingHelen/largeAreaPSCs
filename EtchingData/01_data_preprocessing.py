import numpy as np
import pandas as pd
import re
import warnings
from sklearn.preprocessing import OneHotEncoder

warnings.simplefilter(action='ignore', category=FutureWarning)

# Constant column names
constantHTL = 'HTL'
constantHTL2 = 'HTL-2'
constantStructure = 'Structure'
constantHTL_Passivator = 'HTL_Passivator'
constantHTL_Addictive = 'HTL-Addictive'
constantETL = 'ETL'
constantETL2 = 'ETL-2'
constantETL_Passivator = 'ETL_Passivator'
constantETL_Addictive = 'ETL-Addictive'
constantMetalElectrode = 'Metal_Electrode'
constantGlass = 'Glass'
constantPerovskite = 'Perovskite'
constantPrecursorSolutionAddictive = 'Precursor_Solution_Addictive'
constantPrecursorSolution = 'Precursor_Solution'
constantAntisolvent = 'Antisolvent'
constantAnnealingTemperature1 = 'Annealing_Temperature1'
constantAnnealingTemperature2 = 'Annealing_Temperature2'
constantAnnealingTimeMin = 'Annealing_Time1'
constantAnnealingTime2 = 'Annealing_Time2'
constantDepositionMethod = 'Deposition_Method'

constantP1WavelengthNm = 'P1Wavelength(nm)'
constantP2WavelengthNm = 'P2Wavelength(nm)'
constantP3WavelengthNm = 'P3Wavelength(nm)'
constantTotalScribingLineWidth = 'total_scribing_line_width(μm)'
constantP1Wide = 'P1Width(μm)'
constantP2Wide = 'P2Width(μm)'
constantP3Wide = 'P3Width(μm)'
constantType = 'Type'
constantGFF = 'GFF'
constantsub = 'submodule_number'

constantP1v = 'P1Scan_Velocity(mm/s)'
constantP1frequency = 'P1etching_frequency(kHz)'
constantP1SpotSize = 'P1Spot Size(μm)'
constantP1Power = 'P1etching_Power(W)'
constantP1Powerpercent = 'P1etching_Power_percentage(%)'
constantP2v = 'P2Scan_Velocity'
constantP2frequency = 'P2etching_frequency(kHz)'
constantP2SpotSize = 'P2Spot Size(μm)'
constantP2Power = 'P2etching_Power(W)'
constantP2Powerpercent = 'P2etching_Power_percentage(%)'
constantP3v = 'P3Scan_Velocity'
constantP3frequency = 'P3etching_frequency(kHz)'
constantP3SpotSize = 'P3Spot Size(μm)'
constantP3Power = 'P3etching_Power(W)'
constantP3Powerpercent = 'P3etching_Power_percentage(%)'
constantP1P2 = 'P1_P2Scribing_Spacing(μm)'
constantP2P3 = 'P2_P3Scribing_Spacing(μm)'
constantbrand = 'brand'

# ----- Data Processing -----

# Step 1: Load the target dataset
df = pd.read_excel("OriginData.xlsx")

# Do not process rows (since we will fill missing values)

# Process columns

# Fill missing values in HTL column with '0'
df[constantHTL] = df[constantHTL].fillna('0')

# Fill missing values in HTL-2 column with '0'
df[constantHTL2] = df[constantHTL2].fillna('0')

# Fill missing values in HTL-Passivator column with '0'
df[constantHTL_Passivator] = df[constantHTL_Passivator].fillna('0')

# Fill missing values in HTL-Addictive column with '0'
df[constantHTL_Addictive] = df[constantHTL_Addictive].fillna('0')

# Fill missing values in ETL column with '0'
df[constantETL] = df[constantETL].fillna('0')

# Fill missing values in ETL-2 column with '0'
df[constantETL2] = df[constantETL2].fillna('0')

# Fill missing values in ETL-Passivator column with '0'
df[constantETL_Passivator] = df[constantETL_Passivator].fillna('0')

# Fill missing values in ETL-Addictive column with '0'
df[constantETL_Addictive] = df[constantETL_Addictive].fillna('0')

# Fill missing values in Metal_Electrode column with '0'
df[constantMetalElectrode] = df[constantMetalElectrode].fillna('0')

# Fill missing values in Perovskite column with '0'
df[constantPerovskite] = df[constantPerovskite].fillna('0')

# Fill missing values in Precursor_Solution column with '0'
df[constantPrecursorSolution] = df[constantPrecursorSolution].fillna('0')

# Fill missing values in Precursor_Solution_Addictive column with '0'
df[constantPrecursorSolutionAddictive] = df[constantPrecursorSolutionAddictive].fillna('0')

# Fill missing values in Antisolvent column with '0'
df[constantAntisolvent] = df[constantAntisolvent].fillna('0')

# Get mode for Annealing_Time1 and Annealing_Temperature1
mode_AnnealingTimeMin = df[constantAnnealingTimeMin].mode()
mode_ConstantAnnealingTemperature1 = df[constantAnnealingTemperature1].mode()

# Fill missing values with mode
df[constantAnnealingTimeMin] = df[constantAnnealingTimeMin].fillna(mode_AnnealingTimeMin.iloc[0])
df[constantAnnealingTemperature1] = df[constantAnnealingTemperature1].fillna(mode_ConstantAnnealingTemperature1.iloc[0])

# Fill missing values in Annealing_Temperature2 with '0'
df[constantAnnealingTemperature2] = df[constantAnnealingTemperature2].fillna('0')

# Fill missing values in Annealing_Time2 with '0'
df[constantAnnealingTime2] = df[constantAnnealingTime2].fillna('0')

# Handle Deposition_Method using mode
mode_constantDepositionMethod = df[constantDepositionMethod].mode()
print('Mode of Deposition Method: ')
print(mode_constantDepositionMethod.iloc[0])
df[constantDepositionMethod] = df[constantDepositionMethod].fillna(mode_constantDepositionMethod.iloc[0])

# Fill missing values in P1Wavelength(nm) with '0'
df[constantP1WavelengthNm] = df[constantP1WavelengthNm].fillna('0')

# Fill missing values in P2Wavelength(nm) with '0'
df[constantP2WavelengthNm] = df[constantP2WavelengthNm].fillna('0')

# Fill missing values in P3Wavelength(nm) with '0'
df[constantP3WavelengthNm] = df[constantP3WavelengthNm].fillna('0')

# Fill missing values in total_scribing_line_width(μm) with '0'
df[constantTotalScribingLineWidth] = df[constantTotalScribingLineWidth].fillna('0')

# Fill missing values in P1Width(μm) with '0'
df[constantP1Wide] = df[constantP1Wide].fillna('0')

# Fill missing values in P2Width(μm) with '0'
df[constantP2Wide] = df[constantP2Wide].fillna('0')

# Fill missing values in P3Width(μm) with '0'
df[constantP3Wide] = df[constantP3Wide].fillna('0')

# Fill missing values in Type with '0'
df[constantType] = df[constantType].fillna('0')

# Fill missing values in submodule_number with '0'
df[constantsub] = df[constantsub].fillna('0')

# Fill missing values in P1 scan velocity with '0'
df[constantP1v] = df[constantP1v].fillna('0')
# Fill missing values in P1 etching frequency with '0'
df[constantP1frequency] = df[constantP1frequency].fillna('0')
# Fill missing values in P1 spot size with '0'
df[constantP1SpotSize] = df[constantP1SpotSize].fillna('0')
# Fill missing values in P1 power with '0'
df[constantP1Power] = df[constantP1Power].fillna('0')
# Fill missing values in P1 power percentage with '0'
df[constantP1Powerpercent] = df[constantP1Powerpercent].fillna('0')

# Fill missing values in P2 scan velocity with '0'
df[constantP2v] = df[constantP2v].fillna('0')
# Fill missing values in P2 etching frequency with '0'
df[constantP2frequency] = df[constantP2frequency].fillna('0')
# Fill missing values in P2 spot size with '0'
df[constantP2SpotSize] = df[constantP2SpotSize].fillna('0')
# Fill missing values in P2 power with '0'
df[constantP2Power] = df[constantP2Power].fillna('0')
# Fill missing values in P2 power percentage with '0'
df[constantP2Powerpercent] = df[constantP2Powerpercent].fillna('0')

# Fill missing values in P3 scan velocity with '0'
df[constantP3v] = df[constantP3v].fillna('0')
# Fill missing values in P3 etching frequency with '0'
df[constantP3frequency] = df[constantP3frequency].fillna('0')
# Fill missing values in P3 spot size with '0'
df[constantP3SpotSize] = df[constantP3SpotSize].fillna('0')
# Fill missing values in P3 power with '0'
df[constantP3Power] = df[constantP3Power].fillna('0')
# Fill missing values in P3 power percentage with '0'
df[constantP3Powerpercent] = df[constantP3Powerpercent].fillna('0')

# Fill missing values in P1-P2 scribing spacing with '0'
df[constantP1P2] = df[constantP1P2].fillna('0')
# Fill missing values in P2-P3 scribing spacing with '0'
df[constantP2P3] = df[constantP2P3].fillna('0')
# Fill missing values in laser brand with '0'
df[constantbrand] = df[constantbrand].fillna('0')

# Identify columns that still contain null or empty string values
null_columns = [col for col in df.columns if df[col].isnull().any() or df[col].eq('').any()]
print("Columns containing null or empty values:")
print(null_columns)

# Initialize OneHotEncoder
encoder = OneHotEncoder(
    sparse_output=False,      # Return dense array
    handle_unknown='ignore',  # Ignore unknown categories during transform
    drop='first'              # Avoid dummy variable trap by dropping first category
)

# Perform one-hot encoding on the 'Structure' column
encoded_array = encoder.fit_transform(df[[constantStructure]])

# Save the processed data to an Excel file
df.to_excel('GeneralNullFilled.xlsx', index=False)