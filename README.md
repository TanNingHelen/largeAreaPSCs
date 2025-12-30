# ML_for_Perovskite_Laser_Scribing/
## File tree
```text
ML_for_Perovskite_Laser_Scribing/
├── EchingData/                             # laser scribing sub-dataset
│   ├── 01_data_preprocessing.py            # Data cleaning and missing value handling
│   ├── GeneralNullFilled.xlsx              # The data after missing value imputation (excluding GFF)
│   ├── 02_Column_Splitting.py              # Perovskite chemical formula ratio extraction
│   ├── perovskite_element_ratios.xlsx      # Data after extraction of perovskite component chemical formulas
│   ├── BandgapPredict.ipynb                # Predict Bandgap
│   ├── BandgapDS.xlsx                      # Source data for training the bandgap prediction model
│   ├── BandgapDone.xlsx                    # Data with completed bandgap predictions
│   ├── 03_Preprocessing.py                 # Impute missing GFF values to form the final dataset
│   ├── FinalData.xlsx                      # Final data (laser scribing sub-dataset)
│   ├── CatBoost.py                         # CatBoost model training
│   ├── XGBoost.py                          # XGBoost model training
│   ├── LGBM.py                             # LightGBM model training
│   ├── RandomForest.py                     # XGBoost model training
│   ├── label_mappings/                     # Store the mapping table for categorical variables.
│   ├── models/                             # Store model files
│       ├── best_catboost_model.cbm         # Store best CatBoost Model
│       ├── best_lgbm_model.pkl             # Store best LGBM Model  
│       ├── best_randomforest_model.pkl     # Store best RF Model
│       ├── best_xgboost_model.pkl          # Store best XGBoost Model 
│       ├── best_catboost_bandgap.cbm       # Store best CatBoost Model for bandgap prediction
│   ├── img/                                # Store the scatter plot of predicted vs. actual PCE after model training
│   ├── catboost+shap.py                    # Use catboost for shap
│   ├── catboost_Bandgap.py                 # CatBoost training for bandgap prediction
│   ├── catboost+predict.py                 # Use the existing CatBoost model to predict the PCE of a set of experiments
│   ├── 3ModelPredict.py                    # Use the existing RF,LGBM,XGBOost models to predict the PCE of a set of experiments
│   ├── submoduleWidth/                     # Store data related to submodule width prediction.
│       ├── subwidth.ipynb                  # Find best submodule width
│       ├── subcell.xlsx                    # Submodule width data collected from the literature   
│   ├── pinchangefrekencyPredictMore.py     # Predict the optimal dead-zone width range and generate corresponding laser parameter combinations
│   ├── letsplotT.py                        # Used to store utility functions.                    
├── AllData/                                # full dataset
│   ├── FinalDataAll.xlsx                   # Final data (full dataset) 
│   ├── 1.GPR.py                            # GPR model training
│   ├── 2.RandomForest.py                   # RF model training
│   ├── 3.XGBoost.py                        # XGBoost model training
│   ├── 4.CatBoost.py                       # CatBoost model training
│   ├── 5.LGBM.py                           # LightGBM model training
│   ├── 6.MLP.py                            # MLP model training
│   ├── 7.LinearRegression.py               # LR model training
│   ├── 8.SVR.py                            # SVR model training
│   ├── 9.DT.py                             # DT model training
│   ├── models/                             # Store model files
│   ├── deposhap<10.py                      # Importance distribution of the three mainstream deposition methods on small-area data.
│   ├── deposhap10-100.py                   # Importance distribution of the three mainstream deposition methods on medium-area data.
│   ├── deposhap>100.py                     # Importance distribution of the three mainstream deposition methods on large-area data.
└── README.md                       # 本文件
```

## Version Notes
```text
Python:3.12.2
catboost:1.2.8
joblib:1.4.2
lightgbm:4.6.0
matplotlib:3.10.3
numpy: 1.26.4
pandas: 2.2.3
scipy: 1.13.1
seaborn: 0.13.2
shap: 0.48.0
sklearn: 1.5.1
tqdm: 4.66.5
xgboost: 3.0.2
```