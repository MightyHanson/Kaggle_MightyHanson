import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Machine Learning Libraries
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

# Set display options
pd.set_option('display.max_columns', None)

# Step 1: Define Input and Output Directories
script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, 'input')
output_dir = os.path.join(script_dir, 'output')

# Step 2: Load Data
data = pd.read_csv(os.path.join(input_dir, 'train.csv'))
test_data = pd.read_csv(os.path.join(input_dir, 'test.csv'))

# Step 3: Data Preprocessing

## 3.1 Handle Missing Values

### 3.1.1 Initial Missing Values Check
def check_missing_values(df, df_name):
    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    print(f"Missing values in {df_name}:")
    print(missing)
    print(f"Total columns with missing values in {df_name}: {len(missing)}\n")

check_missing_values(data, "training data")
check_missing_values(test_data, "test data")

### 3.1.2 Fill Missing Values for 'MasVnrType' and 'MasVnrArea'
data['MasVnrType'] = data['MasVnrType'].fillna('None')
test_data['MasVnrType'] = test_data['MasVnrType'].fillna('None')
data['MasVnrArea'] = data['MasVnrArea'].fillna(0)
test_data['MasVnrArea'] = test_data['MasVnrArea'].fillna(0)

### 3.1.3 Impute 'LotFrontage' Based on 'Neighborhood'
for df in [data, test_data]:
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

### 3.1.4 Handle Garage Features
garage_cat_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
garage_num_cols = ['GarageArea', 'GarageCars', 'GarageYrBlt']

# Fill missing values
for col in garage_cat_cols:
    data[col] = data[col].fillna('None')
    test_data[col] = test_data[col].fillna('None')

for col in ['GarageArea', 'GarageCars']:
    data[col] = data[col].fillna(0)
    test_data[col] = test_data[col].fillna(0)

garage_mapping = {'Fin': 1, 'RFn': 2, 'Unf': 3,'None': 4,}
data['GarageFinish'] = data['GarageFinish'].map(garage_mapping)
test_data['GarageFinish'] = test_data['GarageFinish'].map(garage_mapping)

# Replace missing 'GarageYrBlt' with 'YearBuilt'
data['GarageYrBlt'] = data['GarageYrBlt'].fillna(data['YearBuilt'])
test_data['GarageYrBlt'] = test_data['GarageYrBlt'].fillna(test_data['YearBuilt'])

### 3.1.5 Handle Basement Features
basement_cat_cols = ['BsmtFinType2', 'BsmtExposure', 'BsmtQual', 'BsmtCond', 'BsmtFinType1']
basement_num_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']

# Fill missing values
for col in basement_cat_cols:
    data[col] = data[col].fillna('None')
    test_data[col] = test_data[col].fillna('None')

for col in basement_num_cols:
    data[col] = data[col].fillna(0)
    test_data[col] = test_data[col].fillna(0)

# Map 'BsmtExposure'
bsmt_exposure_mapping = {'Gd': 1, 'Av': 2, 'Mn': 3, 'No': 4, 'None': 5}
data['BsmtExposure'] = data['BsmtExposure'].map(bsmt_exposure_mapping)
test_data['BsmtExposure'] = test_data['BsmtExposure'].map(bsmt_exposure_mapping)

### 3.1.6 Handle 'Electrical'
electrical_mapping = {'FuseP': 1, 'FuseF': 2, 'FuseA': 3, 'Mix': 4, 'SBrkr': 5}
data['Electrical'] = data['Electrical'].map(electrical_mapping)
test_data['Electrical'] = test_data['Electrical'].map(electrical_mapping)
median_electrical = data['Electrical'].median()
data['Electrical'] = data['Electrical'].fillna(median_electrical)
test_data['Electrical'] = test_data['Electrical'].fillna(median_electrical)

### 3.1.7 Fill Missing Values with 'None' for Specific Columns
cols_fill_none = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
for col in cols_fill_none:
    data[col] = data[col].fillna('None')
    test_data[col] = test_data[col].fillna('None')

### 3.1.8 Handle 'Functional'
functional_mapping = {
    'Typ': 1, 'Min1': 2, 'Min2': 3, 'Mod': 4,
    'Maj1': 5, 'Maj2': 6, 'Sev': 7, 'Sal': 8
}
data['Functional'] = data['Functional'].map(functional_mapping)
test_data['Functional'] = test_data['Functional'].map(functional_mapping)
test_data['Functional'] = test_data['Functional'].fillna(data['Functional'].mode()[0])

### 3.1.9 Handle 'KitchenQual'
test_data['KitchenQual'] = test_data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])

### 3.1.10 Handle 'MSZoning' and 'SaleType'
test_data['MSZoning'] = test_data['MSZoning'].fillna(data['MSZoning'].mode()[0])
test_data['SaleType'] = test_data['SaleType'].fillna(data['SaleType'].mode()[0])

### 3.1.11 Drop 'Utilities' (Low Variance)
data = data.drop(['Utilities'], axis=1)
test_data = test_data.drop(['Utilities'], axis=1)

### 3.1.12 Handle Remaining Missing Values in Numerical Features
test_data['BsmtHalfBath'] = test_data['BsmtHalfBath'].fillna(0)
test_data['BsmtFullBath'] = test_data['BsmtFullBath'].fillna(0)

## 3.2 Feature Engineering

### 3.2.1 Combine 'Condition1' and 'Condition2'
conditions = ['Artery', 'Feedr', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe']
for cond in conditions:
    data['Condition_' + cond] = ((data['Condition1'] == cond) | (data['Condition2'] == cond)).astype(int)
    test_data['Condition_' + cond] = ((test_data['Condition1'] == cond) | (test_data['Condition2'] == cond)).astype(int)
data = data.drop(['Condition1', 'Condition2'], axis=1)
test_data = test_data.drop(['Condition1', 'Condition2'], axis=1)

### 3.2.2 Handle 'Exterior1st' and 'Exterior2nd'
# Fill missing values
test_data['Exterior2nd'] = test_data['Exterior2nd'].fillna('None')

# Create combined list of unique exterior materials
unique_exteriors = pd.concat([
    data['Exterior1st'], data['Exterior2nd'],
    test_data['Exterior1st'], test_data['Exterior2nd']
]).unique()
exterior_materials = [material for material in unique_exteriors if material != 'None']

# Create binary features for each exterior material
for material in exterior_materials:
    data[f'Exterior_{material}'] = ((data['Exterior1st'] == material) | (data['Exterior2nd'] == material)).astype(int)
    test_data[f'Exterior_{material}'] = ((test_data['Exterior1st'] == material) | (test_data['Exterior2nd'] == material)).astype(int)

# Drop original columns
data = data.drop(['Exterior1st', 'Exterior2nd'], axis=1)
test_data = test_data.drop(['Exterior1st', 'Exterior2nd'], axis=1)

### 3.2.3 Encode Categorical Features

#### 3.2.3.1 Label Encoding for Ordinal Features
qual_mapping = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
ordinal_cols = [
    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
    'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC'
]
for col in ordinal_cols:
    data[col] = data[col].map(qual_mapping)
    test_data[col] = test_data[col].map(qual_mapping)

# Basement Finishing Type
bsmtfintype_mapping = {'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1, 'None':0}
for col in ['BsmtFinType1', 'BsmtFinType2']:
    data[col] = data[col].map(bsmtfintype_mapping)
    test_data[col] = test_data[col].map(bsmtfintype_mapping)

#### 3.2.3.2 Map 'LandSlope', 'LotShape', 'PavedDrive'
land_slope_mapping = {'Gtl': 1, 'Mod': 2, 'Sev': 3}
lot_shape_mapping = {'Reg': 1, 'IR1': 2, 'IR2': 3, 'IR3': 4}
paved_drive_mapping = {'N': 3, 'P': 2, 'Y': 1}

data['LandSlope'] = data['LandSlope'].map(land_slope_mapping)
test_data['LandSlope'] = test_data['LandSlope'].map(land_slope_mapping)
data['LotShape'] = data['LotShape'].map(lot_shape_mapping)
test_data['LotShape'] = test_data['LotShape'].map(lot_shape_mapping)
data['PavedDrive'] = data['PavedDrive'].map(paved_drive_mapping)
test_data['PavedDrive'] = test_data['PavedDrive'].map(paved_drive_mapping)

#### 3.2.3.3 Encode 'CentralAir'
data['CentralAir'] = data['CentralAir'].map({'Y': 1, 'N': 0})
test_data['CentralAir'] = test_data['CentralAir'].map({'Y': 1, 'N': 0})

#### 3.2.3.4 One-Hot Encoding for Nominal Features
# One-Hot Encoding for Nominal Features:
data = pd.get_dummies(data, columns=['Neighborhood',
                                     'MasVnrType', 'Foundation',
                                     'SaleType', 'SaleCondition',
                                     'MSSubClass', 'MSZoning',
                                     'Street','LandContour',
                                     'LotConfig','BldgType',
                                     ])
test_data = pd.get_dummies(test_data, columns=['Neighborhood',
                                     'MasVnrType', 'Foundation',
                                     'SaleType', 'SaleCondition',
                                     'MSSubClass', 'MSZoning',
                                     'Street','LandContour',
                                     'LotConfig','BldgType'])

# List of nominal categorical features to one-hot encode
nominal_cols = ['Alley', 'Fence', 'MiscFeature','HouseStyle','RoofStyle','RoofMatl','Heating', 'GarageType']

# Apply one-hot encoding
data = pd.get_dummies(data, columns=nominal_cols, drop_first=True)
test_data = pd.get_dummies(test_data, columns=nominal_cols, drop_first=True)

## 3.3 Additional Feature Engineering

### 3.3.1 Create New Features
data['TotalBath'] = (data['FullBath'] + data['BsmtFullBath']) + 0.5 * (data['HalfBath'] + data['BsmtHalfBath'])
test_data['TotalBath'] = (test_data['FullBath'] + test_data['BsmtFullBath']) + 0.5 * (test_data['HalfBath'] + test_data['BsmtHalfBath'])
data['HouseAge'] = data['YrSold'] - data['YearBuilt']
test_data['HouseAge'] = test_data['YrSold'] - test_data['YearBuilt']
data['RemodAge'] = data['YrSold'] - data['YearRemodAdd']
test_data['RemodAge'] = test_data['YrSold'] - test_data['YearRemodAdd']

### 3.3.2 Remove Outliers
plt.scatter(data['GrLivArea'], data['SalePrice'])
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.show()

# Remove outliers
data = data.drop(data[(data['GrLivArea'] > 4000) & (data['SalePrice'] < 300000)].index)

## 3.4 Final Adjustments

### 3.4.1 Align Training and Test Data
train_columns = set(data.columns)
test_columns = set(test_data.columns)

missing_in_test = train_columns - test_columns
missing_in_train = test_columns - train_columns

# Add missing columns to test data
for col in missing_in_test:
    test_data[col] = 0

# Add missing columns to training data
for col in missing_in_train:
    data[col] = 0

# Ensure the same column order
test_data = test_data[data.columns.drop('SalePrice')]

### 3.4.2 Verify No Remaining Missing Values
def verify_no_missing_values(df, df_name):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print(f"No missing values in {df_name} after preprocessing.\n")
    else:
        print(f"Missing values remaining in {df_name}:")
        print(missing)

verify_no_missing_values(data, "training data")
verify_no_missing_values(test_data, "test data")



# Step 4: Prepare Data for Modeling

## 4.1 Separate Features and Target Variable
X = data.drop(['SalePrice', 'Id'], axis=1)
y = data['SalePrice']
X_test = test_data.drop(['Id'], axis=1)

## 4.2 Ensure Same Columns in Both Datasets
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)


## 4.3 Log Transformation of Target Variable
y_log = np.log1p(y)

# Step 5: Modeling

## 5.1 Define Cross-Validation Strategy
kf = KFold(n_splits=5, shuffle=True, random_state=42)

## 5.2 Define Cross-Validation Function
def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X, y_log, scoring="neg_mean_squared_error", cv=kf))
    return rmse

## 5.3 Initialize Models
models = {
    'Lasso': Lasso(alpha=0.001, random_state=42),
    'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=1000, learning_rate=0.05,
        max_depth=4, max_features='sqrt',
        min_samples_leaf=15, min_samples_split=10,
        loss='huber', random_state=42
    ),
    'XGBoost': xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.05,
        max_depth=4, min_child_weight=1,
        gamma=0, subsample=0.7, colsample_bytree=0.7,
        objective='reg:squarederror', nthread=-1,
        random_state=42
    )
}

## 5.4 Evaluate Models
for name, model in models.items():
    score = rmse_cv(model)
    print(f"{name} RMSE: {score.mean():.4f}")

# Step 6: Train Final Model and Make Predictions
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [500, 1000, 1500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'min_samples_leaf': [5, 10, 15],
    'min_samples_split': [5, 10, 15],
    'max_features': ['sqrt', 'log2', None],
    'loss': ['huber', 'ls', 'lad']
}
from sklearn.model_selection import KFold, GridSearchCV

# Define cross-validation strategy
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize the model
gbr = GradientBoostingRegressor(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=gbr,
    param_grid=param_grid,
    cv=kf,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2
)

## 6.1 Train Final Model
final_model = GradientBoostingRegressor(
    n_estimators=1000, learning_rate=0.05,
    max_depth=4, max_features='sqrt',
    min_samples_leaf=15, min_samples_split=10,
    loss='huber', random_state=42
)
final_model.fit(X, y_log)

## 6.2 Predict on Test Data
y_test_pred_log = final_model.predict(X_test)
y_test_pred = np.expm1(y_test_pred_log)  # Convert back to original scale

# Step 7: Prepare Submission File
submission = pd.DataFrame({
    'Id': test_data['Id'],
    'SalePrice': y_test_pred
})

# Save to CSV
submission.to_csv(os.path.join(output_dir, 'submission_Han.csv'), index=False)
