import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
test_ids = test_df['Id']

# Combine train and test for preprocessing
all_data = pd.concat([train_df.drop('SalePrice', axis=1), test_df], ignore_index=True)

# Handle missing values based on data description
# For features where NA means "None"
none_features = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                 'MasVnrType']

for feature in none_features:
    all_data[feature] = all_data[feature].fillna('None')

# Fill numeric features with 0 or median
zero_features = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1',
                 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
                 'BsmtHalfBath', 'MasVnrArea']

for feature in zero_features:
    all_data[feature] = all_data[feature].fillna(0)

# Fill LotFrontage with median by Neighborhood
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median()))

# Fill remaining with mode
for col in all_data.columns:
    if all_data[col].isnull().sum() > 0:
        if all_data[col].dtype == 'object':
            all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
        else:
            all_data[col] = all_data[col].fillna(all_data[col].median())

# Feature Engineering
# Total square footage
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

# Total bathrooms
all_data['TotalBath'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) +
                         all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath']))

# Total porch area
all_data['TotalPorchSF'] = (all_data['OpenPorchSF'] + all_data['3SsnPorch'] +
                             all_data['EnclosedPorch'] + all_data['ScreenPorch'] +
                             all_data['WoodDeckSF'])

# Has features
all_data['HasPool'] = (all_data['PoolArea'] > 0).astype(int)
all_data['HasGarage'] = (all_data['GarageArea'] > 0).astype(int)
all_data['HasBsmt'] = (all_data['TotalBsmtSF'] > 0).astype(int)
all_data['HasFireplace'] = (all_data['Fireplaces'] > 0).astype(int)

# Age features
all_data['Age'] = all_data['YrSold'] - all_data['YearBuilt']
all_data['AgeRemod'] = all_data['YrSold'] - all_data['YearRemodAdd']

# Quality scores
all_data['OverallScore'] = all_data['OverallQual'] * all_data['OverallCond']
all_data['GarageScore'] = all_data['GarageArea'] * all_data['GarageCars']

# Transform skewed numeric features
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(lambda x: x.skew()).sort_values(ascending=False)
high_skew = skewed_feats[abs(skewed_feats) > 0.5]
high_skew_features = high_skew.index

for feat in high_skew_features:
    if feat not in ['Age', 'AgeRemod']:  # Skip age features which might have negative values
        all_data[feat] = np.log1p(all_data[feat])

# Encode categorical variables
categorical_features = all_data.select_dtypes(include=['object']).columns

all_data_encoded = pd.get_dummies(all_data, columns=categorical_features, drop_first=True)

# Split back to train and test
X_train = all_data_encoded[:len(train_df)]
X_test = all_data_encoded[len(train_df):]

# Target variable (log transform for better distribution)
y_train = np.log1p(train_df['SalePrice'])

# Replace infinities with large values
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# Fill NaN values with median
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'lasso': Lasso(alpha=0.0005, random_state=42),
    'ridge': Ridge(alpha=10, random_state=42),
    'elastic': ElasticNet(alpha=0.0005, l1_ratio=0.5, random_state=42),
    'rf': RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42),
    'gbm': GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42),
    'xgb': XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42)
}

# Cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

print("Model Performance (5-Fold CV):")
print("-" * 40)

cv_scores = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=kfold, 
                            scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    cv_scores[name] = rmse_scores.mean()
    print(f"{name.upper()}: CV RMSE = {rmse_scores.mean():.4f} (+/- {rmse_scores.std():.4f})")

# Train best models on full data
best_models = ['ridge', 'lasso', 'xgb', 'gbm']
predictions = {}

for name in best_models:
    model = models[name]
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    predictions[name] = np.expm1(pred)  # Convert back from log

# Ensemble predictions (weighted average)
weights = {'ridge': 0.2, 'lasso': 0.2, 'xgb': 0.35, 'gbm': 0.25}
final_predictions = np.zeros(len(X_test))

for name, weight in weights.items():
    final_predictions += weight * predictions[name]

# Create submission
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': final_predictions
})

submission.to_csv('submission.csv', index=False)
print(f"\nSubmission file created: submission.csv")
print(f"Shape: {submission.shape}")
print(f"SalePrice range: {submission['SalePrice'].min():.2f} - {submission['SalePrice'].max():.2f}")