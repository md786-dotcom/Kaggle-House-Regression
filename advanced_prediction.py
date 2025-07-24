import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from xgboost import XGBRegressor
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Averaged base models class for stacking
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        for model in self.models_:
            model.fit(X, y)
        return self
    
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
test_ids = test_df['Id']

# Store original train indices before removing outliers
original_train_index = train_df.index.tolist()

# Remove outliers based on GrLivArea
train_df = train_df[~((train_df['GrLivArea'] > 4000) & (train_df['SalePrice'] < 300000))]

# Store the indices after outlier removal
train_index_after_outliers = train_df.index.tolist()

# Combine for preprocessing
all_data = pd.concat([train_df.drop('SalePrice', axis=1), test_df], ignore_index=True)

# Handle missing values
# None features
none_features = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                 'BsmtFinType2', 'MasVnrType']

for feature in none_features:
    all_data[feature] = all_data[feature].fillna('None')

# Zero features
zero_features = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1',
                 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
                 'BsmtHalfBath', 'MasVnrArea']

for feature in zero_features:
    all_data[feature] = all_data[feature].fillna(0)

# Special handling
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median()))

# Mode for remaining
for col in ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 
            'Exterior2nd', 'SaleType', 'Functional']:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

# Utilities - assume typical
all_data['Utilities'] = all_data['Utilities'].fillna('AllPub')

# Advanced Feature Engineering
# Total square footage
all_data['TotalSF'] = (all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + 
                       all_data['2ndFlrSF'])

# Quality features
all_data['OverallScore'] = all_data['OverallQual'] * all_data['OverallCond']
all_data['ExterScore'] = all_data['ExterQual'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}) * \
                         all_data['ExterCond'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1})

# Bathroom features
all_data['TotalBath'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) +
                         all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath']))

# Porch features
all_data['TotalPorchSF'] = (all_data['OpenPorchSF'] + all_data['3SsnPorch'] +
                             all_data['EnclosedPorch'] + all_data['ScreenPorch'] +
                             all_data['WoodDeckSF'])

# Has features
has_features = ['Pool', 'Garage', 'Bsmt', 'Fireplace', '2ndFlr', 'Porch']
all_data['HasPool'] = (all_data['PoolArea'] > 0).astype(int)
all_data['HasGarage'] = (all_data['GarageArea'] > 0).astype(int)
all_data['HasBsmt'] = (all_data['TotalBsmtSF'] > 0).astype(int)
all_data['HasFireplace'] = (all_data['Fireplaces'] > 0).astype(int)
all_data['Has2ndFlr'] = (all_data['2ndFlrSF'] > 0).astype(int)
all_data['HasPorch'] = (all_data['TotalPorchSF'] > 0).astype(int)

# Age and remodel features
all_data['Age'] = all_data['YrSold'] - all_data['YearBuilt']
all_data['AgeRemod'] = all_data['YrSold'] - all_data['YearRemodAdd']
all_data['RemodFlag'] = (all_data['YearBuilt'] != all_data['YearRemodAdd']).astype(int)

# Neighborhood price encoding (using median from train data)
neighborhood_map = train_df.groupby('Neighborhood')['SalePrice'].median().to_dict()
all_data['NeighborhoodPrice'] = all_data['Neighborhood'].map(neighborhood_map)
all_data['NeighborhoodPrice'].fillna(all_data['NeighborhoodPrice'].median(), inplace=True)

# Room ratios
all_data['BedPerRoom'] = all_data['BedroomAbvGr'] / (all_data['TotRmsAbvGrd'] + 1)
all_data['BathPerBed'] = all_data['TotalBath'] / (all_data['BedroomAbvGr'] + 1)

# Quality counts
quality_cols = ['ExterQual', 'BsmtQual', 'KitchenQual', 'GarageQual', 'FireplaceQu']
all_data['QualityCount'] = 0
for col in quality_cols:
    all_data['QualityCount'] += all_data[col].map({'Ex': 1, 'Gd': 0.75, 'TA': 0.5, 
                                                    'Fa': 0.25, 'Po': 0, 'None': 0, None: 0})

# Transform highly skewed features
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(lambda x: x.skew()).sort_values(ascending=False)
high_skew = skewed_feats[abs(skewed_feats) > 0.75]

for feat in high_skew.index:
    if feat not in ['Age', 'AgeRemod'] and all_data[feat].min() >= 0:
        all_data[feat] = np.log1p(all_data[feat])

# One-hot encoding
all_data_encoded = pd.get_dummies(all_data, drop_first=True)

# Split back
X_train = all_data_encoded[:len(train_df)]
X_test = all_data_encoded[len(train_df):]
y_train = np.log1p(train_df['SalePrice'])

# Handle any infinities
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_test.median())

# Define base models with optimized parameters
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=42))
ridge = make_pipeline(RobustScaler(), Ridge(alpha=13, random_state=42))
elastic = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=42))
kernel_ridge = make_pipeline(RobustScaler(), KernelRidge(alpha=0.6, kernel='polynomial', degree=2))

gbm = GradientBoostingRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=4,
    max_features='sqrt', min_samples_leaf=15, min_samples_split=10,
    loss='huber', random_state=42
)

xgb = XGBRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=3,
    min_child_weight=0, gamma=0, subsample=0.7,
    colsample_bytree=0.7, objective='reg:linear',
    reg_alpha=0.0006, random_state=42, n_jobs=-1
)

lgb_model = lgb.LGBMRegressor(
    objective='regression', num_leaves=5, learning_rate=0.05,
    n_estimators=720, max_bin=55, bagging_fraction=0.8,
    bagging_freq=5, feature_fraction=0.2, feature_fraction_seed=9,
    bagging_seed=9, min_data_in_leaf=6, min_sum_hessian_in_leaf=11
)

# Stacked model
averaged_models = AveragingModels(models=[ridge, lasso, elastic, kernel_ridge])

# Cross validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

print("Advanced Model Performance (5-Fold CV):")
print("-" * 50)

models = {
    'Lasso': lasso,
    'Ridge': ridge,
    'ElasticNet': elastic,
    'KernelRidge': kernel_ridge,
    'GradientBoosting': gbm,
    'XGBoost': xgb,
    'LightGBM': lgb_model,
    'AveragedModels': averaged_models
}

for name, model in models.items():
    score = cross_val_score(model, X_train, y_train, cv=kfold, 
                           scoring='neg_mean_squared_error', n_jobs=-1)
    rmse = np.sqrt(-score)
    print(f"{name}: CV RMSE = {rmse.mean():.4f} (+/- {rmse.std():.4f})")

# Train final models
print("\nTraining final models...")
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = np.expm1(model.predict(X_test))
    predictions[name] = pred
    print(f"{name} trained")

# Weighted ensemble
weights = {
    'Lasso': 0.10,
    'Ridge': 0.10,
    'ElasticNet': 0.10,
    'KernelRidge': 0.10,
    'GradientBoosting': 0.20,
    'XGBoost': 0.20,
    'LightGBM': 0.15,
    'AveragedModels': 0.05
}

final_predictions = np.zeros(len(X_test))
for name, weight in weights.items():
    final_predictions += weight * predictions[name]

# Create submission
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': final_predictions
})

submission.to_csv('advanced_submission.csv', index=False)
print(f"\nAdvanced submission file created: advanced_submission.csv")
print(f"Predictions range: ${submission['SalePrice'].min():,.2f} - ${submission['SalePrice'].max():,.2f}")
print(f"Mean prediction: ${submission['SalePrice'].mean():,.2f}")