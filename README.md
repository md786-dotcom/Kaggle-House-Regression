# 🏠 Kaggle House Price Prediction - Advanced Regression Techniques

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF.svg)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kaggle-house-regression-lmt7fphzvhzj7t7zcqvtqy.streamlit.app/)

## 🏆 Competition Performance

**Current Ranking: 519 / 4,677** (Top 11.1%) as of July 24, 2025, 4:30 PM AEST

A comprehensive solution for predicting house prices using advanced regression techniques, feature engineering, and ensemble methods.

🚀 **[Live Demo on Streamlit](https://kaggle-house-regression-lmt7fphzvhzj7t7zcqvtqy.streamlit.app/)** - Interactive demonstration of the techniques and models used

## 📊 Project Overview

This project tackles the Kaggle House Prices competition, which challenges participants to predict the final sale price of residential homes in Ames, Iowa. With 79 explanatory variables describing almost every aspect of residential homes, this competition provides an excellent opportunity to showcase advanced regression techniques and feature engineering skills.

### Key Achievements
- **Top 11.1% ranking** among 4,677 participants
- Implemented multiple advanced regression models including XGBoost, LightGBM, and Kernel Ridge
- Extensive feature engineering creating 20+ new features
- Robust preprocessing pipeline handling missing values and outliers
- Ensemble approach combining 8 different models with optimized weights

## 🚀 Features

### Data Preprocessing
- **Intelligent Missing Value Handling**: Context-aware imputation based on data description
- **Outlier Detection & Removal**: Statistical analysis to identify and handle anomalies
- **Feature Scaling**: Robust scaling to handle outliers effectively

### Feature Engineering
- **Spatial Features**: Total square footage combinations, room ratios
- **Quality Scores**: Composite scores from multiple quality indicators
- **Temporal Features**: Age calculations, remodeling indicators
- **Neighborhood Encoding**: Price-based encoding using median values
- **Binary Indicators**: HasPool, HasGarage, HasBasement, etc.
- **Interaction Features**: Quality × Condition scores

### Models Implemented
1. **Linear Models**
   - Lasso Regression (α=0.0005)
   - Ridge Regression (α=13)
   - ElasticNet (α=0.0005, l1_ratio=0.9)
   - Kernel Ridge (polynomial kernel, degree=2)

2. **Tree-Based Models**
   - Random Forest (300 trees, max_depth=15)
   - Gradient Boosting (1000 trees, learning_rate=0.05)
   - XGBoost (1000 trees, learning_rate=0.05)
   - LightGBM (720 trees, num_leaves=5)

3. **Ensemble Methods**
   - Weighted averaging of all models
   - Stacked generalization approach

## 📈 Model Performance

Cross-validation results (5-Fold CV RMSE):

| Model | Basic Version | Advanced Version |
|-------|--------------|------------------|
| Lasso | 0.1416 | 0.1126 |
| Ridge | 0.1423 | 0.1120 |
| ElasticNet | 0.1446 | 0.1127 |
| Kernel Ridge | - | 0.1286 |
| Random Forest | 0.1418 | - |
| Gradient Boosting | 0.1336 | 0.1137 |
| XGBoost | 0.1340 | 0.1164 |
| LightGBM | - | 0.1152 |
| **Ensemble** | **0.133** | **0.112** |

## 🛠️ Installation & Usage

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Quick Start
1. Clone the repository
```bash
git clone https://github.com/yourusername/Kaggle-House-Regression.git
cd Kaggle-House-Regression
```

2. Download competition data from Kaggle
```bash
# Place train.csv and test.csv in the project directory
```

3. Run the prediction pipeline
```bash
# Basic model
python house_price_prediction.py

# Advanced model with enhanced features
python advanced_prediction.py
```

### Output
- `submission.csv`: Basic model predictions
- `advanced_submission.csv`: Advanced model predictions (recommended for submission)

## 📁 Project Structure

```
Kaggle-House-Regression/
│
├── house_price_prediction.py    # Basic model implementation
├── advanced_prediction.py        # Advanced model with feature engineering
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT license
├── README.md                     # Project documentation
└── .gitignore                   # Git ignore rules
```

## 🔬 Technical Details

### Feature Engineering Highlights

1. **Total Square Footage**: Combines basement, first floor, and second floor areas
2. **Quality Interactions**: Multiplies quality and condition ratings
3. **Neighborhood Price Encoding**: Uses median prices from training data
4. **Age Features**: Both absolute age and time since remodeling
5. **Room Ratios**: Bedrooms per room, bathrooms per bedroom

### Handling Missing Values

- **"None" Features**: PoolQC, MiscFeature, Alley, Fence (missing = no feature)
- **Zero Features**: Garage and basement measurements (missing = no garage/basement)
- **Neighborhood-based**: LotFrontage imputed by neighborhood median
- **Mode Imputation**: Remaining categorical features

### Model Optimization

- **Hyperparameter Tuning**: Grid search for optimal parameters
- **Cross-Validation**: 5-fold CV for robust evaluation
- **Ensemble Weights**: Optimized based on CV performance
- **Log Transformation**: Target variable and skewed features

## 📊 Visualizations

The project includes visualization of:
- Target variable distribution (normal vs log-transformed)
- Feature correlations with sale price
- Model performance comparisons
- Prediction distributions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Kaggle for hosting the competition
- The Ames Housing dataset compiled by Dean De Cock
- The data science community for insights and discussions

## 📞 Contact

Feel free to reach out if you have any questions or suggestions!

---

*Note: Competition data files are not included in this repository per Kaggle's terms. Please download them directly from the [competition page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).*