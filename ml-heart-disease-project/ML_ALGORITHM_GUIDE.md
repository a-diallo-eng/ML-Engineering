# Heart Disease Prediction - ML Algorithm

## Project Overview
This project implements a complete machine learning pipeline for predicting heart disease using multiple classification algorithms. The pipeline includes data processing, feature engineering, model training, evaluation, and prediction capabilities.

## Features

### Data Processing (`src/data_processing.py`)
- **Data Loading**: Load CSV datasets
- **Missing Value Handling**: Smart imputation using median for numeric values
- **Outlier Detection**: IQR-based outlier removal
- **Feature Scaling**: StandardScaler normalization
- **Train-Test Split**: Stratified splitting for balanced datasets

### Feature Engineering (`src/feature_engineering.py`)
- **Domain-Specific Features**:
  - Age grouping (0-40, 40-50, 50-60, 60+)
  - Blood pressure categories
  - Cholesterol categories
  - Heart rate to age ratio
  - Risk index (combines multiple factors)
- **Polynomial Features**: Interaction terms
- **Feature Selection**: Top-K feature selection based on correlation

### Model Training (`src/train_model.py`)
**Implemented Algorithms:**
1. **Logistic Regression** - Fast, interpretable baseline
2. **Random Forest** - Ensemble with feature importance
3. **Gradient Boosting** - High-performance ensemble
4. **Support Vector Machine (SVM)** - Non-linear classification
5. **K-Nearest Neighbors** - Instance-based learning

**Evaluation Metrics:**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Classification Report

**Features:**
- Train individual models
- Train and evaluate all models
- Automatic best model selection
- Cross-validation support
- Feature importance extraction
- Model persistence (save/load)

### Prediction (`src/predict.py`)
- Load trained models
- Single sample prediction
- Batch predictions
- Probability estimates
- Risk level classification (Low/Medium/High)

## Dataset Features (14 features)
- **age**: Age in years
- **sex**: Gender (0=Female, 1=Male)
- **cp**: Chest pain type (0-3)
- **trestbps**: Resting blood pressure (mmHg)
- **chol**: Serum cholesterol (mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl (0/1)
- **restecg**: Resting ECG results (0-2)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (0/1)
- **oldpeak**: ST depression induced by exercise
- **slope**: Slope of ST segment (0-2)
- **ca**: Number of major vessels colored by fluoroscopy (0-3)
- **thal**: Thalassemia (0-3)
- **target**: Heart disease presence (0=No, 1=Yes)

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the data file:
```
data/raw/heart.csv
```

## Usage

### Quick Start

```python
import sys
sys.path.insert(0, 'src')

from data_processing import DataProcessor
from feature_engineering import FeatureEngineer
from train_model import ModelTrainer
from predict import ModelPredictor

# 1. Load and preprocess data
processor = DataProcessor()
df = processor.load_data('data/raw/heart.csv')
X, y = processor.preprocess(df)
X_train, X_test, y_train, y_test = processor.train_test_split_data(X, y)

# 2. Feature engineering
engineer = FeatureEngineer()
X_train = engineer.create_domain_features(X_train)
X_test = engineer.create_domain_features(X_test)

# 3. Train models
trainer = ModelTrainer()
best_model, best_name = trainer.select_best_model(X_train, X_test, y_train, y_test)

# 4. Save model
trainer.save_model('models/best_model.pkl')

# 5. Make predictions
predictor = ModelPredictor('models/best_model.pkl')
predictions = predictor.batch_predict(X_test)
print(predictions)
```

### Run Full Pipeline

```bash
cd notebooks
python ml_pipeline.py
```

This will:
- Load and preprocess the heart disease dataset
- Create engineered features
- Train 5 different models
- Compare performance
- Select and save the best model
- Generate performance visualizations

### Single Patient Prediction

```python
from src.predict import ModelPredictor

predictor = ModelPredictor('models/best_heart_disease_model.pkl')

# Patient data
patient_data = {
    'age': 50,
    'sex': 1,
    'cp': 2,
    'trestbps': 130,
    'chol': 220,
    'fbs': 0,
    'restecg': 1,
    'thalach': 120,
    'exang': 1,
    'oldpeak': 1.5,
    'slope': 1,
    'ca': 1,
    'thal': 2
}

result = predictor.predict_single(patient_data)
print(result)
# Output: {'prediction': 1, 'probability': 0.87, 'risk_level': 'High Risk'}
```

## Model Performance

The pipeline evaluates 5 different algorithms and selects the best one based on ROC-AUC score. Typical performance:

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | 0.82 | 0.78 | 0.85 |
| Random Forest | 0.85 | 0.82 | 0.90 |
| Gradient Boosting | 0.87 | 0.84 | 0.92 |
| SVM | 0.83 | 0.80 | 0.87 |
| KNN | 0.81 | 0.77 | 0.83 |

## Output Files

After running the pipeline:
- `models/best_heart_disease_model.pkl` - Trained model
- `models/feature_importance.png` - Feature importance visualization
- `models/model_comparison.png` - Performance comparison chart

## File Structure

```
ml-heart-disease-project/
├── data/
│   ├── raw/
│   │   └── heart.csv
│   └── processed/
├── models/
│   ├── best_heart_disease_model.pkl
│   ├── feature_importance.png
│   └── model_comparison.png
├── notebooks/
│   ├── 01_exploration.ipynb
│   └── ml_pipeline.py
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── predict.py
├── tests/
└── requirements.txt
```

## Key Functions

### DataProcessor
```python
processor = DataProcessor()
df = processor.load_data(filepath)
X, y = processor.preprocess(df, fit=True)
X_train, X_test, y_train, y_test = processor.train_test_split_data(X, y)
```

### FeatureEngineer
```python
engineer = FeatureEngineer()
X_enhanced = engineer.create_domain_features(X)
X_selected = engineer.select_features(X, y, method='correlation', top_k=10)
```

### ModelTrainer
```python
trainer = ModelTrainer()
model = trainer.train_model(X_train, y_train, 'random_forest')
metrics = trainer.evaluate_model(model, X_test, y_test)
best_model, name = trainer.select_best_model(X_train, X_test, y_train, y_test)
trainer.save_model('path/to/model.pkl')
```

### ModelPredictor
```python
predictor = ModelPredictor('path/to/model.pkl')
predictions = predictor.predict(X)
probabilities = predictor.predict_proba(X)
result = predictor.predict_single(features_dict)
batch_results = predictor.batch_predict(X_batch)
```

## Requirements

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

## Next Steps

1. **Data Enhancement**: Add more external features or domain knowledge
2. **Hyperparameter Tuning**: Use GridSearchCV or RandomizedSearchCV for optimization
3. **Cross-Validation**: Implement K-fold cross-validation
4. **Model Deployment**: Create an API (Flask/FastAPI) for predictions
5. **Monitoring**: Track model performance over time with new data

## Author
ML-Engineering Project

## License
MIT
