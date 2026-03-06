import sys
import os
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

from data_processing import DataProcessor
from feature_engineering import FeatureEngineer
from train_model import ModelTrainer
from predict import ModelPredictor

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 60)
print("HEART DISEASE PREDICTION ML PIPELINE")
print("=" * 60)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================
print("\n[1] Loading and Preprocessing Data...")
processor = DataProcessor()

# Load data
data_path = project_root / 'data' / 'raw' / 'heart.csv'
df = processor.load_data(str(data_path))
print(f"Original dataset shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nDataset info:\n{df.info()}")
print(f"\nTarget distribution:\n{df['target'].value_counts()}")

# Preprocess data
X, y = processor.preprocess(df, fit=True)
print(f"\nProcessed dataset shape: {X.shape}")
print(f"Features: {list(X.columns)}")

# Split data
X_train, X_test, y_train, y_test = processor.train_test_split_data(X, y)
print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
print("\n[2] Feature Engineering...")
engineer = FeatureEngineer()

# Create domain-specific features
X_train_enhanced = engineer.create_domain_features(X_train)
X_test_enhanced = engineer.create_domain_features(X_test)

print(f"Enhanced features shape: {X_train_enhanced.shape}")
print(f"New features created: {X_train_enhanced.shape[1] - X_train.shape[1]}")

# ============================================================================
# 3. MODEL TRAINING AND EVALUATION
# ============================================================================
print("\n[3] Training Multiple Models...")
trainer = ModelTrainer()

# Train and evaluate all models
results_summary = trainer.train_and_evaluate_all(X_train_enhanced, X_test_enhanced, y_train, y_test)

# ============================================================================
# 4. SELECT BEST MODEL
# ============================================================================
print("\n[4] Selecting Best Model...")
best_model, best_model_name = trainer.select_best_model(X_train_enhanced, X_test_enhanced, y_train, y_test, metric='roc_auc')

# Print detailed classification report
trainer.print_classification_report(best_model, X_test_enhanced, y_test)

# ============================================================================
# 5. FEATURE IMPORTANCE (if applicable)
# ============================================================================
if hasattr(best_model, 'feature_importances_'):
    print("\n[5] Feature Importance...")
    feature_importance = best_model.feature_importances_
    feature_names = X_train_enhanced.columns
    
    # Sort and display top 10
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False).head(10)
    
    print("\nTop 10 Most Important Features:")
    print(importance_df)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
    plt.title(f'Top 10 Feature Importance - {best_model_name}')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plot_path = project_root / 'models' / 'feature_importance.png'
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to {plot_path}")

# ============================================================================
# 6. MODEL COMPARISON
# ============================================================================
print("\n[6] Model Comparison...")
comparison_df = pd.DataFrame(results_summary).T
print("\nModel Performance Comparison:")
print(comparison_df[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']])

# Plot model comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

metrics_to_plot = ['accuracy', 'f1', 'roc_auc']
comparison_df[metrics_to_plot].plot(kind='bar', ax=axes[0])
axes[0].set_title('Model Performance Comparison')
axes[0].set_ylabel('Score')
axes[0].legend(title='Metrics')
axes[0].grid(True, alpha=0.3)

# ROC Curve for best model
y_pred_proba = best_model.predict_proba(X_test_enhanced)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

axes[1].plot(fpr, tpr, label=f'{best_model_name} (AUC = {roc_auc:.3f})', linewidth=2)
axes[1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve - Best Model')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
comparison_plot_path = project_root / 'models' / 'model_comparison.png'
comparison_plot_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(str(comparison_plot_path), dpi=300, bbox_inches='tight')
print(f"\nModel comparison plot saved to {comparison_plot_path}")

# ============================================================================
# 7. SAVE BEST MODEL
# ============================================================================
print("\n[7] Saving Best Model...")
model_save_path = project_root / 'models' / 'best_heart_disease_model.pkl'
model_save_path.parent.mkdir(parents=True, exist_ok=True)
trainer.save_model(str(model_save_path))
print(f"Model saved to {model_save_path}")

# ============================================================================
# 8. EXAMPLE PREDICTIONS
# ============================================================================
print("\n[8] Making Predictions...")
predictor = ModelPredictor(str(model_save_path))

# Example: Predict for a single patient
sample_patient = {
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

# Note: In production, you would scale the input using the fitted scaler
# prediction = predictor.predict_single(sample_patient)
# print(f"\nSample Prediction: {prediction}")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE!")
print("=" * 60)
