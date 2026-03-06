import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve)
import joblib

class ModelTrainer:
    """Train and evaluate multiple ML models"""
    
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', probability=True, random_state=42),
            'knn': KNeighborsClassifier(n_neighbors=5)
        }
        self.best_model = None
        self.best_model_name = None
        self.results = {}
    
    def train_model(self, X_train, y_train, model_name='random_forest'):
        """Train a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name='model'):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        
        # For models with probability prediction
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            roc_auc = roc_auc_score(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
        }
        
        self.results[model_name] = metrics
        return metrics
    
    def train_and_evaluate_all(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all models"""
        results_summary = {}
        
        for model_name in self.models.keys():
            print(f"\nTraining {model_name}...")
            model = self.train_model(X_train, y_train, model_name)
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            results_summary[model_name] = metrics
            
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1-Score: {metrics['f1']:.4f}")
            print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return results_summary
    
    def select_best_model(self, X_train, X_test, y_train, y_test, metric='roc_auc'):
        """Select the best model based on metric"""
        best_score = -1
        best_name = None
        
        for model_name in self.models.keys():
            model = self.train_model(X_train, y_train, model_name)
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            
            if metrics[metric] > best_score:
                best_score = metrics[metric]
                best_name = model_name
                self.best_model = model
        
        self.best_model_name = best_name
        print(f"\nBest Model: {best_name} with {metric} = {best_score:.4f}")
        return self.best_model, best_name
    
    def cross_validation(self, X_train, y_train, model_name='random_forest', cv=5):
        """Perform cross-validation"""
        model = self.models[model_name]
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        
        print(f"\nCross-Validation Results for {model_name}:")
        print(f"Scores: {scores}")
        print(f"Mean: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return scores
    
    def feature_importance(self, model_name='best'):
        """Get feature importance for tree-based models"""
        if model_name == 'best':
            model = self.best_model
        else:
            model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        else:
            print(f"Model {model_name} doesn't have feature importance")
            return None
    
    def save_model(self, filepath):
        """Save the best model"""
        if self.best_model is None:
            raise ValueError("No model trained yet")
        joblib.dump(self.best_model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.best_model = joblib.load(filepath)
        return self.best_model
    
    def print_classification_report(self, model, X_test, y_test):
        """Print detailed classification report"""
        y_pred = model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
