import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

class ModelPredictor:
    """Make predictions using trained model"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.scaler = StandardScaler()
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load a trained model"""
        self.model = joblib.load(model_path)
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError("Model doesn't support probability predictions")
    
    def predict_single(self, features_dict):
        """Predict for a single sample given as dictionary"""
        X = pd.DataFrame([features_dict])
        prediction = self.predict(X)
        
        if hasattr(self.model, 'predict_proba'):
            proba = self.predict_proba(X)
            return {
                'prediction': int(prediction[0]),
                'probability': float(proba[0][1]),
                'risk_level': self._get_risk_level(proba[0][1])
            }
        else:
            return {
                'prediction': int(prediction[0]),
                'probability': None,
                'risk_level': 'unknown'
            }
    
    def _get_risk_level(self, probability):
        """Convert probability to risk level"""
        if probability < 0.3:
            return 'Low Risk'
        elif probability < 0.7:
            return 'Medium Risk'
        else:
            return 'High Risk'
    
    def batch_predict(self, X_batch):
        """Predict for multiple samples"""
        predictions = self.predict(X_batch)
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.predict_proba(X_batch)
            results = pd.DataFrame({
                'prediction': predictions,
                'probability': probabilities[:, 1],
                'risk_level': [self._get_risk_level(p) for p in probabilities[:, 1]]
            })
        else:
            results = pd.DataFrame({
                'prediction': predictions
            })
        
        return results
