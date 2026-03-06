import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

class FeatureEngineer:
    """Engineer features for better model performance"""
    
    def __init__(self):
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
    
    def create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial and interaction features"""
        X_poly = self.poly.fit_transform(X)
        
        # Create feature names for polynomial features
        feature_names = self.poly.get_feature_names_out(X.columns)
        X_enhanced = pd.DataFrame(X_poly, columns=feature_names)
        
        return X_enhanced
    
    def create_domain_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific features for heart disease"""
        X_enhanced = X.copy()
        
        # Age-based features
        age_cut = pd.cut(X['age'], bins=[0, 40, 50, 60, 100], labels=[0, 1, 2, 3])
        X_enhanced['age_group'] = age_cut.fillna(0).astype(int)
        
        # Blood pressure categories
        bp_cut = pd.cut(X['trestbps'], 
                        bins=[0, 120, 140, 180, 300], 
                        labels=[0, 1, 2, 3])
        X_enhanced['bp_category'] = bp_cut.fillna(0).astype(int)
        
        # Cholesterol categories
        chol_cut = pd.cut(X['chol'], 
                          bins=[0, 200, 240, 1000], 
                          labels=[0, 1, 2])
        X_enhanced['chol_category'] = chol_cut.fillna(0).astype(int)
        
        # Heart rate features
        X_enhanced['max_hr_age_ratio'] = X['thalach'] / X['age']
        
        # Create risk score based on multiple factors
        X_enhanced['risk_index'] = (X['exang'] * 2 + X['oldpeak'] * 1.5 + 
                                    X['ca'] * 2 + (X['thal'] >= 2).astype(int))
        
        return X_enhanced
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, method='correlation', top_k=10):
        """Select top features based on correlation with target"""
        if method == 'correlation':
            # Calculate correlation with target
            correlations = []
            for col in X.columns:
                corr = X[col].corr(y)
                correlations.append(abs(corr))
            
            # Get top k features
            top_features = X.columns[np.argsort(correlations)[-top_k:]].tolist()
            return X[top_features]
        
        return X
