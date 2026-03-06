import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple

class DataProcessor:
    """Handle data loading, cleaning, and preprocessing"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders = {}
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load CSV data"""
        return pd.read_csv(filepath)
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in dataset"""
        # Drop rows with missing target variable
        df = df.dropna(subset=['target'])
        # Fill numeric missing values with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        return df
    
    def remove_outliers(self, df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
        """Remove outliers using IQR method"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col == 'target':  # Skip target variable
                continue
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        return df
    
    def preprocess(self, df: pd.DataFrame, fit=True) -> pd.DataFrame:
        """Complete preprocessing pipeline"""
        df = self.handle_missing_values(df)
        df = self.remove_outliers(df)
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        X_processed = pd.DataFrame(X_scaled, columns=X.columns)
        
        return X_processed, y
    
    def train_test_split_data(self, X: pd.DataFrame, y: pd.Series, 
                             test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """Split data into train and test sets"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
