"""
Wrapper for Streamlit Cloud deployment
Routes to the actual app in ml-heart-disease-project/app/
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Set up path
project_root = Path(__file__).parent / 'ml-heart-disease-project'
sys.path.insert(0, str(project_root / 'src'))

# Import required modules
from data_processing import DataProcessor
from feature_engineering import FeatureEngineer
from predict import ModelPredictor
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Run the app - import from actual location
# Since imports are already set up above, we can now run the app content
exec(open(str(project_root / 'app' / 'streamlit_app.py')).read())

