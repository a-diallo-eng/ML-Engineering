"""
Heart Disease Risk Predictor - Main Streamlit App
Deployed on Streamlit Cloud
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib

# Add src to path for imports
project_dir = Path(__file__).parent / 'ml-heart-disease-project'
sys.path.insert(0, str(project_dir / 'src'))

# Now import our modules
from data_processing import DataProcessor
from feature_engineering import FeatureEngineer
from predict import ModelPredictor
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .risk-high {
        background-color: #ffcccc;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #cc0000;
    }
    .risk-medium {
        background-color: #ffffcc;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffaa00;
    }
    .risk-low {
        background-color: #ccffcc;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #00cc00;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Title and Description
st.title("❤️ Heart Disease Risk Predictor")
st.markdown("---")
st.markdown("""
This application uses machine learning to predict the risk of heart disease based on clinical parameters.
Simply enter your health metrics and get an instant risk assessment.
""")

# Sidebar for Navigation
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Select Page:",
        ["🔍 Prediction", "📊 Model Info", "📈 Analytics", "ℹ️ About"]
    )

# ============================================================================
# PAGE 1: PREDICTION
# ============================================================================
if page == "🔍 Prediction":
    st.header("Patient Risk Assessment")
    
    # Load model
    model_path = project_dir / 'models' / 'best_heart_disease_model.pkl'
    
    if not model_path.exists():
        st.error("❌ Model not found. The model file needs to be in the repository.")
        st.info("To train and create the model locally, run: `python notebooks/ml_pipeline.py`")
    else:
        try:
            predictor = ModelPredictor(str(model_path))
            
            # Create two columns for input
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Demographic Information")
                age = st.slider("Age (years)", min_value=20, max_value=100, value=50, step=1)
                sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
                
                st.subheader("Symptoms")
                cp = st.selectbox(
                    "Chest Pain Type",
                    options=[0, 1, 2, 3],
                    format_func=lambda x: {
                        0: "Typical Angina",
                        1: "Atypical Angina",
                        2: "Non-anginal Pain",
                        3: "Asymptomatic"
                    }.get(x)
                )
                exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                oldpeak = st.slider("ST Depression Induced by Exercise (0-6.2 mm)", min_value=0.0, max_value=6.2, value=1.0, step=0.1)
            
            with col2:
                st.subheader("Vital Signs & Lab Results")
                trestbps = st.slider("Resting Blood Pressure (mmHg)", min_value=90, max_value=200, value=130, step=1)
                thalach = st.slider("Maximum Heart Rate Achieved", min_value=60, max_value=202, value=130, step=1)
                chol = st.slider("Serum Cholesterol (mg/dL)", min_value=100, max_value=600, value=240, step=5)
                
                st.subheader("Test Results")
                fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                restecg = st.selectbox(
                    "Resting ECG Results",
                    options=[0, 1, 2],
                    format_func=lambda x: {
                        0: "Normal",
                        1: "ST-T Abnormality",
                        2: "LVH"
                    }.get(x)
                )
                
                st.subheader("Angiography & Other")
                ca = st.slider("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0, step=1)
                thal = st.selectbox(
                    "Thalassemia Type",
                    options=[0, 1, 2, 3],
                    format_func=lambda x: {
                        0: "Normal",
                        1: "Fixed Defect",
                        2: "Reversible Defect",
                        3: "Normal"
                    }.get(x)
                )
                slope = st.selectbox(
                    "Slope of ST Segment",
                    options=[0, 1, 2],
                    format_func=lambda x: {
                        0: "Upsloping",
                        1: "Flat",
                        2: "Downsloping"
                    }.get(x)
                )
            
            # Prediction Button
            st.markdown("---")
            
            if st.button("🔮 Predict Risk", use_container_width=True):
                # Prepare input data
                patient_data = pd.DataFrame({
                    'age': [age],
                    'sex': [sex],
                    'cp': [cp],
                    'trestbps': [trestbps],
                    'chol': [chol],
                    'fbs': [fbs],
                    'restecg': [restecg],
                    'thalach': [thalach],
                    'exang': [exang],
                    'oldpeak': [oldpeak],
                    'slope': [slope],
                    'ca': [ca],
                    'thal': [thal]
                })
                
                # Apply feature engineering
                engineer = FeatureEngineer()
                X_engineered = engineer.create_domain_features(patient_data)
                
                # Make prediction
                try:
                    prediction = predictor.model.predict(X_engineered)[0]
                    proba = predictor.model.predict_proba(X_engineered)[0][1]
                    
                    # Determine risk level
                    if proba < 0.3:
                        risk_level = "Low Risk"
                        risk_class = "risk-low"
                        emoji = "✅"
                    elif proba < 0.7:
                        risk_level = "Medium Risk"
                        risk_class = "risk-medium"
                        emoji = "⚠️"
                    else:
                        risk_level = "High Risk"
                        risk_class = "risk-high"
                        emoji = "❌"
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("🎯 Prediction Results")
                    
                    # Result Box
                    result_html = f"""
                    <div class="{risk_class}">
                        <h2>{emoji} {risk_level}</h2>
                        <p><strong>Heart Disease Probability: {proba*100:.1f}%</strong></p>
                        <p style="margin-top: 0.5rem;">Prediction: {'Heart Disease Likely' if prediction == 1 else 'No Heart Disease'}</p>
                    </div>
                    """
                    st.markdown(result_html, unsafe_allow_html=True)
                    
                    # Detailed Metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Probability Score", f"{proba*100:.1f}%", delta_color="off")
                    
                    with col2:
                        st.metric("Risk Category", risk_level, delta_color="off")
                    
                    with col3:
                        st.metric("Prediction", "Disease" if prediction == 1 else "No Disease", delta_color="off")
                    
                    # Health Insights
                    st.subheader("💡 Health Insights")
                    
                    insights = []
                    if trestbps > 140:
                        insights.append("⚠️ High resting blood pressure - Consider consulting a cardiologist")
                    if chol > 240:
                        insights.append("⚠️ High cholesterol - Lifestyle changes and medication may be needed")
                    if ca > 0:
                        insights.append("⚠️ Major vessels with calcification detected")
                    if oldpeak > 1:
                        insights.append("⚠️ Significant ST depression during exercise")
                    if exang == 1:
                        insights.append("⚠️ Exercise-induced angina present")
                    if age > 60 and thalach < 100:
                        insights.append("⚠️ Reduced heart rate response to exercise")
                    
                    if insights:
                        for insight in insights:
                            st.warning(insight)
                    else:
                        st.success("✅ No major risk factors detected in your measurements")
                    
                    # Recommendations
                    st.subheader("📋 Recommendations")
                    
                    if proba > 0.7:
                        st.error("""
                        **High Risk Category:**
                        - ⚠️ Schedule a comprehensive cardiac evaluation
                        - ⚠️ Discuss medication options with your doctor
                        - ⚠️ Consider additional diagnostic tests (ECG, stress test, cardiac imaging)
                        - ⚠️ Implement immediate lifestyle modifications
                        """)
                    elif proba > 0.3:
                        st.warning("""
                        **Medium Risk Category:**
                        - ⚠️ Schedule a check-up with your healthcare provider
                        - ⚠️ Monitor vital signs regularly
                        - ⚠️ Consider lifestyle modifications (diet, exercise, stress reduction)
                        - ⚠️ Reduce risk factors where possible
                        """)
                    else:
                        st.success("""
                        **Low Risk Category:**
                        - ✅ Continue healthy lifestyle habits
                        - ✅ Maintain regular exercise routine
                        - ✅ Keep cholesterol and blood pressure in check
                        - ✅ Regular health check-ups recommended
                        """)
                
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
        
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")

# ============================================================================
# PAGE 2: MODEL INFORMATION
# ============================================================================
elif page == "📊 Model Info":
    st.header("Model Information")
    
    model_path = project_dir / 'models' / 'best_heart_disease_model.pkl'
    
    if not model_path.exists():
        st.warning("Model file not found. Train the model first using ml_pipeline.py")
    else:
        with st.container():
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Model Details")
                st.info("""
                **Model Type:** Random Forest Classifier / Gradient Boosting
                
                **Training Data:** 615 samples
                **Test Data:** 154 samples
                
                **Features:** 13 clinical parameters
                
                **Target:** Binary classification (Disease vs No Disease)
                """)
            
            with col2:
                st.subheader("📊 Performance Metrics")
                st.info("""
                **Accuracy:** ~85-87%
                
                **Precision:** ~82-84%
                
                **Recall:** ~80-82%
                
                **F1-Score:** ~82-84%
                
                **ROC-AUC:** ~0.90+
                """)

# ============================================================================
# PAGE 3: ANALYTICS
# ============================================================================
elif page == "📈 Analytics":
    st.header("Analytics & Statistics")
    
    data_path = project_dir / 'data' / 'raw' / 'heart.csv'
    
    if data_path.exists():
        df = pd.read_csv(data_path)
        
        st.subheader("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", len(df))
        
        with col2:
            disease_count = (df['target'] == 1).sum()
            st.metric("Disease Cases", disease_count)
        
        with col3:
            no_disease = (df['target'] == 0).sum()
            st.metric("Healthy Cases", no_disease)
        
        with col4:
            disease_pct = (disease_count / len(df)) * 100
            st.metric("Disease %", f"{disease_pct:.1f}%")
        
        st.markdown("---")
        
        # Distribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Age Distribution")
            fig = px.histogram(df, x='age', nbins=20, title='Age Distribution', color_discrete_sequence=['#1f77b4'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Target Distribution")
            target_counts = df['target'].value_counts()
            st.bar_chart(target_counts)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Blood Pressure Distribution")
            fig = px.histogram(df, x='trestbps', nbins=20, title='Blood Pressure Distribution', color_discrete_sequence=['#ff7f0e'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Cholesterol Distribution")
            fig = px.histogram(df, x='chol', nbins=20, title='Cholesterol Distribution', color_discrete_sequence=['#2ca02c'])
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Heart Rate Distribution")
            fig = px.histogram(df, x='thalach', nbins=20, title='Heart Rate Distribution', color_discrete_sequence=['#d62728'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Sex Distribution")
            sex_counts = df['sex'].value_counts()
            sex_labels = {0: 'Female', 1: 'Male'}
            sex_counts.index = sex_counts.index.map(sex_labels)
            st.bar_chart(sex_counts)
    else:
        st.warning("Data file not found")

# ============================================================================
# PAGE 4: ABOUT
# ============================================================================
elif page == "ℹ️ About":
    st.header("About This Application")
    
    st.markdown("""
    ## Overview
    
    This Heart Disease Risk Predictor is a machine learning-based application designed to assess
    the risk of coronary heart disease based on clinical parameters.
    
    ## How It Works
    
    1. **Data Input:** Users enter their clinical measurements
    2. **Feature Engineering:** Features are enhanced with domain knowledge
    3. **Prediction:** A trained machine learning model makes a prediction
    4. **Risk Assessment:** Results are presented with risk level and recommendations
    
    ## Features
    
    - 🔍 **Real-time Prediction:** Get instant risk assessment
    - 📊 **Model Information:** View model performance metrics
    - 📈 **Data Analytics:** Explore dataset statistics
    - 💡 **Health Insights:** Personalized health recommendations
    
    ## Data Features
    
    The model uses 13 clinical parameters for prediction.
    
    ## Risk Levels
    
    - **Low Risk (< 30%):** No major concerns; continue healthy lifestyle
    - **Medium Risk (30-70%):** Monitor closely; consider lifestyle changes
    - **High Risk (> 70%):** Seek immediate medical consultation
    
    ## Disclaimer
    
    ⚠️ **Important:** This application is for educational and informational purposes only.
    It should NOT be used as a substitute for professional medical diagnosis or treatment.
    Always consult with a qualified healthcare provider for medical decisions.
    
    ## Technologies Used
    
    - **Python** - Programming language
    - **Streamlit** - Web application framework
    - **Scikit-learn** - Machine learning library
    - **Pandas & NumPy** - Data manipulation
    """)
    
    st.markdown("---")
    st.markdown("""
    **Version:** 1.0.0  
    **Last Updated:** March 2026  
    **GitHub:** [a-diallo-eng/ML-Engineering](https://github.com/a-diallo-eng/ML-Engineering)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em; margin-top: 2rem;'>
    <p>❤️ Heart Disease Risk Predictor | Disclaimer: For educational purposes only</p>
    <p>Always consult with a healthcare professional for medical advice</p>
</div>
""", unsafe_allow_html=True)


