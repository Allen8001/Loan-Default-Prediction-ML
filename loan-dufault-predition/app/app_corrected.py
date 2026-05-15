"""
Clarity in Credit - CORRECTED AI-Powered Loan Risk Assessment Platform
Uses corrected feature scaling that preserves logical relationships between features and default risk
"""

# Fix Fortran runtime issues BEFORE any other imports
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

from src.real_data_loader_corrected import get_real_data_samples, prepare_sample_for_prediction_corrected
from src.ai_explanations_enhanced import EnhancedAIExplanationGenerator, create_enhanced_comprehensive_explanation
from src.paths import CORRECTED_SCALING_MODEL, WORKING_MODEL, get_model_path

class CorrectedLoanRiskAssessment:
    """Corrected loan risk assessment with proper feature relationships."""
    
    def __init__(self):
        self.model_data = None
        self.load_corrected_model()
    
    def load_corrected_model(self):
        """Load the corrected model with proper feature scaling."""
        try:
            self.model_data = joblib.load(get_model_path(CORRECTED_SCALING_MODEL))
            st.success("✅ Loaded corrected model with proper feature relationships")
        except FileNotFoundError:
            try:
                self.model_data = joblib.load(get_model_path(WORKING_MODEL))
                st.warning("⚠️ Using fallback model - feature relationships may not be optimal")
            except FileNotFoundError:
                st.error("❌ No model found. Please run the model training first.")
                return
        
        print(f"Model type: {self.model_data.get('model_type', 'Unknown')}")
        print(f"Model version: {self.model_data.get('model_version', 'Unknown')}")
        print(f"Scaling method: {self.model_data.get('scaling_method', 'Unknown')}")
    
    def get_risk_indicator_corrected(self, probability):
        """Return risk indicator styling based on probability with corrected thresholds."""
        if probability < 0.15:
            return "risk-low", "LOW RISK", "🟢"
        elif probability < 0.35:
            return "risk-medium", "MEDIUM RISK", "🟡"
        elif probability < 0.60:
            return "risk-high", "HIGH RISK", "🟠"
        else:
            return "risk-critical", "CRITICAL RISK", "🔴"
    
    def get_decision_recommendation_corrected(self, probability):
        """Get corrected decision recommendation based on probability."""
        if probability < 0.10:
            return "✅ AUTO-APPROVE", "success", "Excellent credit profile with minimal risk"
        elif probability < 0.20:
            return "✅ APPROVE - STANDARD TERMS", "success", "Good credit profile with acceptable risk"
        elif probability < 0.35:
            return "⚠️ APPROVE - CONDITIONAL TERMS", "warning", "Moderate risk requiring enhanced terms"
        elif probability < 0.50:
            return "🔍 MANUAL REVIEW REQUIRED", "warning", "High risk requiring human evaluation"
        else:
            return "❌ DECLINE RECOMMENDED", "error", "Excessive risk for responsible lending"
    
    def run_prediction_corrected(self, form_inputs):
        """Run prediction with corrected model and scaling."""
        try:
            if 'scalers' in self.model_data:
                processed_sample = prepare_sample_for_prediction_corrected(
                    form_inputs, 
                    self.model_data['scalers'], 
                    self.model_data['feature_names']
                )
            else:
                from src.real_data_loader import prepare_sample_for_prediction
                processed_sample = prepare_sample_for_prediction(
                    form_inputs,
                    self.model_data.get('scaler'),
                    self.model_data['feature_names']
                )
            
            model = self.model_data['model']
            prediction = model.predict(processed_sample)[0]
            probability = model.predict_proba(processed_sample)[0, 1]
            
            explanation_data = {
                'prediction': prediction,
                'probability': probability,
                'raw_input_data': form_inputs
            }
            
            explanation_generator = EnhancedAIExplanationGenerator()
            explanation = create_enhanced_comprehensive_explanation(explanation_data)
            
            return {
                'prediction': prediction,
                'probability': probability,
                'explanation': explanation,
                'model_type': self.model_data.get('model_type', 'Unknown'),
                'model_version': self.model_data.get('model_version', 'Unknown'),
                'scaling_method': self.model_data.get('scaling_method', 'Unknown')
            }
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            return None
    
    def display_corrected_results(self, results):
        """Display results with corrected decision logic."""
        if not results:
            return
        
        probability = results['probability']
        prediction = results['prediction']
        explanation = results['explanation']
        
        risk_class, risk_text, risk_icon = self.get_risk_indicator_corrected(probability)
        decision, decision_color, decision_reasoning = self.get_decision_recommendation_corrected(probability)
        
        st.markdown(f"""
        <div class="risk-indicator {risk_class}">
            <h3>{risk_icon} {risk_text}</h3>
            <p>Default Probability: {probability:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        getattr(st, decision_color)(f"**{decision}**")
        st.info(f"**Reasoning:** {decision_reasoning}")
        
        st.markdown("### 🔧 Model Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Type", results.get('model_type', 'Unknown'))
        with col2:
            st.metric("Model Version", results.get('model_version', 'Unknown'))
        with col3:
            st.metric("Scaling Method", results.get('scaling_method', 'Unknown'))
        
        st.markdown("### 🤖 AI Explanation")
        st.markdown(explanation)

def main():
    """Main application function."""
    st.set_page_config(
        page_title="Clarity in Credit - CORRECTED",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .risk-critical {
        background-color: #fef2f2;
        color: #dc2626;
        border-color: #fecaca;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 1rem 0;
        border: 2px solid;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-header">🏦 Clarity in Credit</div>
    <div class="subtitle">CORRECTED AI-Powered Loan Risk Assessment</div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="real-data-banner">
        ✅ CORRECTED MODEL: Proper feature relationships preserved<br>
        Higher Credit Score → Lower Default Risk | Lower DTI → Lower Default Risk
    </div>
    """, unsafe_allow_html=True)
    
    if 'assessment' not in st.session_state:
        st.session_state.assessment = CorrectedLoanRiskAssessment()
    
    assessment = st.session_state.assessment
    
    with st.sidebar:
        st.markdown("### 📝 Loan Application Details")
        
        if 'form_inputs' not in st.session_state:
            st.session_state.form_inputs = {
                'Age': 35, 'Income': 50000, 'LoanAmount': 25000,
                'CreditScore': 650, 'MonthsEmployed': 36, 'NumCreditLines': 4,
                'InterestRate': 12.0, 'LoanTerm': 36, 'DTIRatio': 0.5,
                'Education': "Bachelor's", 'EmploymentType': 'Full-time',
                'MaritalStatus': 'Married', 'HasMortgage': 'No',
                'HasDependents': 'No', 'LoanPurpose': 'Auto', 'HasCoSigner': 'No'
            }
        
        age = st.slider("Age", 18, 80, st.session_state.form_inputs['Age'])
        income = st.number_input("Annual Income ($)", 10000, 500000, st.session_state.form_inputs['Income'])
        loan_amount = st.number_input("Loan Amount ($)", 1000, 500000, st.session_state.form_inputs['LoanAmount'])
        
        credit_score = st.slider("Credit Score", 300, 850, st.session_state.form_inputs['CreditScore'])
        months_employed = st.slider("Months Employed", 0, 120, st.session_state.form_inputs['MonthsEmployed'])
        num_credit_lines = st.slider("Number of Credit Lines", 1, 20, st.session_state.form_inputs['NumCreditLines'])
        
        interest_rate = st.slider("Interest Rate (%)", 3.0, 30.0, st.session_state.form_inputs['InterestRate'])
        loan_term = st.slider("Loan Term (months)", 12, 84, st.session_state.form_inputs['LoanTerm'])
        
        dti_ratio = st.slider("Debt-to-Income Ratio", 0.1, 1.0, st.session_state.form_inputs['DTIRatio'])
        
        education = st.selectbox("Education", 
                                ["Bachelor's", "High School", "Master's", "PhD"],
                                index=["Bachelor's", "High School", "Master's", "PhD"].index(st.session_state.form_inputs['Education']))
        
        employment_type = st.selectbox("Employment Type",
                                     ["Full-time", "Part-time", "Self-employed", "Unemployed"],
                                     index=["Full-time", "Part-time", "Self-employed", "Unemployed"].index(st.session_state.form_inputs['EmploymentType']))
        
        marital_status = st.selectbox("Marital Status",
                                    ["Single", "Married", "Divorced"],
                                    index=["Single", "Married", "Divorced"].index(st.session_state.form_inputs['MaritalStatus']))
        
        has_mortgage = st.selectbox("Has Mortgage", ["Yes", "No"],
                                  index=["Yes", "No"].index(st.session_state.form_inputs['HasMortgage']))
        
        has_dependents = st.selectbox("Has Dependents", ["Yes", "No"],
                                    index=["Yes", "No"].index(st.session_state.form_inputs['HasDependents']))
        
        loan_purpose = st.selectbox("Loan Purpose",
                                  ["Auto", "Business", "Education", "Home", "Other"],
                                  index=["Auto", "Business", "Education", "Home", "Other"].index(st.session_state.form_inputs['LoanPurpose']))
        
        has_cosigner = st.selectbox("Has Co-signer", ["Yes", "No"],
                                  index=["Yes", "No"].index(st.session_state.form_inputs['HasCoSigner']))
        
        st.session_state.form_inputs = {
            'Age': age, 'Income': income, 'LoanAmount': loan_amount,
            'CreditScore': credit_score, 'MonthsEmployed': months_employed,
            'NumCreditLines': num_credit_lines, 'InterestRate': interest_rate,
            'LoanTerm': loan_term, 'DTIRatio': dti_ratio,
            'Education': education, 'EmploymentType': employment_type,
            'MaritalStatus': marital_status, 'HasMortgage': has_mortgage,
            'HasDependents': has_dependents, 'LoanPurpose': loan_purpose,
            'HasCoSigner': has_cosigner
        }
        
        st.markdown("---")
        st.markdown("### 📊 Real-Time Risk Indicators")
        
        if credit_score >= 750:
            st.success(f"💳 Credit: {credit_score} (Excellent)")
        elif credit_score >= 700:
            st.info(f"💳 Credit: {credit_score} (Good)")
        elif credit_score >= 600:
            st.warning(f"💳 Credit: {credit_score} (Fair)")
        else:
            st.error(f"💳 Credit: {credit_score} (Poor)")
        
        if dti_ratio < 0.3:
            st.success(f"📊 DTI: {dti_ratio:.1%} (Excellent)")
        elif dti_ratio < 0.4:
            st.info(f"📊 DTI: {dti_ratio:.1%} (Good)")
        elif dti_ratio < 0.5:
            st.warning(f"📊 DTI: {dti_ratio:.1%} (Fair)")
        else:
            st.error(f"📊 DTI: {dti_ratio:.1%} (High Risk)")
        
        if st.button("🔍 Assess Loan Risk", type="primary"):
            with st.spinner("Analyzing loan application..."):
                results = assessment.run_prediction_corrected(st.session_state.form_inputs)
                if results:
                    st.session_state.prediction_results = results
                    st.rerun()
    
    if 'prediction_results' in st.session_state:
        assessment.display_corrected_results(st.session_state.prediction_results)
    
    st.markdown("### 🧪 Test with Sample Data")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📈 Excellent Credit Sample"):
            sample_data = get_real_data_samples()["excellent_credit"]
            st.session_state.form_inputs = sample_data
            st.rerun()
    
    with col2:
        if st.button("📉 Poor Credit Sample"):
            sample_data = get_real_data_samples()["poor_credit"]
            st.session_state.form_inputs = sample_data
            st.rerun()
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-size: 0.9rem;">
        ✅ <strong>CORRECTED MODEL</strong> - Proper feature relationships preserved<br>
        Higher Credit Score → Lower Default Risk | Lower DTI → Lower Default Risk
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
