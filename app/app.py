from pathlib import Path
import sys

import pandas as pd
import streamlit as st


# --------------------------------------------------
# Make project root importable
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.data_loader import load_loan_data
from src.explainability import (
    explain_prediction,
    generate_plain_language_explanation,
)
from src.model_loader import load_model_artifact
from src.preprocessing import FEATURE_COLUMNS


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Explainable Credit Risk",
    page_icon="📊",
    layout="wide",
)


# --------------------------------------------------
# Cached resources
# --------------------------------------------------
@st.cache_resource
def get_model_artifact():
    return load_model_artifact()


@st.cache_data
def get_reference_data():
    return load_loan_data()


artifact = get_model_artifact()
reference_data = get_reference_data()

pipeline = artifact["pipeline"]
cost_threshold = artifact["cost_threshold"]


# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("Explainable Credit Risk Assessment")

st.write(
    "Assess applicant credit risk using an XGBoost model "
    "with cost-sensitive decision thresholds and SHAP explanations."
)

st.info(
    "The model output below is presented as a risk score, "
    "not a calibrated probability of default."
)


# --------------------------------------------------
# Applicant form
# --------------------------------------------------
st.subheader("Applicant Information")

with st.form("applicant_form"):

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------
    # Column 1
    # --------------------------------------------------
    with col1:
        age = st.number_input(
            "Age",
            min_value=int(
                reference_data["Age"].min()
            ),
            max_value=int(
                reference_data["Age"].max()
            ),
            value=int(
                reference_data["Age"].median()
            ),
        )

        income = st.number_input(
            "Annual Income",
            min_value=int(
                reference_data["Income"].min()
            ),
            max_value=int(
                reference_data["Income"].max()
            ),
            value=int(
                reference_data["Income"].median()
            ),
            step=1000,
        )

        education = st.selectbox(
            "Education",
            sorted(
                reference_data[
                    "Education"
                ].unique()
            ),
        )

        employment_type = st.selectbox(
            "Employment Type",
            sorted(
                reference_data[
                    "EmploymentType"
                ].unique()
            ),
        )

        marital_status = st.selectbox(
            "Marital Status",
            sorted(
                reference_data[
                    "MaritalStatus"
                ].unique()
            ),
        )

    # --------------------------------------------------
    # Column 2
    # --------------------------------------------------
    with col2:
        loan_amount = st.number_input(
            "Loan Amount",
            min_value=int(
                reference_data[
                    "LoanAmount"
                ].min()
            ),
            max_value=int(
                reference_data[
                    "LoanAmount"
                ].max()
            ),
            value=int(
                reference_data[
                    "LoanAmount"
                ].median()
            ),
            step=1000,
        )

        interest_rate = st.number_input(
            "Interest Rate",
            min_value=float(
                reference_data[
                    "InterestRate"
                ].min()
            ),
            max_value=float(
                reference_data[
                    "InterestRate"
                ].max()
            ),
            value=float(
                reference_data[
                    "InterestRate"
                ].median()
            ),
            step=0.1,
            format="%.2f",
        )

        loan_term = st.number_input(
            "Loan Term",
            min_value=int(
                reference_data[
                    "LoanTerm"
                ].min()
            ),
            max_value=int(
                reference_data[
                    "LoanTerm"
                ].max()
            ),
            value=int(
                reference_data[
                    "LoanTerm"
                ].median()
            ),
        )

        loan_purpose = st.selectbox(
            "Loan Purpose",
            sorted(
                reference_data[
                    "LoanPurpose"
                ].unique()
            ),
        )

        has_mortgage = st.selectbox(
            "Has Mortgage",
            sorted(
                reference_data[
                    "HasMortgage"
                ].unique()
            ),
        )

    # --------------------------------------------------
    # Column 3
    # --------------------------------------------------
    with col3:
        credit_score = st.number_input(
            "Credit Score",
            min_value=int(
                reference_data[
                    "CreditScore"
                ].min()
            ),
            max_value=int(
                reference_data[
                    "CreditScore"
                ].max()
            ),
            value=int(
                reference_data[
                    "CreditScore"
                ].median()
            ),
        )

        months_employed = st.number_input(
            "Months Employed",
            min_value=int(
                reference_data[
                    "MonthsEmployed"
                ].min()
            ),
            max_value=int(
                reference_data[
                    "MonthsEmployed"
                ].max()
            ),
            value=int(
                reference_data[
                    "MonthsEmployed"
                ].median()
            ),
        )

        num_credit_lines = st.number_input(
            "Number of Credit Lines",
            min_value=int(
                reference_data[
                    "NumCreditLines"
                ].min()
            ),
            max_value=int(
                reference_data[
                    "NumCreditLines"
                ].max()
            ),
            value=int(
                reference_data[
                    "NumCreditLines"
                ].median()
            ),
        )

        dti_ratio = st.number_input(
            "Debt-to-Income Ratio",
            min_value=float(
                reference_data[
                    "DTIRatio"
                ].min()
            ),
            max_value=float(
                reference_data[
                    "DTIRatio"
                ].max()
            ),
            value=float(
                reference_data[
                    "DTIRatio"
                ].median()
            ),
            step=0.01,
            format="%.2f",
        )

        has_dependents = st.selectbox(
            "Has Dependents",
            sorted(
                reference_data[
                    "HasDependents"
                ].unique()
            ),
        )

        has_cosigner = st.selectbox(
            "Has Co-Signer",
            sorted(
                reference_data[
                    "HasCoSigner"
                ].unique()
            ),
        )

    submitted = st.form_submit_button(
        "Assess Credit Risk",
        type="primary",
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------
if submitted:

    applicant_data = pd.DataFrame(
        [
            {
                "Age": age,
                "Income": income,
                "LoanAmount": loan_amount,
                "CreditScore": credit_score,
                "MonthsEmployed": months_employed,
                "NumCreditLines": num_credit_lines,
                "InterestRate": interest_rate,
                "LoanTerm": loan_term,
                "DTIRatio": dti_ratio,
                "Education": education,
                "EmploymentType": employment_type,
                "MaritalStatus": marital_status,
                "HasMortgage": has_mortgage,
                "HasDependents": has_dependents,
                "LoanPurpose": loan_purpose,
                "HasCoSigner": has_cosigner,
            }
        ]
    )

    applicant_data = applicant_data[
        FEATURE_COLUMNS
    ]

    risk_score = float(
        pipeline.predict_proba(
            applicant_data
        )[0, 1]
    )

    predicted_default = (
        risk_score >= cost_threshold
    )

    # --------------------------------------------------
    # Result summary
    # --------------------------------------------------
    st.divider()

    st.subheader("Risk Assessment")

    result_col1, result_col2, result_col3 = (
        st.columns(3)
    )

    with result_col1:
        st.metric(
            "Model Risk Score",
            f"{risk_score:.3f}",
        )

    with result_col2:
        st.metric(
            "Decision Threshold",
            f"{cost_threshold:.3f}",
        )

    with result_col3:
        if predicted_default:
            st.metric(
                "Risk Classification",
                "Higher Risk",
            )
        else:
            st.metric(
                "Risk Classification",
                "Lower Risk",
            )

    if predicted_default:
        st.warning(
            "The applicant's model risk score is above "
            "the cost-sensitive decision threshold."
        )
    else:
        st.success(
            "The applicant's model risk score is below "
            "the cost-sensitive decision threshold."
        )

    # --------------------------------------------------
    # SHAP explanation
    # --------------------------------------------------
    st.subheader(
        "Prediction Explanation"
    )

    explanation = explain_prediction(
        pipeline,
        applicant_data,
        top_n=10,
    )

    display_explanation = explanation.copy()

    display_explanation[
        "shap_value"
    ] = (
        display_explanation[
            "shap_value"
        ].round(4)
    )

    st.dataframe(
        display_explanation,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------
    # Plain-language explanation
    # --------------------------------------------------
    st.subheader(
        "Plain-Language Summary"
    )

    plain_explanation = (
        generate_plain_language_explanation(
            explanation,
            top_n=5,
        )
    )

    st.write(
        plain_explanation
    )

    st.caption(
        "SHAP values explain how features influenced "
        "this model prediction. They do not establish "
        "causal relationships."
    )