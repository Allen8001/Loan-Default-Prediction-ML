from pathlib import Path
import sys

import pandas as pd
import streamlit as st


# ==================================================
# Project imports
# ==================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_loan_data
from src.explainability import (
    explain_prediction,
    generate_plain_language_explanation,
)
from src.model_loader import load_model_artifact
from src.preprocessing import FEATURE_COLUMNS


# ==================================================
# Page config
# ==================================================
st.set_page_config(
    page_title="Clarity in Credit",
    page_icon="💳",
    layout="wide",
)


# ==================================================
# Styling
# ==================================================
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1350px;
        }

        .hero-title {
            font-size: 2.6rem;
            font-weight: 750;
            margin-bottom: 0.2rem;
        }

        .hero-subtitle {
            font-size: 1.05rem;
            color: #6b7280;
            margin-bottom: 1.8rem;
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 700;
            margin-top: 1rem;
            margin-bottom: 0.8rem;
        }

        .risk-card {
            border: 1px solid rgba(128,128,128,0.25);
            border-radius: 14px;
            padding: 1.2rem 1.4rem;
            margin-top: 0.5rem;
            margin-bottom: 1rem;
        }

        .risk-high {
            font-size: 1.5rem;
            font-weight: 700;
        }

        .risk-low {
            font-size: 1.5rem;
            font-weight: 700;
        }

        .small-note {
            font-size: 0.9rem;
            color: #6b7280;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128,128,128,0.18);
            padding: 14px;
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# Cached resources
# ==================================================
@st.cache_resource
def get_model_artifact():
    return load_model_artifact()


@st.cache_data
def get_reference_data():
    return load_loan_data()


artifact = get_model_artifact()
reference_data = get_reference_data()

pipeline = artifact["pipeline"]
cost_threshold = float(
    artifact["cost_threshold"]
)


# ==================================================
# Sidebar
# ==================================================
with st.sidebar:
    st.title("Model Overview")

    st.write("**Model**")
    st.write(
        artifact.get(
            "model_name",
            "XGBoost",
        )
    )

    st.write("**Decision strategy**")
    st.write(
        "Cost-sensitive threshold"
    )

    st.metric(
        "Decision Threshold",
        f"{cost_threshold:.3f}",
    )

    st.write("**Dataset**")
    st.write(
        f"{len(reference_data):,} loan records"
    )

    st.write("**Default Rate**")
    st.write(
        f"{reference_data['Default'].mean():.1%}"
    )

    st.divider()

    st.caption(
        "Portfolio demonstration only. "
        "This application is not intended "
        "for real lending decisions."
    )


# ==================================================
# Header
# ==================================================
st.markdown(
    """
    <div class="hero-title">
        Clarity in Credit
    </div>

    <div class="hero-subtitle">
        Explainable credit risk assessment using
        XGBoost, cost-sensitive decision thresholds
        and SHAP.
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# Overview metrics
# ==================================================
metric_1, metric_2, metric_3, metric_4 = (
    st.columns(4)
)

with metric_1:
    st.metric(
        "Loan Records",
        f"{len(reference_data):,}",
    )

with metric_2:
    st.metric(
        "ROC-AUC",
        "0.759",
    )

with metric_3:
    st.metric(
        "PR-AUC",
        "0.332",
    )

with metric_4:
    st.metric(
        "Cost Threshold",
        f"{cost_threshold:.3f}",
    )


st.info(
    "The model output is presented as a risk score, "
    "not as a calibrated probability of default."
)


# ==================================================
# Applicant form
# ==================================================
st.markdown(
    '<div class="section-title">'
    'Applicant Information'
    '</div>',
    unsafe_allow_html=True,
)

st.write(
    "Enter applicant and loan information to "
    "generate an explainable risk assessment."
)

with st.form(
    "applicant_form"
):

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------
    # Personal / employment
    # --------------------------------------------------
    with col1:
        st.markdown("#### Personal & Employment")

        age = st.number_input(
            "Age",
            min_value=int(
                reference_data[
                    "Age"
                ].min()
            ),
            max_value=int(
                reference_data[
                    "Age"
                ].max()
            ),
            value=int(
                reference_data[
                    "Age"
                ].median()
            ),
        )

        income = st.number_input(
            "Annual Income",
            min_value=int(
                reference_data[
                    "Income"
                ].min()
            ),
            max_value=int(
                reference_data[
                    "Income"
                ].max()
            ),
            value=int(
                reference_data[
                    "Income"
                ].median()
            ),
            step=1000,
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

        employment_type = st.selectbox(
            "Employment Type",
            sorted(
                reference_data[
                    "EmploymentType"
                ].unique()
            ),
        )

        education = st.selectbox(
            "Education",
            sorted(
                reference_data[
                    "Education"
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
    # Loan information
    # --------------------------------------------------
    with col2:
        st.markdown("#### Loan Details")

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

    # --------------------------------------------------
    # Credit profile
    # --------------------------------------------------
    with col3:
        st.markdown("#### Credit Profile")

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

        has_mortgage = st.selectbox(
            "Has Mortgage",
            sorted(
                reference_data[
                    "HasMortgage"
                ].unique()
            ),
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

    st.write("")

    submitted = st.form_submit_button(
        "Assess Credit Risk",
        type="primary",
        use_container_width=True,
    )


# ==================================================
# Prediction
# ==================================================
if submitted:

    applicant_data = pd.DataFrame(
        [
            {
                "Age": age,
                "Income": income,
                "LoanAmount": loan_amount,
                "CreditScore": credit_score,
                "MonthsEmployed": (
                    months_employed
                ),
                "NumCreditLines": (
                    num_credit_lines
                ),
                "InterestRate": (
                    interest_rate
                ),
                "LoanTerm": loan_term,
                "DTIRatio": dti_ratio,
                "Education": education,
                "EmploymentType": (
                    employment_type
                ),
                "MaritalStatus": (
                    marital_status
                ),
                "HasMortgage": (
                    has_mortgage
                ),
                "HasDependents": (
                    has_dependents
                ),
                "LoanPurpose": (
                    loan_purpose
                ),
                "HasCoSigner": (
                    has_cosigner
                ),
            }
        ]
    )

    applicant_data = (
        applicant_data[
            FEATURE_COLUMNS
        ]
    )

    risk_score = float(
        pipeline.predict_proba(
            applicant_data
        )[0, 1]
    )

    predicted_default = bool(
        risk_score
        >= cost_threshold
    )

    # ==================================================
    # Results
    # ==================================================
    st.divider()

    st.markdown(
        '<div class="section-title">'
        'Credit Risk Assessment'
        '</div>',
        unsafe_allow_html=True,
    )

    score_col, threshold_col, decision_col = (
        st.columns(3)
    )

    with score_col:
        st.metric(
            "Model Risk Score",
            f"{risk_score:.3f}",
        )

    with threshold_col:
        st.metric(
            "Decision Threshold",
            f"{cost_threshold:.3f}",
        )

    with decision_col:
        st.metric(
            "Risk Classification",
            (
                "Higher Risk"
                if predicted_default
                else "Lower Risk"
            ),
        )

    # --------------------------------------------------
    # Risk indicator
    # --------------------------------------------------
    st.write("#### Risk Score Position")

    st.progress(
        min(
            max(
                risk_score,
                0.0,
            ),
            1.0,
        )
    )

    st.caption(
        f"Risk score: {risk_score:.3f} | "
        f"Decision threshold: {cost_threshold:.3f}"
    )

    if predicted_default:
        st.warning(
            "Higher Risk — the applicant's model "
            "risk score is above the "
            "cost-sensitive decision threshold."
        )
    else:
        st.success(
            "Lower Risk — the applicant's model "
            "risk score is below the "
            "cost-sensitive decision threshold."
        )

    # ==================================================
    # Explainability
    # ==================================================
    explanation = explain_prediction(
        pipeline,
        applicant_data,
        top_n=10,
    )

    st.markdown(
        '<div class="section-title">'
        'Why did the model make this decision?'
        '</div>',
        unsafe_allow_html=True,
    )

    explanation_col, summary_col = (
        st.columns(
            [1.3, 1]
        )
    )

    # --------------------------------------------------
    # SHAP table
    # --------------------------------------------------
    with explanation_col:
        st.write(
            "#### Top Model Drivers"
        )

        display_explanation = (
            explanation.copy()
        )

        display_explanation[
            "shap_value"
        ] = (
            display_explanation[
                "shap_value"
            ].round(4)
        )

        display_explanation = (
            display_explanation.rename(
                columns={
                    "feature": "Feature",
                    "applicant_value": (
                        "Applicant Value"
                    ),
                    "shap_value": (
                        "SHAP Impact"
                    ),
                    "impact": (
                        "Direction"
                    ),
                }
            )
        )

        st.dataframe(
            display_explanation,
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------
    # Plain language explanation
    # --------------------------------------------------
    with summary_col:
        st.write(
            "#### Plain-Language Summary"
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
            "SHAP explains how features "
            "influenced this model prediction. "
            "It does not establish causal "
            "relationships."
        )

    # ==================================================
    # Applicant snapshot
    # ==================================================
    with st.expander(
        "View applicant data used by the model"
    ):
        st.dataframe(
            applicant_data,
            use_container_width=True,
            hide_index=True,
        )


# ==================================================
# Footer
# ==================================================
st.divider()

st.caption(
    "Explainable Credit Risk Assessment | "
    "XGBoost • SHAP • Streamlit | "
    "Portfolio project by Allen Wang"
)