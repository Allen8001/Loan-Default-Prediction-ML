import pandas as pd
import shap

from src.preprocessing import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


def explain_prediction(
    pipeline,
    applicant_data: pd.DataFrame,
    top_n: int = 8,
):
    """
    Generate a human-readable SHAP explanation
    for one applicant.

    Numerical features are displayed using their
    original values.

    One-hot encoded categorical features are
    aggregated back into their original feature.
    """

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    model = pipeline.named_steps[
        "model"
    ]

    # --------------------------------------------------
    # Transform applicant using fitted preprocessing
    # --------------------------------------------------
    transformed_data = preprocessor.transform(
        applicant_data
    )

    if hasattr(
        transformed_data,
        "toarray",
    ):
        transformed_data = (
            transformed_data.toarray()
        )

    # --------------------------------------------------
    # Calculate SHAP values
    # --------------------------------------------------
    explainer = shap.TreeExplainer(
        model
    )

    shap_explanation = explainer(
        transformed_data
    )

    shap_values = shap_explanation.values

    # Handle possible multiclass-style output
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    shap_values = shap_values[0]

    explanation_rows = []

    # --------------------------------------------------
    # Numeric features
    # --------------------------------------------------
    for index, feature in enumerate(
        NUMERIC_FEATURES
    ):
        raw_value = applicant_data.iloc[0][
            feature
        ]

        feature_shap = float(
            shap_values[index]
        )

        explanation_rows.append(
            {
                "feature": feature,
                "applicant_value": raw_value,
                "shap_value": feature_shap,
            }
        )

    # --------------------------------------------------
    # Categorical features
    #
    # OneHotEncoder creates multiple columns for each
    # original feature. Sum their SHAP contributions
    # back into one business-level feature.
    # --------------------------------------------------
    categorical_encoder = (
        preprocessor.named_transformers_[
            "categorical"
        ]
    )

    position = len(
        NUMERIC_FEATURES
    )

    for feature, categories in zip(
        CATEGORICAL_FEATURES,
        categorical_encoder.categories_,
    ):
        number_of_categories = len(
            categories
        )

        feature_shap_values = (
            shap_values[
                position:
                position + number_of_categories
            ]
        )

        aggregated_shap = float(
            feature_shap_values.sum()
        )

        raw_value = applicant_data.iloc[0][
            feature
        ]

        explanation_rows.append(
            {
                "feature": feature,
                "applicant_value": raw_value,
                "shap_value": aggregated_shap,
            }
        )

        position += number_of_categories

    # --------------------------------------------------
    # Build readable explanation table
    # --------------------------------------------------
    explanation_df = pd.DataFrame(
        explanation_rows
    )

    explanation_df[
        "absolute_shap"
    ] = (
        explanation_df[
            "shap_value"
        ].abs()
    )

    explanation_df[
        "impact"
    ] = explanation_df[
        "shap_value"
    ].apply(
        lambda value:
        "Higher default risk"
        if value > 0
        else "Lower default risk"
    )

    explanation_df = (
        explanation_df
        .sort_values(
            "absolute_shap",
            ascending=False,
        )
        .head(top_n)
    )

    return explanation_df[
        [
            "feature",
            "applicant_value",
            "shap_value",
            "impact",
        ]
    ]

def generate_plain_language_explanation(
    explanation_df: pd.DataFrame,
    top_n: int = 5,
) -> str:
    """
    Convert SHAP results into a simple,
    human-readable credit risk explanation.
    """

    top_features = explanation_df.head(top_n)

    higher_risk = top_features[
        top_features["shap_value"] > 0
    ]

    lower_risk = top_features[
        top_features["shap_value"] < 0
    ]

    sentences = []

    if not higher_risk.empty:
        higher_parts = []

        for _, row in higher_risk.iterrows():
            feature = row["feature"]
            value = row["applicant_value"]

            higher_parts.append(
                f"{feature} ({value})"
            )

        sentences.append(
            "The main factors pushing the model "
            "toward higher predicted default risk were "
            + ", ".join(higher_parts)
            + "."
        )

    if not lower_risk.empty:
        lower_parts = []

        for _, row in lower_risk.iterrows():
            feature = row["feature"]
            value = row["applicant_value"]

            lower_parts.append(
                f"{feature} ({value})"
            )

        sentences.append(
            "Factors partially reducing the predicted "
            "risk included "
            + ", ".join(lower_parts)
            + "."
        )

    sentences.append(
        "These factors describe how the model arrived "
        "at this prediction and should not be interpreted "
        "as causal relationships."
    )

    return " ".join(sentences)