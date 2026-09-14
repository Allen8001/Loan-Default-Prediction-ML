import numpy as np

from src.explainability import (
    explain_prediction,
    generate_plain_language_explanation,
)
from src.train_model import train_xgboost


def main():
    # Train model and retrieve untouched test data
    results = train_xgboost()

    pipeline = results["pipeline"]
    cost_threshold = results["cost_threshold"]

    X_test = results["X_test"]
    y_test = results["y_test"]

    # Get model risk scores for test applicants
    risk_scores = pipeline.predict_proba(X_test)[:, 1]

    # Select the applicant with the highest model risk score
    selected_position = int(
        np.argmax(risk_scores)
    )

    applicant = X_test.iloc[
        [selected_position]
    ]

    actual_label = int(
        y_test.iloc[selected_position]
    )

    risk_score = float(
        risk_scores[selected_position]
    )

    predicted_label = int(
        risk_score >= cost_threshold
    )

    print()
    print("=" * 60)
    print("SELECTED TEST APPLICANT")
    print("=" * 60)

    print(
        f"Model risk score : "
        f"{risk_score:.4f}"
    )

    print(
        f"Decision threshold: "
        f"{cost_threshold:.4f}"
    )

    print(
        f"Predicted class  : "
        f"{predicted_label}"
    )

    print(
        f"Actual class     : "
        f"{actual_label}"
    )

    print()
    print("Applicant Features")
    print("-" * 60)

    print(
        applicant.T.to_string(
            header=False
        )
    )

    # Generate SHAP explanation
    explanation = explain_prediction(
        pipeline,
        applicant,
        top_n=10,
    )

    print()
    print("=" * 60)
    print("TOP SHAP RISK DRIVERS")
    print("=" * 60)

    print(
        explanation.to_string(
            index=False
        )
    )

    plain_explanation = (
    generate_plain_language_explanation(
        explanation,
        top_n=5,
    )
)

    print()
    print("=" * 60)
    print("PLAIN-LANGUAGE EXPLANATION")
    print("=" * 60)

    print(plain_explanation)    


if __name__ == "__main__":
    main()