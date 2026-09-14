import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_loader import load_loan_data
from src.paths import MODEL_ARTIFACT_PATH
from src.preprocessing import build_preprocessor, split_features_target


RANDOM_STATE = 42

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

FALSE_NEGATIVE_COST = 5
FALSE_POSITIVE_COST = 1


def calculate_metrics(
    y_true,
    probabilities,
    threshold,
):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
    ).ravel()

    weighted_cost = (
        FALSE_NEGATIVE_COST * fn
        + FALSE_POSITIVE_COST * fp
    )

    return {
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "weighted_cost": float(
            weighted_cost
        ),
        "cost_per_case": float(
            weighted_cost / len(y_true)
        ),
    }


def find_best_f1_threshold(
    y_true,
    probabilities,
):
    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            probabilities,
        )
    )

    precision = precision[:-1]
    recall = recall[:-1]

    f1_scores = (
        2 * precision * recall
        / (
            precision
            + recall
            + 1e-12
        )
    )

    best_index = np.argmax(
        f1_scores
    )

    return (
        float(
            thresholds[best_index]
        ),
        float(
            f1_scores[best_index]
        ),
    )


def find_best_cost_threshold(
    y_true,
    probabilities,
):
    fpr, tpr, thresholds = roc_curve(
        y_true,
        probabilities,
    )

    positive_count = np.sum(
        y_true == 1
    )

    negative_count = np.sum(
        y_true == 0
    )

    false_negatives = (
        positive_count
        * (1 - tpr)
    )

    false_positives = (
        negative_count
        * fpr
    )

    costs = (
        FALSE_NEGATIVE_COST
        * false_negatives
        + FALSE_POSITIVE_COST
        * false_positives
    )

    valid_indices = np.isfinite(
        thresholds
    )

    valid_thresholds = (
        thresholds[
            valid_indices
        ]
    )

    valid_costs = (
        costs[
            valid_indices
        ]
    )

    best_index = np.argmin(
        valid_costs
    )

    return (
        float(
            valid_thresholds[
                best_index
            ]
        ),
        float(
            valid_costs[
                best_index
            ]
        ),
    )


def print_evaluation(
    title,
    metrics,
    threshold,
):
    print()
    print(title)
    print("-" * 50)

    print(
        f"threshold     : "
        f"{threshold:.4f}"
    )
    print(
        f"accuracy      : "
        f"{metrics['accuracy']:.4f}"
    )
    print(
        f"precision     : "
        f"{metrics['precision']:.4f}"
    )
    print(
        f"recall        : "
        f"{metrics['recall']:.4f}"
    )
    print(
        f"f1            : "
        f"{metrics['f1']:.4f}"
    )
    print(
        f"roc_auc       : "
        f"{metrics['roc_auc']:.4f}"
    )
    print(
        f"pr_auc        : "
        f"{metrics['pr_auc']:.4f}"
    )

    print()
    print("Confusion Matrix")

    print(
        f"TN: {metrics['tn']:,}    "
        f"FP: {metrics['fp']:,}"
    )

    print(
        f"FN: {metrics['fn']:,}    "
        f"TP: {metrics['tp']:,}"
    )

    print()
    print(
        f"weighted cost : "
        f"{metrics['weighted_cost']:,.0f}"
    )

    print(
        f"cost per case : "
        f"{metrics['cost_per_case']:.4f}"
    )


def train_xgboost():
    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    df = load_loan_data()

    X, y = split_features_target(
        df
    )

    # --------------------------------------------------
    # Split final untouched test set
    # --------------------------------------------------
    (
        X_dev,
        X_test,
        y_dev,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # --------------------------------------------------
    # Split development set into
    # train and validation
    # --------------------------------------------------
    (
        X_train,
        X_validation,
        y_train,
        y_validation,
    ) = train_test_split(
        X_dev,
        y_dev,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_dev,
    )

    print("Dataset Split")
    print("-" * 50)

    print(
        f"Train      : "
        f"{len(X_train):,}"
    )
    print(
        f"Validation : "
        f"{len(X_validation):,}"
    )
    print(
        f"Test       : "
        f"{len(X_test):,}"
    )

    print()
    print(
        f"Train default rate      : "
        f"{y_train.mean():.4f}"
    )
    print(
        f"Validation default rate : "
        f"{y_validation.mean():.4f}"
    )
    print(
        f"Test default rate       : "
        f"{y_test.mean():.4f}"
    )

    # --------------------------------------------------
    # Class imbalance weight
    # --------------------------------------------------
    negative_count = (
        y_train == 0
    ).sum()

    positive_count = (
        y_train == 1
    ).sum()

    scale_pos_weight = (
        negative_count
        / positive_count
    )

    print()
    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    print(
        f"FN cost assumption: "
        f"{FALSE_NEGATIVE_COST}x FP"
    )

    # --------------------------------------------------
    # XGBoost
    # --------------------------------------------------
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(
            scale_pos_weight
        ),
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                model,
            ),
        ]
    )

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    pipeline.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------
    # Validation probabilities
    # --------------------------------------------------
    validation_probabilities = (
        pipeline.predict_proba(
            X_validation
        )[:, 1]
    )

    (
        f1_threshold,
        best_validation_f1,
    ) = find_best_f1_threshold(
        y_validation,
        validation_probabilities,
    )

    (
        cost_threshold,
        best_validation_cost,
    ) = find_best_cost_threshold(
        y_validation,
        validation_probabilities,
    )

    print()
    print("=" * 50)
    print("VALIDATION RESULTS")
    print("=" * 50)

    default_validation = (
        calculate_metrics(
            y_validation,
            validation_probabilities,
            threshold=0.5,
        )
    )

    f1_validation = (
        calculate_metrics(
            y_validation,
            validation_probabilities,
            threshold=f1_threshold,
        )
    )

    cost_validation = (
        calculate_metrics(
            y_validation,
            validation_probabilities,
            threshold=cost_threshold,
        )
    )

    print_evaluation(
        "Validation - Default Threshold",
        default_validation,
        0.5,
    )

    print_evaluation(
        "Validation - F1 Optimised",
        f1_validation,
        f1_threshold,
    )

    print_evaluation(
        "Validation - Cost Sensitive",
        cost_validation,
        cost_threshold,
    )

    print()
    print(
        f"Best validation F1: "
        f"{best_validation_f1:.4f}"
    )

    print(
        f"Minimum validation cost: "
        f"{best_validation_cost:,.0f}"
    )

    # --------------------------------------------------
    # Final untouched test evaluation
    # --------------------------------------------------
    test_probabilities = (
        pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    default_test = calculate_metrics(
        y_test,
        test_probabilities,
        threshold=0.5,
    )

    f1_test = calculate_metrics(
        y_test,
        test_probabilities,
        threshold=f1_threshold,
    )

    cost_test = calculate_metrics(
        y_test,
        test_probabilities,
        threshold=cost_threshold,
    )

    print()
    print("=" * 50)
    print("FINAL TEST RESULTS")
    print("=" * 50)

    print_evaluation(
        "TEST - Default Threshold",
        default_test,
        0.5,
    )

    print_evaluation(
        "TEST - F1 Optimised",
        f1_test,
        f1_threshold,
    )

    print_evaluation(
        "TEST - Cost Sensitive",
        cost_test,
        cost_threshold,
    )

    # --------------------------------------------------
    # Save model artifact
    # --------------------------------------------------
    MODEL_ARTIFACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_artifact = {
        "pipeline": pipeline,
        "model_name": "XGBoost",
        "f1_threshold": (
            float(f1_threshold)
        ),
        "cost_threshold": (
            float(cost_threshold)
        ),
        "false_negative_cost": (
            FALSE_NEGATIVE_COST
        ),
        "false_positive_cost": (
            FALSE_POSITIVE_COST
        ),
        "scale_pos_weight": float(
            scale_pos_weight
        ),
        "feature_columns": list(
            X.columns
        ),
        "test_metrics_default": (
            default_test
        ),
        "test_metrics_f1": (
            f1_test
        ),
        "test_metrics_cost": (
            cost_test
        ),
    }

    joblib.dump(
        model_artifact,
        MODEL_ARTIFACT_PATH,
    )

    print()
    print("=" * 50)
    print("MODEL SAVED")
    print("=" * 50)
    print(
        MODEL_ARTIFACT_PATH
    )

    return {
        "pipeline": pipeline,
        "f1_threshold": (
            f1_threshold
        ),
        "cost_threshold": (
            cost_threshold
        ),
        "default_test": (
            default_test
        ),
        "f1_test": (
            f1_test
        ),
        "cost_test": (
            cost_test
        ),
        "X_test": X_test,
        "y_test": y_test,
    }


if __name__ == "__main__":
    train_xgboost()