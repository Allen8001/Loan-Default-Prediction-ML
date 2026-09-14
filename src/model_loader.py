import joblib

from src.paths import MODEL_ARTIFACT_PATH


def load_model_artifact():
    if not MODEL_ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            "Model artifact not found. "
            "Run `python -m src.train_model` first."
        )

    artifact = joblib.load(
        MODEL_ARTIFACT_PATH
    )

    required_keys = [
        "pipeline",
        "model_name",
        "f1_threshold",
        "cost_threshold",
        "feature_columns",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in artifact
    ]

    if missing_keys:
        raise ValueError(
            "Model artifact is missing required fields: "
            + ", ".join(missing_keys)
        )

    return artifact


def get_pipeline():
    artifact = load_model_artifact()

    return artifact["pipeline"]


def get_cost_threshold():
    artifact = load_model_artifact()

    return artifact["cost_threshold"]