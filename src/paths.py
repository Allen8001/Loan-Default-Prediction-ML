from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

LOAN_DATA_PATH = DATA_DIR / "Loan_default.csv"

MODEL_ARTIFACT_PATH = (
    MODELS_DIR / "credit_risk_model.joblib"
)