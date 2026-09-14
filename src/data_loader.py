import pandas as pd

from src.paths import LOAN_DATA_PATH


def load_loan_data() -> pd.DataFrame:
    if not LOAN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {LOAN_DATA_PATH}"
        )

    return pd.read_csv(LOAN_DATA_PATH)