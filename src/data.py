"""Chargement et nettoyage du dataset Telco Customer Churn."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"

TARGET = "Churn"
ID_COL = "customerID"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_INT = ["SeniorCitizen"]
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC + BINARY_INT + CATEGORICAL


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie le dataset.

    - ``TotalCharges`` est lu comme texte car 11 valeurs sont des espaces vides.
      Ces clients ont ``tenure == 0`` (nouveaux abonnés, pas encore facturés) :
      la valeur correcte est donc 0, pas la médiane.
    - ``customerID`` est supprimé (identifiant sans pouvoir prédictif).
    - La cible ``Churn`` est convertie en 0/1.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    df = df.drop(columns=[ID_COL])
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def load_clean() -> pd.DataFrame:
    return clean(load_raw())
