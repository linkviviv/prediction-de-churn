"""Entraînement et comparaison des modèles de prédiction du churn.

Usage :  python -m src.train
Produit : models/churn_model.joblib  et  reports/metrics.json
"""
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .data import (BINARY_INT, CATEGORICAL, FEATURES, NUMERIC, ROOT, TARGET,
                   load_clean)

RANDOM_STATE = 42
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("bin", "passthrough", BINARY_INT),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), CATEGORICAL),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    """Les trois modèles de l'atelier, dans un Pipeline avec prétraitement."""
    estimators = {
        "Régression logistique": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Arbre de décision": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=20, class_weight="balanced",
            random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=25),
    }
    return {n: Pipeline([("prep", build_preprocessor()), ("model", m)])
            for n, m in estimators.items()}


def evaluate(pipe: Pipeline, X_test, y_test) -> dict:
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_test, proba)
    step = max(1, len(fpr) // 100)  # courbe allégée pour le JSON
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "roc_curve": {"fpr": fpr[::step].round(4).tolist(),
                      "tpr": tpr[::step].round(4).tolist()},
    }


def main() -> None:
    df = load_clean()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    fitted = {}
    for name, pipe in build_models().items():
        cvres = cross_validate(pipe, X_train, y_train, cv=cv,
                               scoring=["recall", "f1", "roc_auc"])
        pipe.fit(X_train, y_train)
        res = evaluate(pipe, X_test, y_test)
        res["cv"] = {k: float(np.mean(cvres[f"test_{k}"])) for k in ("recall", "f1", "roc_auc")}
        results[name] = res
        fitted[name] = pipe
        print(f"{name:24s} acc={res['accuracy']:.3f} recall={res['recall']:.3f} "
              f"f1={res['f1']:.3f} auc={res['roc_auc']:.3f} | CV auc={res['cv']['roc_auc']:.3f}")

    # Meilleur modèle : ROC-AUC en validation croisée (indépendant du seuil)
    best = max(results, key=lambda n: results[n]["cv"]["roc_auc"])
    print(f"\nMeilleur modèle : {best}")

    # Importance des variables (coefficients de la régression logistique)
    lr = fitted["Régression logistique"]
    names = lr.named_steps["prep"].get_feature_names_out()
    coefs = lr.named_steps["model"].coef_[0]
    importance = (pd.DataFrame({"feature": names, "coef": coefs})
                  .assign(abs_coef=lambda d: d.coef.abs())
                  .sort_values("abs_coef", ascending=False)
                  .head(15).drop(columns="abs_coef"))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"pipeline": fitted[best], "name": best,
                 "lr_pipeline": lr, "features": FEATURES}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({
        "best_model": best,
        "sklearn_version": sklearn.__version__,
        "n_train": len(X_train), "n_test": len(X_test),
        "churn_rate": float(y.mean()),
        "models": results,
        "top_coefficients": importance.round(4).to_dict("records"),
    }, indent=2, ensure_ascii=False))
    print(f"Modèle sauvegardé : {MODEL_PATH}")


def ensure_model() -> None:
    """Entraîne le modèle s'il manque ou s'il a été créé avec une autre version
    de scikit-learn (un modèle sérialisé n'est pas portable entre versions)."""
    up_to_date = False
    if MODEL_PATH.exists() and METRICS_PATH.exists():
        saved = json.loads(METRICS_PATH.read_text()).get("sklearn_version")
        up_to_date = saved == sklearn.__version__
    if not up_to_date:
        main()


if __name__ == "__main__":
    main()
