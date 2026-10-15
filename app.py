"""Interface Streamlit — Prédiction du churn client (Telco).

Lancer :  streamlit run app.py
"""
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data import FEATURES, TARGET, load_clean
from src.train import METRICS_PATH, MODEL_PATH, ensure_model

st.set_page_config(page_title="Prédiction du churn", page_icon="📉", layout="wide")

CHURN_COLOR, STAY_COLOR = "#d1495b", "#2e86ab"


def show(fig):
    """Affiche une figure matplotlib puis la ferme (évite l'accumulation en mémoire)."""
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Chargement (mis en cache)
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text())


@st.cache_data
def load_data():
    return load_clean()


@st.cache_resource(show_spinner="Entraînement du modèle (première exécution, ~30 s)…")
def prepare():
    ensure_model()


prepare()
bundle, metrics, data = load_model(), load_metrics(), load_data()


# --------------------------------------------------------------------------- #
# Aides
# --------------------------------------------------------------------------- #
def explain(row: pd.DataFrame, top: int = 6) -> pd.DataFrame:
    """Contribution de chaque variable au score (régression logistique)."""
    pipe = bundle["lr_pipeline"]
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    x = prep.transform(row)
    x = x.toarray()[0] if hasattr(x, "toarray") else x[0]
    contrib = pd.Series(model.coef_[0] * x, index=prep.get_feature_names_out())
    contrib.index = [i.split("__", 1)[1].replace("_", " = ") for i in contrib.index]
    contrib = contrib[contrib.abs() > 1e-6]
    return contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(top)


def churn_rate_by(col: str) -> pd.Series:
    return data.groupby(col)[TARGET].mean().sort_values() * 100


# --------------------------------------------------------------------------- #
# Barre latérale
# --------------------------------------------------------------------------- #
st.sidebar.title("📉 Churn Telco")
page = st.sidebar.radio("Navigation",
                        ["Prédiction", "Exploration des données", "Performance des modèles"])
st.sidebar.caption(f"Modèle actif : **{bundle['name']}**")
st.sidebar.caption(f"{len(data):,} clients · taux de churn {metrics['churn_rate']:.1%}")


# --------------------------------------------------------------------------- #
# Page 1 — Prédiction
# --------------------------------------------------------------------------- #
if page == "Prédiction":
    st.title("Ce client va-t-il quitter l'opérateur ?")
    st.write("Renseignez le profil du client pour estimer son risque de churn.")

    threshold = st.sidebar.slider(
        "Seuil d'alerte", 0.10, 0.90, 0.50, 0.05,
        help="Un client est signalé « à risque » si sa probabilité de churn dépasse ce seuil. "
             "Baissez-le pour détecter plus de churners (au prix de plus de fausses alertes).")

    with st.form("client"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Profil")
            gender = st.selectbox("Genre", ["Female", "Male"])
            senior = st.selectbox("Senior", [0, 1], format_func=lambda v: "Oui" if v else "Non")
            partner = st.selectbox("En couple", ["No", "Yes"])
            dependents = st.selectbox("Personnes à charge", ["No", "Yes"])
            tenure = st.slider("Ancienneté (mois)", 0, 72, 12)
        with c2:
            st.subheader("Services")
            phone = st.selectbox("Service téléphonique", ["Yes", "No"])
            multi = st.selectbox("Lignes multiples", ["No", "Yes"])
            internet = st.selectbox("Internet", ["Fiber optic", "DSL", "No"])
            security = st.selectbox("Sécurité en ligne", ["No", "Yes"])
            backup = st.selectbox("Sauvegarde en ligne", ["No", "Yes"])
            protection = st.selectbox("Protection appareil", ["No", "Yes"])
            support = st.selectbox("Support technique", ["No", "Yes"])
            tv = st.selectbox("Streaming TV", ["No", "Yes"])
            movies = st.selectbox("Streaming films", ["No", "Yes"])
        with c3:
            st.subheader("Contrat & facturation")
            contract = st.selectbox("Contrat", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Facture électronique", ["Yes", "No"])
            payment = st.selectbox("Paiement", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly = st.number_input("Frais mensuels", 18.0, 120.0, 70.0, 0.5)
            total = st.number_input("Frais totaux (0 = estimé)", 0.0, 9000.0, 0.0, 10.0,
                                    help="Laissé à 0, il est estimé : ancienneté × frais mensuels.")
        submitted = st.form_submit_button("Estimer le risque", type="primary")

    if submitted:
        # Cohérence des modalités « sans service »
        if phone == "No":
            multi = "No phone service"
        if internet == "No":
            security = backup = protection = support = tv = movies = "No internet service"
        total = total or tenure * monthly

        row = pd.DataFrame([{
            "tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total,
            "SeniorCitizen": senior, "gender": gender, "Partner": partner,
            "Dependents": dependents, "PhoneService": phone, "MultipleLines": multi,
            "InternetService": internet, "OnlineSecurity": security,
            "OnlineBackup": backup, "DeviceProtection": protection,
            "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies,
            "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        }])[FEATURES]

        proba = float(bundle["pipeline"].predict_proba(row)[0, 1])
        at_risk = proba >= threshold

        st.divider()
        m1, m2 = st.columns([1, 2])
        with m1:
            st.metric("Probabilité de churn", f"{proba:.0%}")
            st.progress(proba)
            if at_risk:
                st.error("⚠️ Client à risque — action de rétention recommandée")
            else:
                st.success("✅ Client probablement fidèle")
            st.caption(f"Seuil d'alerte : {threshold:.0%}")
        with m2:
            st.markdown("**Facteurs qui pèsent le plus sur ce score**")
            contrib = explain(row)
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.barh(contrib.index[::-1], contrib.values[::-1],
                    color=[CHURN_COLOR if v > 0 else STAY_COLOR for v in contrib.values[::-1]])
            ax.axvline(0, color="#888", lw=0.8)
            ax.set_xlabel("← réduit le risque  |  augmente le risque →")
            ax.spines[["top", "right"]].set_visible(False)
            show(fig)
            st.caption("Contributions issues de la régression logistique (variables standardisées).")


# --------------------------------------------------------------------------- #
# Page 2 — Exploration
# --------------------------------------------------------------------------- #
elif page == "Exploration des données":
    st.title("Exploration des données")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Clients", f"{len(data):,}")
    k2.metric("Taux de churn", f"{data[TARGET].mean():.1%}")
    k3.metric("Frais mensuels moyens", f"{data['MonthlyCharges'].mean():.2f}")
    k4.metric("Ancienneté moyenne", f"{data['tenure'].mean():.0f} mois")

    st.info("La cible est **déséquilibrée** (≈ 73 % de clients fidèles contre 27 % de churners) : "
            "l'accuracy seule est donc trompeuse, on suit aussi le recall et le ROC-AUC.")

    left, right = st.columns(2)
    for col, title, container in [("tenure", "Ancienneté (mois)", left),
                                  ("MonthlyCharges", "Frais mensuels", right)]:
        with container:
            fig, ax = plt.subplots(figsize=(5, 3))
            for val, label, color in [(0, "Reste", STAY_COLOR), (1, "Part", CHURN_COLOR)]:
                ax.hist(data.loc[data[TARGET] == val, col], bins=30, alpha=0.65,
                        label=label, color=color, density=True)
            ax.set_title(f"{title} selon le churn")
            ax.legend()
            ax.spines[["top", "right"]].set_visible(False)
            show(fig)

    st.subheader("Taux de churn par catégorie")
    cat = st.selectbox("Variable", ["Contract", "InternetService", "PaymentMethod",
                                    "TechSupport", "OnlineSecurity", "PaperlessBilling",
                                    "SeniorCitizen", "Partner", "Dependents"])
    rates = churn_rate_by(cat)
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.barh(rates.index.astype(str), rates.values, color=CHURN_COLOR)
    for i, v in enumerate(rates.values):
        ax.text(v + 0.5, i, f"{v:.0f} %", va="center")
    ax.set_xlabel("Taux de churn (%)")
    ax.spines[["top", "right"]].set_visible(False)
    show(fig)

    with st.expander("Aperçu du dataset nettoyé"):
        st.dataframe(data.head(100))


# --------------------------------------------------------------------------- #
# Page 3 — Performance
# --------------------------------------------------------------------------- #
else:
    st.title("Performance des modèles")
    st.caption(f"Jeu de test : {metrics['n_test']:,} clients (split stratifié 80/20). "
               "Seuil de décision : 0,5.")

    table = pd.DataFrame({
        name: {"Accuracy": m["accuracy"], "Precision": m["precision"],
               "Recall": m["recall"], "F1": m["f1"], "ROC-AUC": m["roc_auc"],
               "ROC-AUC (CV 5 plis)": m["cv"]["roc_auc"]}
        for name, m in metrics["models"].items()}).T
    st.dataframe(table.style.format("{:.3f}").highlight_max(
                     axis=0, props="background-color:#cfe8d5;color:#0b3d1e;font-weight:600"),
                 width="stretch")
    st.success(f"Modèle retenu : **{metrics['best_model']}** (meilleur ROC-AUC en validation croisée).")

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(5, 4))
        for name, m in metrics["models"].items():
            ax.plot(m["roc_curve"]["fpr"], m["roc_curve"]["tpr"],
                    label=f"{name} ({m['roc_auc']:.2f})")
        ax.plot([0, 1], [0, 1], "--", color="#999")
        ax.set_xlabel("Faux positifs")
        ax.set_ylabel("Vrais positifs")
        ax.set_title("Courbes ROC")
        ax.legend(loc="lower right", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        show(fig)
    with right:
        choice = st.selectbox("Matrice de confusion", list(metrics["models"]))
        cm = metrics["models"][choice]["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=14,
                        color="white" if cm[i][j] > max(map(max, cm)) / 2 else "black")
        ax.set_xticks([0, 1], ["Reste", "Part"])
        ax.set_yticks([0, 1], ["Reste", "Part"])
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        show(fig)

    st.subheader("Variables les plus influentes (régression logistique)")
    coefs = pd.DataFrame(metrics["top_coefficients"])
    coefs["feature"] = coefs["feature"].str.split("__", n=1).str[1].str.replace("_", " = ")
    coefs = coefs.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(coefs["feature"], coefs["coef"],
            color=[CHURN_COLOR if c > 0 else STAY_COLOR for c in coefs["coef"]])
    ax.axvline(0, color="#888", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    show(fig)
