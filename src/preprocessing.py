"""
preprocessing.py
================
Fonctions de nettoyage et d'encodage des données Telco Customer Churn.
Utilisé par tous les notebooks du projet.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_data(path: str) -> pd.DataFrame:
    """Charge le dataset brut depuis le chemin donné."""
    df = pd.read_csv(path)
    print(f"✅ Dataset chargé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    return df


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige la colonne TotalCharges :
    - Remplace les espaces vides par NaN
    - Convertit en float
    - Impute les valeurs manquantes par MonthlyCharges × tenure
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].replace(" ", np.nan), errors="coerce"
    )
    n_missing = df["TotalCharges"].isnull().sum()
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["MonthlyCharges"] * df["tenure"]
    )
    print(f"✅ TotalCharges corrigé ({n_missing} valeurs imputées)")
    return df


def encode_senior_citizen(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit SeniorCitizen (0/1) en string ('Non'/'Oui')."""
    df = df.copy()
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "Non", 1: "Oui"})
    return df


def encode_categorical(df: pd.DataFrame, cat_cols: list) -> tuple:
    """
    Encode les variables catégorielles avec LabelEncoder.

    Retourne :
        df_encoded (pd.DataFrame) : dataset encodé
        le_dict (dict)            : dictionnaire des encodeurs {col: LabelEncoder}
    """
    df_enc = df.copy()
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        le_dict[col] = le
    print(f"✅ {len(cat_cols)} variables encodées")
    return df_enc, le_dict


def get_variable_lists() -> tuple:
    """
    Retourne les listes officielles de variables du projet
    après tests statistiques et feature selection.

    Retourne :
        num_vars  : variables quantitatives
        cat_vars  : variables qualitatives retenues
        all_vars  : toutes les variables finales
        excluded  : variables exclues (non significatives)
    """
    num_vars = ["tenure", "MonthlyCharges", "TotalCharges"]

    cat_vars = [
        "Contract", "OnlineSecurity", "TechSupport", "OnlineBackup",
        "InternetService", "PaymentMethod", "PaperlessBilling",
        "SeniorCitizen", "Partner", "Dependents"
    ]

    excluded = ["gender", "PhoneService", "MultipleLines",
                "StreamingTV", "StreamingMovies", "DeviceProtection"]

    all_vars = num_vars + cat_vars
    return num_vars, cat_vars, all_vars, excluded


def full_preprocessing(path: str) -> tuple:
    """
    Pipeline complet de preprocessing en une seule fonction.

    Retourne :
        X (pd.DataFrame) : features encodées
        y (pd.Series)    : cible encodée (0/1)
        le_dict (dict)   : encodeurs pour décoder si besoin
        feature_names    : liste des noms de features
    """
    df = load_data(path)
    df = fix_total_charges(df)
    df = encode_senior_citizen(df)

    num_vars, cat_vars, all_vars, _ = get_variable_lists()

    df_sel = df[all_vars + ["Churn"]].copy()
    df_enc, le_dict = encode_categorical(df_sel, cat_vars + ["Churn"])

    X = df_enc[all_vars]
    y = df_enc["Churn"]

    print(f"✅ Preprocessing terminé : X={X.shape}, y={y.shape}")
    print(f"   Distribution cible : {y.value_counts().to_dict()}")
    return X, y, le_dict, all_vars


if __name__ == "__main__":
    # Test rapide
    X, y, le_dict, features = full_preprocessing(
        "../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )
    print(f"\nFeatures : {features}")
