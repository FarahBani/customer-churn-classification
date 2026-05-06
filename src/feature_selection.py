"""
feature_selection.py
====================
Fonctions de sélection de variables :
SelectKBest, RFE, Feature Importance RandomForest.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier


def select_kbest(X: pd.DataFrame, y: pd.Series, k: int = 10) -> pd.DataFrame:
    """
    Sélection par score F (ANOVA).

    Retourne un DataFrame trié par score décroissant.
    """
    skb = SelectKBest(f_classif, k=k)
    skb.fit(X, y)
    result = pd.DataFrame({
        "Variable": X.columns,
        "F-score": skb.scores_.round(2),
        "p-value": skb.pvalues_,
        "Sélectionnée": skb.get_support()
    }).sort_values("F-score", ascending=False).reset_index(drop=True)
    print(f"✅ SelectKBest : {k} variables sélectionnées")
    return result


def select_rfe(X: pd.DataFrame, y: pd.Series,
               n_features: int = 10,
               n_estimators: int = 100) -> pd.DataFrame:
    """
    Sélection par RFE avec RandomForest.

    Retourne un DataFrame avec le rang de chaque variable.
    """
    rf = RandomForestClassifier(n_estimators=n_estimators,
                                 random_state=42, n_jobs=-1)
    rfe = RFE(estimator=rf, n_features_to_select=n_features, step=1)
    rfe.fit(X, y)
    result = pd.DataFrame({
        "Variable": X.columns,
        "Rang RFE": rfe.ranking_,
        "Sélectionnée": rfe.support_
    }).sort_values("Rang RFE").reset_index(drop=True)
    print(f"✅ RFE : {n_features} variables sélectionnées")
    return result


def feature_importance_rf(X: pd.DataFrame, y: pd.Series,
                           n_estimators: int = 200) -> pd.DataFrame:
    """
    Importance des variables par RandomForest (Gini).

    Retourne un DataFrame trié par importance décroissante.
    """
    rf = RandomForestClassifier(n_estimators=n_estimators,
                                 random_state=42, n_jobs=-1)
    rf.fit(X, y)
    result = pd.DataFrame({
        "Variable": X.columns,
        "Importance": rf.feature_importances_,
        "Importance (%)": (rf.feature_importances_ * 100).round(2)
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    print(f"✅ Feature Importance calculée sur {n_estimators} arbres")
    return result


def combine_methods(skb_df, rfe_df, fi_df,
                    top_k: int = 10, min_votes: int = 2) -> pd.DataFrame:
    """
    Combine les 3 méthodes et retourne les variables robustes
    (sélectionnées par au moins min_votes méthodes sur 3).
    """
    top_skb = skb_df.head(top_k)["Variable"].tolist()
    top_rfe = rfe_df[rfe_df["Sélectionnée"]]["Variable"].tolist()
    top_fi  = fi_df.head(top_k)["Variable"].tolist()

    all_vars = list(set(top_skb + top_rfe + top_fi))
    comparison = pd.DataFrame({
        "Variable": all_vars,
        "SelectKBest": [v in top_skb for v in all_vars],
        "RFE": [v in top_rfe for v in all_vars],
        "FeatureImportance": [v in top_fi for v in all_vars],
    })
    comparison["Votes"] = comparison[
        ["SelectKBest", "RFE", "FeatureImportance"]].sum(axis=1)
    comparison["Retenue"] = comparison["Votes"] >= min_votes
    comparison = comparison.sort_values("Votes", ascending=False)

    selected = comparison[comparison["Retenue"]]["Variable"].tolist()
    print(f"✅ {len(selected)} variables retenues (≥{min_votes}/3 méthodes)")
    return comparison
