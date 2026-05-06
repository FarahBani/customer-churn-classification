"""
models.py
=========
Fonctions d'entraînement, évaluation et comparaison des modèles
de classification pour le projet Customer Churn.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report)
from sklearn.model_selection import cross_val_score, StratifiedKFold


def get_models() -> dict:
    """Retourne les 4 modèles du projet avec leurs hyperparamètres par défaut."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree"      : DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest"      : RandomForestClassifier(n_estimators=100,
                                                       random_state=42, n_jobs=-1),
        "AdaBoost"           : AdaBoostClassifier(n_estimators=100, random_state=42)
    }


def evaluate_model(model, X_test, y_test) -> dict:
    """Calcule toutes les métriques d'évaluation pour un modèle."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy" : round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall"   : round(recall_score(y_test, y_pred), 4),
        "F1-Score" : round(f1_score(y_test, y_pred), 4),
        "ROC-AUC"  : round(roc_auc_score(y_test, y_proba), 4),
        "y_pred"   : y_pred,
        "y_proba"  : y_proba
    }


def train_and_evaluate(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """
    Entraîne et évalue les 4 modèles.

    Retourne un DataFrame avec les métriques de chaque modèle.
    """
    models = get_models()
    results = []
    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        trained[name] = {"model": model, **metrics}
        results.append({"Modèle": name,
                         "Accuracy" : metrics["Accuracy"],
                         "Precision": metrics["Precision"],
                         "Recall"   : metrics["Recall"],
                         "F1-Score" : metrics["F1-Score"],
                         "ROC-AUC"  : metrics["ROC-AUC"]})
        print(f"✅ {name:25s} F1={metrics['F1-Score']} | AUC={metrics['ROC-AUC']}")

    return pd.DataFrame(results), trained


def cross_validate_models(trained: dict, X_train, y_train,
                           n_splits: int = 5) -> pd.DataFrame:
    """
    Validation croisée stratifiée sur tous les modèles.

    Retourne un DataFrame avec F1 et AUC moyens ± écart-type.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_results = []

    for name, data in trained.items():
        model = data["model"]
        f1s  = cross_val_score(model, X_train, y_train,
                                cv=cv, scoring="f1", n_jobs=-1)
        aucs = cross_val_score(model, X_train, y_train,
                                cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_results.append({
            "Modèle"   : name,
            "F1 moyen" : round(f1s.mean(), 4),
            "F1 std"   : round(f1s.std(), 4),
            "AUC moyen": round(aucs.mean(), 4),
            "AUC std"  : round(aucs.std(), 4)
        })
        print(f"{name:25s} F1={f1s.mean():.3f}±{f1s.std():.3f} | AUC={aucs.mean():.3f}±{aucs.std():.3f}")

    return pd.DataFrame(cv_results)


def get_best_model(results_df: pd.DataFrame,
                   metric: str = "F1-Score") -> str:
    """Retourne le nom du meilleur modèle selon la métrique choisie."""
    best = results_df.loc[results_df[metric].idxmax(), "Modèle"]
    val  = results_df[metric].max()
    print(f"🏆 Meilleur modèle ({metric}) : {best} ({val})")
    return best
