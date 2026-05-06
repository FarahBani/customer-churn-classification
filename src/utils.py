"""
utils.py
========
Fonctions utilitaires pour la visualisation et la gestion des fichiers.
"""

import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


def save_json(data: dict, path: str):
    """Sauvegarde un dictionnaire en JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Sauvegardé : {path}")


def load_json(path: str) -> dict:
    """Charge un fichier JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_model(model, path: str):
    """Sauvegarde un modèle sklearn en pickle."""
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Modèle sauvegardé : {path}")


def load_model(path: str):
    """Charge un modèle sklearn depuis pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_confusion_matrix(y_true, y_pred, model_name: str,
                           color: str = "#4C9BE8",
                           save_path: str = None):
    """Affiche une matrice de confusion annotée."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                cmap=sns.light_palette(color, as_cmap=True),
                linewidths=0.5, linecolor="white",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    tn, fp, fn, tp = cm.ravel()
    ax.set_title(f"{model_name}\nRecall={tp/(tp+fn):.2f} | Precision={tp/(tp+fp):.2f}")
    ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_roc_curves(trained: dict, y_test,
                    colors: list = None, save_path: str = None):
    """Trace les courbes ROC de tous les modèles."""
    if colors is None:
        colors = ["#4C9BE8","#E8734C","#4CAF7D","#9B59B6"]
    fig, ax = plt.subplots(figsize=(8, 7))
    for (name, data), color in zip(trained.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, data["y_proba"])
        auc = roc_auc_score(y_test, data["y_proba"])
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f"{name} (AUC={auc:.3f})")
    ax.plot([0,1],[0,1],"k--", lw=1, label="Aléatoire")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("Courbes ROC", fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
