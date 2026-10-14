"""Explicabilité SHAP : rapport par alerte (pour le gestionnaire) et rapport global.
Nécessite xgboost + shap."""
import numpy as np

from src.inference import charger_modele, preparer


def expliquer(df_features, version="v1", top_k=5, models_dir=None):
    """Pour chaque facture : les top_k variables qui poussent le score vers la fraude."""
    import shap
    b = charger_modele(version, models_dir)
    meta = b["meta"]
    Xp = preparer(df_features, meta["categories"], meta["colonnes"])
    vals = shap.TreeExplainer(b["model"]).shap_values(Xp)
    if isinstance(vals, list):  # certaines versions renvoient une liste par classe
        vals = vals[1]
    rapports = []
    for i in range(len(Xp)):
        ordre = np.argsort(-np.abs(vals[i]))[:top_k]
        rapports.append([{"variable": Xp.columns[j], "valeur": str(Xp.iloc[i, j]),
                          "contribution": round(float(vals[i, j]), 4),
                          "sens": "augmente le risque" if vals[i, j] > 0 else "diminue le risque"}
                         for j in ordre])
    return rapports


def rapport_global(df_features, version="v1", chemin="reports/shap_global.png",
                   n=5000, models_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    b = charger_modele(version, models_dir)
    meta = b["meta"]
    Xp = preparer(df_features.sample(min(n, len(df_features)), random_state=42),
                  meta["categories"], meta["colonnes"])
    vals = shap.TreeExplainer(b["model"]).shap_values(Xp)
    shap.summary_plot(vals[1] if isinstance(vals, list) else vals, Xp, show=False,
                      max_display=20)
    plt.tight_layout()
    plt.savefig(chemin, dpi=120)
    plt.close()
    return chemin
