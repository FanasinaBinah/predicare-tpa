"""Fonction de prédiction unique, appelée par l'API (Membre 2).

Le même code de préparation sert à l'entraînement et à la production :
catégories figées, booléens convertis, colonnes dans le même ordre.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
SEUIL_ALERTE_DEFAUT = 0.8  # seuil Slack/n8n du cahier des charges, modifiable via .env
# Règles métier dures : un seul de ces signaux suffit à envoyer la facture au contrôle,
# même si le modèle la juge peu risquée (le modèle les rate parfois).
REGLES_DURES = ["presta_radie_a_la_date", "soin_avant_convention", "soin_apres_radiation",
                "soin_avant_adhesion", "soin_en_carence", "ord_perimee"]


def preparer(X, categories, colonnes):
    """Aligne X sur ce que le modèle a vu à l'entraînement."""
    X = X.copy()
    for c in colonnes:
        if c not in X.columns:
            X[c] = np.nan
    X = X[colonnes]
    for c in X.columns:
        if c in categories:
            connu = X[c].isin(categories[c])
            X[c] = pd.Categorical(X[c].where(connu), categories=categories[c])
        elif X[c].dtype == bool:
            X[c] = X[c].astype("int8")
    return X.replace([np.inf, -np.inf], np.nan)


@lru_cache(maxsize=4)
def charger_modele(version="v1", models_dir=None):
    dossier = Path(models_dir or MODELS_DIR) / version
    meta = json.loads((dossier / "metadata.json").read_text(encoding="utf-8"))
    return {"model": joblib.load(dossier / "model.joblib"), "meta": meta}


def predire(df_features, version="v1", seuil_alerte=None, models_dir=None):
    """df_features : colonnes de features (facture_id facultatif).
    Retourne facture_id, score, regle_dure, a_controler (modèle au seuil choisi pour le
    rappel OU règle dure) et alerte (score >= seuil Slack, configurable)."""
    bundle = charger_modele(version, models_dir)
    meta, model = bundle["meta"], bundle["model"]
    if seuil_alerte is None:
        seuil_alerte = float(os.getenv("ALERT_THRESHOLD", SEUIL_ALERTE_DEFAUT))
    Xp = preparer(df_features, meta["categories"], meta["colonnes"])
    score = model.predict_proba(Xp)[:, 1]
    ids = df_features["facture_id"].values if "facture_id" in df_features else np.arange(len(Xp))
    dures = [c for c in REGLES_DURES if c in df_features.columns]
    regle_dure = (df_features[dures].fillna(False).astype(bool).any(axis=1).values
                  if dures else np.zeros(len(Xp), dtype=bool))
    return pd.DataFrame({"facture_id": ids, "score": score, "regle_dure": regle_dure,
                         "a_controler": (score >= meta["seuil"]) | regle_dure,
                         "alerte": score >= seuil_alerte,
                         "version_modele": version})
