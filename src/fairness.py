"""Audit d'équité par sous-groupe (sexe, tranche d'âge, formule, type de facture...)."""
import numpy as np
import pandas as pd


def audit_equite(audit, y, pred, groupes=("sexe", "age_tranche", "formule", "type_facture"),
                 min_n=50):
    """Une ligne par (attribut, modalité) : taux de fraude réel, taux d'alerte,
    rappel, taux de faux positifs. `audit` et y/pred doivent avoir le même ordre."""
    df = audit.reset_index(drop=True).copy()
    df["y"], df["pred"] = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    lignes = []
    for g in groupes:
        for modalite, sous in df.groupby(g, observed=True):
            if len(sous) < min_n:
                continue
            pos, neg = sous[sous.y], sous[~sous.y]
            lignes.append({"attribut": g, "modalite": str(modalite), "n": len(sous),
                           "taux_fraude_reel": sous.y.mean(),
                           "taux_alerte": sous.pred.mean(),
                           "rappel": pos.pred.mean() if len(pos) else np.nan,
                           "taux_faux_positifs": neg.pred.mean() if len(neg) else np.nan})
    return pd.DataFrame(lignes).round(4)


def ecarts(rapport, colonne="taux_faux_positifs"):
    """Rapport min/max par attribut (1 = parfaitement équilibré ; < 0,8 = à examiner)."""
    return (rapport.groupby("attribut")[colonne]
            .agg(lambda s: s.min() / s.max() if s.max() > 0 else np.nan).round(3))
