import numpy as np
import pandas as pd

from src.fairness import audit_equite, ecarts
from src.inference import predire, preparer
from src.train import (choisir_seuil, charger_jeu, entrainer,  # noqa: F401
                       split_temporel)


def _jeu(n=3000, seed=0):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({"facture_id": [f"F{i}" for i in range(n)],
                      "nb_signaux": r.integers(0, 4, n),
                      "montant": r.gamma(2, 1e4, n),
                      "a_ordonnance": r.random(n) > 0.5,
                      "type_facture": pd.Categorical(r.choice(["A", "B", "C"], n)),
                      "date_soin": pd.date_range("2025-01-01", periods=n, freq="4h")})
    p = 1 / (1 + np.exp(-(d.nb_signaux * 1.5 - 4)))
    d["est_fraude"] = (r.random(n) < p).astype(int)
    d["type_fraude"] = np.where(d.est_fraude == 1, "X", None)
    return d


def test_split_temporel_sans_chevauchement():
    d = _jeu()
    tr, va, te = split_temporel(d, "2025-06-01", "2025-09-01")
    assert tr.date_soin.max() < va.date_soin.min() <= va.date_soin.max() < te.date_soin.min()
    assert len(tr) + len(va) + len(te) == len(d)


def test_choisir_seuil_respecte_le_rappel():
    r = np.random.default_rng(1)
    y = (r.random(2000) < 0.1).astype(int)
    p = np.clip(y * 0.5 + r.random(2000) * 0.5, 0, 1)
    s = choisir_seuil(y, p, 0.95)
    assert ((p >= s) & (y == 1)).sum() / y.sum() >= 0.95


def test_preparer_categorie_inconnue_devient_nan_et_ordre_colonnes():
    X = pd.DataFrame({"type_facture": ["A", "Z"], "a_ordonnance": [True, False]})
    out = preparer(X, {"type_facture": ["A", "B"]}, ["a_ordonnance", "type_facture", "absente"])
    assert list(out.columns) == ["a_ordonnance", "type_facture", "absente"]
    assert pd.isna(out.type_facture.iloc[1]) and out.a_ordonnance.dtype == "int8"


def test_entrainement_puis_inference(tmp_path):
    d = _jeu()
    _, meta, _ = entrainer(d.assign(date_soin=d.date_soin), "hgb", 0.9, "vtest", tmp_path)
    assert meta["resultats"]["test"]["rappel"] >= 0.5
    out = predire(d.drop(columns=["est_fraude", "type_fraude", "date_soin"]).head(10),
                  "vtest", seuil_alerte=0.8, models_dir=tmp_path)
    assert len(out) == 10 and out.score.between(0, 1).all()
    assert {"score", "a_controler", "alerte", "version_modele"} <= set(out.columns)


def test_audit_equite():
    audit = pd.DataFrame({"sexe": ["F"] * 100 + ["M"] * 100})
    y = np.r_[np.ones(10), np.zeros(90), np.ones(10), np.zeros(90)]
    pred = np.r_[np.ones(10), np.zeros(90), np.ones(10), np.ones(10), np.zeros(80)]
    rap = audit_equite(audit, y, pred, groupes=["sexe"], min_n=10)
    assert set(rap.modalite) == {"F", "M"} and (rap.rappel == 1).all()
    assert 0 <= ecarts(rap).iloc[0] <= 1
