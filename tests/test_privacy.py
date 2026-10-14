import re

import pandas as pd
import pytest
from cryptography.fernet import Fernet

from src.privacy import (chiffrer_fichier, lire_identites_chiffrees,
                         pseudonymiser, verifier_features_ml,
                         verifier_sans_pii)


@pytest.fixture
def csv_identites(tmp_path):
    chemin = tmp_path / "id.csv"
    chemin.write_text("assure_id,nom\nASS-1,DUPONT\n", encoding="utf-8")
    return chemin


# --------------------------------------------------------------- chiffrement
def test_aller_retour_chiffrement(tmp_path, csv_identites):
    cle = Fernet.generate_key().decode()
    chiffre = tmp_path / "id.csv.enc"
    chiffrer_fichier(csv_identites, chiffre, cle)
    assert b"DUPONT" not in chiffre.read_bytes()
    assert lire_identites_chiffrees(chiffre, cle).loc[0, "nom"] == "DUPONT"


def test_mauvaise_cle_refusee(tmp_path, csv_identites):
    chiffre = tmp_path / "id.csv.enc"
    chiffrer_fichier(csv_identites, chiffre, Fernet.generate_key().decode())
    with pytest.raises(ValueError):
        lire_identites_chiffrees(chiffre, Fernet.generate_key().decode())


def test_fichier_corrompu_refuse(tmp_path):
    chiffre = tmp_path / "id.csv.enc"
    chiffre.write_bytes(b"pas du tout un fichier chiffre")
    with pytest.raises(ValueError):
        lire_identites_chiffrees(chiffre, Fernet.generate_key().decode())


def test_dossier_destination_cree(tmp_path, csv_identites):
    chiffre = tmp_path / "nouveau" / "sous_dossier" / "id.csv.enc"
    chiffrer_fichier(csv_identites, chiffre, Fernet.generate_key().decode())
    assert chiffre.exists()


def test_fichier_source_conserve(tmp_path, csv_identites):
    chiffrer_fichier(csv_identites, tmp_path / "id.csv.enc",
                     Fernet.generate_key().decode())
    assert csv_identites.exists()


# ------------------------------------------------------------ gestion des clés
def test_pii_key_absente(monkeypatch, tmp_path, csv_identites):
    monkeypatch.delenv("PII_KEY", raising=False)
    with pytest.raises(RuntimeError, match="PII_KEY"):
        chiffrer_fichier(csv_identites, tmp_path / "x.enc")


def test_pii_key_invalide(tmp_path, csv_identites):
    with pytest.raises(RuntimeError, match="invalide"):
        chiffrer_fichier(csv_identites, tmp_path / "x.enc", "pas-une-cle-fernet")


def test_pii_key_identique_a_pseudo_key_refusee(monkeypatch, tmp_path, csv_identites):
    cle = Fernet.generate_key().decode()
    monkeypatch.setenv("PSEUDO_KEY", cle)
    with pytest.raises(RuntimeError, match="différentes"):
        chiffrer_fichier(csv_identites, tmp_path / "x.enc", cle)


def test_pseudo_key_absente(monkeypatch):
    monkeypatch.delenv("PSEUDO_KEY", raising=False)
    with pytest.raises(RuntimeError, match="PSEUDO_KEY"):
        pseudonymiser("Jean")


# ----------------------------------------------------------- pseudonymisation
def test_pseudonyme_stable_et_dependant_de_la_cle():
    assert pseudonymiser("Jean", "cle1") == pseudonymiser("Jean", "cle1")
    assert pseudonymiser("Jean", "cle1") != pseudonymiser("Jean", "cle2")
    assert pseudonymiser("Jean", "cle1") != pseudonymiser("Paul", "cle1")


def test_pseudonyme_format():
    assert re.fullmatch(r"ASS-[0-9A-F]{10}", pseudonymiser("Jean", "cle1"))
    assert re.fullmatch(r"ASS-[0-9A-F]{16}", pseudonymiser("Jean", "cle1", longueur=16))


def test_pseudonyme_ignore_espaces_autour():
    assert pseudonymiser(" Jean ", "cle1") == pseudonymiser("Jean", "cle1")


@pytest.mark.parametrize("vide", [None, float("nan"), "", "   "])
def test_pseudonyme_refuse_valeur_vide(vide):
    with pytest.raises(ValueError):
        pseudonymiser(vide, "cle1")


# ------------------------------------------------------------ garde-fous colonnes
def test_verifier_sans_pii():
    verifier_sans_pii(pd.DataFrame({"assure_id": ["ASS-1"], "sexe": ["F"]}))
    with pytest.raises(ValueError):
        verifier_sans_pii(pd.DataFrame({"assure_id": ["ASS-1"], "nom": ["X"]}))


def test_verifier_sans_pii_date_naissance():
    with pytest.raises(ValueError, match="date_naissance"):
        verifier_sans_pii(pd.DataFrame({"date_naissance": ["1990-01-01"]}))


@pytest.mark.parametrize("colonne", [" Nom ", "EMAIL", "Numero_CIN"])
def test_verifier_sans_pii_casse_et_espaces(colonne):
    with pytest.raises(ValueError):
        verifier_sans_pii(pd.DataFrame({colonne: ["X"]}))


@pytest.mark.parametrize("colonne", ["diagnostic", "renseignement_clinique",
                                     "annee_naissance", "assure_id", "ville_assure"])
def test_features_ml_refuse_colonnes_sensibles(colonne):
    with pytest.raises(ValueError, match=colonne):
        verifier_features_ml(pd.DataFrame({colonne: ["x"]}))


def test_features_ml_refuse_aussi_les_identites():
    with pytest.raises(ValueError):
        verifier_features_ml(pd.DataFrame({"telephone": ["0340000000"]}))


def test_features_ml_accepte_matrice_propre():
    verifier_features_ml(pd.DataFrame({"age_tranche": ["18-35"], "nb_signaux": [2]}))