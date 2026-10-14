"""Protection des données personnelles : chiffrement du fichier d'identités,
pseudonymisation des identifiants et contrôle des colonnes sensibles.

Les clés ne sont jamais écrites dans le code : elles sont lues dans le fichier
.env (variables PII_KEY et PSEUDO_KEY), qui n'est pas versionné.
"""
import hashlib
import hmac
import io
import os
import sys
from pathlib import Path

import pandas as pd
from cryptography.fernet import Fernet, InvalidToken
from dotenv import load_dotenv

load_dotenv()

COLONNES_IDENTIFIANTES = ["nom", "prenoms", "date_naissance", "telephone",
                          "email", "adresse", "numero_cin"]
# Données de santé (art. 9 RGPD) : interdites en clair dans les features ML
COLONNES_SANTE = ["diagnostic", "renseignement_clinique"]
# Quasi-identifiants et identifiants : utiles comme clés de jointure, pas comme features
COLONNES_QUASI_IDENTIFIANTS = ["annee_naissance", "ville_assure", "assure_id",
                               "assure_principal_id", "ord_assure_id"]


def _fernet(cle=None):
    cle = cle or os.getenv("PII_KEY")
    if not cle:
        raise RuntimeError("PII_KEY absente : ajoutez-la dans le fichier .env")
    if cle == os.getenv("PSEUDO_KEY"):
        raise RuntimeError("PII_KEY et PSEUDO_KEY doivent être différentes")
    try:
        return Fernet(cle.encode())
    except (ValueError, TypeError) as erreur:
        raise RuntimeError("PII_KEY invalide : générez-la avec "
                           "Fernet.generate_key()") from erreur


def chiffrer_fichier(source, destination, cle=None):
    """Chiffre un fichier (ex. identites_pii.csv) et écrit la version chiffrée.
    Le fichier source n'est pas supprimé : à vous de le garder hors du dépôt."""
    donnees = Path(source).read_bytes()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_fernet(cle).encrypt(donnees))


def lire_identites_chiffrees(chemin, cle=None):
    """Déchiffre en mémoire et renvoie un DataFrame (rien n'est écrit en clair)."""
    try:
        clair = _fernet(cle).decrypt(Path(chemin).read_bytes())
    except InvalidToken as erreur:
        raise ValueError("Clé incorrecte ou fichier corrompu") from erreur
    return pd.read_csv(io.BytesIO(clair))


def pseudonymiser(valeur, cle=None, prefixe="ASS-", longueur=10):
    """Pseudonyme stable : même valeur + même clé = même pseudonyme.
    HMAC-SHA256 : sans la clé, impossible de refaire le calcul."""
    secret = cle or os.getenv("PSEUDO_KEY")
    if not secret:
        raise RuntimeError("PSEUDO_KEY absente : ajoutez-la dans le fichier .env")
    if valeur is None or pd.isna(valeur) or str(valeur).strip() == "":
        raise ValueError("Valeur vide : impossible de pseudonymiser")
    h = hmac.new(secret.encode(), str(valeur).strip().encode(), hashlib.sha256)
    return prefixe + h.hexdigest()[:longueur].upper()


def _presentes(df, interdites):
    """Colonnes de df appartenant à la liste, sans tenir compte casse/espaces."""
    interdites = {c.lower() for c in interdites}
    return [c for c in df.columns if str(c).strip().lower() in interdites]


def verifier_sans_pii(df):
    """Lève une erreur si le tableau contient une colonne d'identité."""
    presentes = _presentes(df, COLONNES_IDENTIFIANTES)
    if presentes:
        raise ValueError(f"Colonnes d'identité interdites : {presentes}")


def verifier_features_ml(df):
    """Contrôle la matrice d'entraînement : ni identité, ni donnée de santé
    brute, ni identifiant ou quasi-identifiant direct."""
    presentes = _presentes(df, COLONNES_IDENTIFIANTES + COLONNES_SANTE
                           + COLONNES_QUASI_IDENTIFIANTS)
    if presentes:
        raise ValueError(f"Colonnes interdites dans les features : {presentes}")


if __name__ == "__main__":
    # python -m src.privacy chiffrer <source> <destination>
    # python -m src.privacy verifier <fichier.enc>
    commande = sys.argv[1] if len(sys.argv) > 1 else ""
    if commande == "chiffrer" and len(sys.argv) == 4:
        chiffrer_fichier(sys.argv[2], sys.argv[3])
        print("Fichier chiffré :", sys.argv[3])
    elif commande == "verifier" and len(sys.argv) == 3:
        df = lire_identites_chiffrees(sys.argv[2])
        print("Déchiffrement OK :", df.shape[0], "lignes,", df.shape[1], "colonnes")
    else:
        print("Usage : chiffrer <source> <destination> | verifier <fichier.enc>")