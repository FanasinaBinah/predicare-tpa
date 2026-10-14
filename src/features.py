"""Construction de la matrice de features (version module du notebook d'exploration).

Une seule fonction, build_features(), utilisée à l'entraînement et (plus tard) par l'API :
mêmes règles en entraînement et en production.

Sorties
- X      : facture_id + variables du modèle (sans identité, santé brute, fuite, identifiant)
- cles   : identifiants et date_soin, pour jointures et découpage temporel (jamais au modèle)
- audit  : attributs protégés (sexe, tranche d'âge...) réservés à l'audit d'équité (jamais au modèle)
- decisions : décisions historiques, réservées au tableau de bord (jamais au modèle)

Usage : python -m src.features
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.privacy import (COLONNES_QUASI_IDENTIFIANTS, COLONNES_SANTE,
                         verifier_features_ml, verifier_sans_pii)

# Colonnes issues de la décision historique : fuite de données
FUITE = ["statut", "motif_rejet", "montant_pris_en_charge", "date_traitement",
         "reste_a_charge"]
# Attributs protégés : exclus du modèle, conservés pour l'audit d'équité
PROTEGES = ["sexe", "age_soin"]
IDENTIFIANTS = ["numero_facture", "prestataire_id", "contrat_id", "ordonnance_id",
                "ord_prescripteur_id", "attestation_pec_id", "entente_prealable_id",
                "heure_soin", "jours_ouverture"]
CLES = ["facture_id", "assure_id", "prestataire_id", "contrat_id", "ordonnance_id",
        "date_soin"]


def build_features(raw_dir="data/raw/dataset"):
    raw_dir = Path(raw_dir)
    # identites_pii.csv n'est jamais lu
    t = {f.stem: pd.read_csv(f) for f in raw_dir.glob("*.csv") if f.stem != "identites_pii"}

    # --- (cellule 5 du notebook)
    decisions = t['factures'][['facture_id'] + FUITE].copy()
    decisions['date_traitement'] = pd.to_datetime(decisions['date_traitement'])
    t['factures'] = t['factures'].drop(columns=FUITE)
    assert not set(FUITE) & set(t['factures'].columns)

    # --- (cellule 8 du notebook)
    colonnes_dates = {'factures': ['date_soin', 'date_facture', 'date_reception'], 'ordonnances': ['date_prescription'], 'assures': ['date_adhesion', 'date_radiation'], 'contrats': ['date_effet_police', 'date_effet', 'date_fin'], 'prestataires': ['date_conventionnement', 'date_fin_convention']}
    for nom, cols in colonnes_dates.items():
        for c in cols:
            t[nom][c] = pd.to_datetime(t[nom][c])

    # --- (cellule 10 du notebook)
    f = t['factures']
    assures = t['assures'][['assure_id', 'assure_principal_id', 'rang_beneficiaire', 'sexe', 'annee_naissance', 'ville', 'date_adhesion', 'date_radiation']].rename(columns={'ville': 'ville_assure'})
    contrats = t['contrats'][['contrat_id', 'formule', 'date_effet_police', 'date_effet', 'date_fin', 'statut', 'nb_beneficiaires']].rename(columns={'date_effet_police': 'contrat_date_police', 'date_effet': 'contrat_date_effet', 'date_fin': 'contrat_date_fin', 'statut': 'contrat_statut'})
    presta = t['prestataires'][['prestataire_id', 'type_prestataire', 'specialite', 'ville', 'jours_ouverture', 'heure_ouverture', 'heure_fermeture', 'niveau_tarif', 'note_qualite', 'date_conventionnement', 'date_fin_convention', 'statut']].rename(columns={'specialite': 'presta_specialite', 'ville': 'ville_presta', 'statut': 'presta_statut'})
    fact_full = f.merge(assures, on='assure_id', how='left', validate='m:1').merge(contrats, on='contrat_id', how='left', validate='m:1').merge(presta, on='prestataire_id', how='left', validate='m:1')

    # --- (cellule 13 du notebook)
    actes = t['referentiel_actes'].drop(columns=['libelle']).rename(columns={'code_acte': 'code', 'categorie': 'acte_categorie', 'specialite': 'acte_specialite'})
    meds = t['referentiel_medicaments'].drop(columns=['libelle']).rename(columns={'code_medicament': 'code', 'classe': 'med_classe', 'chronique': 'med_chronique'})
    lignes = t['lignes_facture'].drop(columns=['libelle']).merge(f[['facture_id', 'prestataire_id']], on='facture_id', how='left', validate='m:1').merge(actes, on='code', how='left', validate='m:1').merge(meds, on='code', how='left', validate='m:1').merge(t['grille_tarifaire'], on=['prestataire_id', 'code'], how='left', validate='m:1')
    lignes['prix_ref'] = lignes['tarif_reference'].fillna(lignes['prix_reference'])
    for nouveau, ref in [('ratio_prix', 'prix_ref'), ('ratio_grille', 'tarif_pratique'), ('ratio_max', 'tarif_max')]:
        lignes[nouveau] = lignes['prix_unitaire'] / lignes[ref].replace(0, np.nan)

    # --- (cellule 14 du notebook)
    lignes['hors_grille'] = lignes.tarif_pratique.isna()
    lignes['depasse_max'] = lignes.ratio_max > 1
    lignes['au_dessus_grille'] = lignes.ratio_grille > 1.0001
    prix_fact = lignes.groupby('facture_id').agg(ratio_prix_max=('ratio_prix', 'max'), ratio_grille_max=('ratio_grille', 'max'), ratio_max_max=('ratio_max', 'max'), nb_lignes_hors_grille=('hors_grille', 'sum'), nb_lignes_depasse_max=('depasse_max', 'sum'), nb_lignes_au_dessus_grille=('au_dessus_grille', 'sum'), quantite_max=('quantite', 'max')).reset_index()
    fact_full = fact_full.merge(prix_fact, on='facture_id', how='left', validate='1:1')

    # --- (cellule 16 du notebook)
    ordo = t['ordonnances'][['ordonnance_id', 'assure_id', 'prescripteur_id', 'date_prescription', 'validite_jours', 'type_ordonnance', 'diagnostic']].rename(columns={'assure_id': 'ord_assure_id', 'prescripteur_id': 'ord_prescripteur_id', 'type_ordonnance': 'ord_type'})
    fact_full = fact_full.merge(ordo, on='ordonnance_id', how='left', validate='m:1')
    fact_full['delai_ord_jours'] = (fact_full.date_soin - fact_full.date_prescription).dt.days
    fact_full['ord_perimee'] = fact_full.delai_ord_jours > fact_full.validite_jours
    fact_full['ord_apres_soin'] = fact_full.delai_ord_jours < 0
    fact_full['ord_autre_assure'] = fact_full.ord_assure_id.notna() & (fact_full.assure_id != fact_full.ord_assure_id)
    fact_full['ord_sans_diagnostic'] = fact_full.ordonnance_id.notna() & fact_full.diagnostic.isna()

    # --- (cellule 17 du notebook)
    l_ord = t['lignes_facture'][['facture_id', 'code', 'quantite']].merge(f[['facture_id', 'ordonnance_id']], on='facture_id', validate='m:1').dropna(subset=['ordonnance_id']).merge(t['ordonnance_lignes'][['ordonnance_id', 'code', 'quantite']].rename(columns={'quantite': 'qte_prescrite'}), on=['ordonnance_id', 'code'], how='left', validate='m:1')
    l_ord['non_prescrite'] = l_ord.qte_prescrite.isna()
    l_ord['qte_excedentaire'] = l_ord.quantite > l_ord.qte_prescrite
    agg = l_ord.groupby('facture_id').agg(nb_lignes_non_prescrites=('non_prescrite', 'sum'), nb_lignes_qte_excedentaire=('qte_excedentaire', 'sum')).reset_index()
    fact_full = fact_full.merge(agg, on='facture_id', how='left', validate='1:1')
    cols = ['nb_lignes_non_prescrites', 'nb_lignes_qte_excedentaire']
    fact_full[cols] = fact_full[cols].fillna(0).astype(int)

    # --- (cellule 18 du notebook)
    fact_full = fact_full.drop(columns=['devise'])

    # --- (cellule 21 du notebook)
    fact_full['jours_depuis_adhesion'] = (fact_full.date_soin - fact_full.date_adhesion).dt.days
    fact_full['soin_avant_adhesion'] = fact_full.date_soin < fact_full.date_adhesion
    fact_full['soin_apres_radiation'] = fact_full.date_radiation.notna() & (fact_full.date_soin > fact_full.date_radiation)
    fact_full['soin_avant_convention'] = fact_full.date_soin < fact_full.date_conventionnement
    fact_full['presta_radie_a_la_date'] = fact_full.date_fin_convention.notna() & (fact_full.date_soin > fact_full.date_fin_convention)
    fact_full['jour_semaine'] = fact_full.date_soin.dt.dayofweek + 1
    fact_full['jour_ferme'] = [str(j) not in str(o) for j, o in zip(fact_full.jour_semaine, fact_full.jours_ouverture)]
    heure_soin = pd.to_datetime(fact_full.heure_soin, format='%H:%M')
    fact_full['heure'] = heure_soin.dt.hour + heure_soin.dt.minute / 60
    fact_full['hors_horaires'] = (fact_full.heure < fact_full.heure_ouverture) | (fact_full.heure >= fact_full.heure_fermeture)
    indicateurs = ['soin_avant_adhesion', 'soin_apres_radiation', 'soin_avant_convention', 'presta_radie_a_la_date', 'jour_ferme', 'hors_horaires']

    # --- (cellule 22 du notebook)
    fact_full['age_soin'] = fact_full.date_soin.dt.year - fact_full.annee_naissance
    profil = lignes.merge(fact_full[['facture_id', 'sexe', 'age_soin']], on='facture_id', how='left', validate='m:1')
    profil['sexe_incoherent'] = profil.sexe_requis.notna() & (profil.sexe != profil.sexe_requis)
    profil['age_incoherent'] = (profil.age_soin < profil.age_min) | (profil.age_soin > profil.age_max)
    profil_fact = profil.groupby('facture_id').agg(nb_lignes_sexe_incoherent=('sexe_incoherent', 'sum'), nb_lignes_age_incoherent=('age_incoherent', 'sum')).reset_index()
    fact_full = fact_full.merge(profil_fact, on='facture_id', how='left', validate='1:1')
    cols = ['nb_lignes_sexe_incoherent', 'nb_lignes_age_incoherent']
    fact_full[cols] = fact_full[cols].fillna(0).astype(int)

    # --- (cellule 23 du notebook)
    fact_full['nb_meme_numero'] = fact_full.groupby('numero_facture')['facture_id'].transform('size')
    cle_doublon = ['assure_id', 'prestataire_id', 'montant_total', 'date_soin']
    fact_full['nb_meme_cle'] = fact_full.groupby(cle_doublon)['facture_id'].transform('size')
    rang = fact_full.groupby(cle_doublon)['facture_id'].rank(method='first')
    fact_full['est_repetition'] = rang > 1
    fact_full['nb_meme_jour'] = fact_full.groupby(['assure_id', 'prestataire_id', 'date_soin'])['facture_id'].transform('size')
    avec_ord = fact_full.dropna(subset=['ordonnance_id'])
    fact_full['nb_fact_ord'] = fact_full.ordonnance_id.map(avec_ord.groupby('ordonnance_id').size())
    fact_full['nb_presta_ord'] = fact_full.ordonnance_id.map(avec_ord.groupby('ordonnance_id')['prestataire_id'].nunique())
    fact_full[['nb_fact_ord', 'nb_presta_ord']] = fact_full[['nb_fact_ord', 'nb_presta_ord']].fillna(0).astype(int)

    # --- (cellule 24 du notebook)
    garanties = t['garanties'][['formule', 'categorie', 'delai_carence_jours', 'attestation_requise']].rename(columns={'categorie': 'type_facture'})
    fact_full = fact_full.merge(garanties, on=['formule', 'type_facture'], how='left', validate='m:1')
    fact_full['soin_en_carence'] = (fact_full.jours_depuis_adhesion >= 0) & (fact_full.jours_depuis_adhesion < fact_full.delai_carence_jours)
    fact_full['attestation_absente'] = (fact_full.attestation_requise == 1) & fact_full.attestation_pec_id.isna()
    exclus = t['garanties'].drop_duplicates('formule').set_index('formule')['actes_exclus'].fillna('')
    ex = lignes[['facture_id', 'code']].merge(fact_full[['facture_id', 'formule']], on='facture_id', validate='m:1')
    ex['exclu'] = [c in exclus[fm].split('|') for c, fm in zip(ex.code, ex.formule)]
    ex_fact = ex.groupby('facture_id')['exclu'].sum().rename('nb_lignes_exclues').reset_index()
    ent = lignes[['facture_id', 'entente_prealable']].merge(fact_full[['facture_id', 'entente_prealable_id']], on='facture_id', validate='m:1')
    ent['manquante'] = (ent.entente_prealable == 1) & ent.entente_prealable_id.isna()
    ent_fact = ent.groupby('facture_id')['manquante'].any().rename('entente_absente').reset_index()
    fact_full = fact_full.merge(ex_fact, on='facture_id', how='left', validate='1:1').merge(ent_fact, on='facture_id', how='left', validate='1:1')
    fact_full['nb_lignes_exclues'] = fact_full.nb_lignes_exclues.fillna(0).astype(int)
    fact_full['entente_absente'] = fact_full.entente_absente.fillna(False).astype(bool)

    # --- (cellule 26) synthèse des signaux
    signaux = pd.DataFrame({'ordonnance périmée': fact_full.ord_perimee, 'soin avant prescription': fact_full.ord_apres_soin, "ordonnance d'un autre assuré": fact_full.ord_autre_assure, 'ordonnance sans diagnostic': fact_full.ord_sans_diagnostic, 'ligne non prescrite': fact_full.nb_lignes_non_prescrites > 0, 'quantité > prescrit': fact_full.nb_lignes_qte_excedentaire > 0, 'prix > tarif maximum': fact_full.nb_lignes_depasse_max > 0, 'prix > grille du prestataire': fact_full.nb_lignes_au_dessus_grille > 0, 'code hors grille': fact_full.nb_lignes_hors_grille > 0, 'acte incompatible (sexe)': fact_full.nb_lignes_sexe_incoherent > 0, 'acte incompatible (âge)': fact_full.nb_lignes_age_incoherent > 0, 'soin avant adhésion': fact_full.soin_avant_adhesion, 'soin après radiation': fact_full.soin_apres_radiation, 'soin pendant la carence': fact_full.soin_en_carence, 'prestataire hors convention': fact_full.presta_radie_a_la_date | fact_full.soin_avant_convention, 'jour fermé / hors horaires': fact_full.jour_ferme | fact_full.hors_horaires, 'doublon (répétition)': fact_full.est_repetition, 'ordonnance multi-prestataires': fact_full.nb_presta_ord > 1, 'attestation absente': fact_full.attestation_absente, 'entente préalable absente': fact_full.entente_absente, 'acte exclu de la formule': fact_full.nb_lignes_exclues > 0})
    fact_full['nb_signaux'] = signaux.sum(axis=1)

    # --- CORRECTION 1 : doublon = la COPIE (réception la plus tardive), pas "le plus petit id"
    # Deux clés : même numéro de facture, ou même (assuré, prestataire, montant, date de soin)
    fact_full = fact_full.sort_values(["date_reception", "facture_id"])
    rang_num = fact_full.groupby("numero_facture").cumcount()
    rang_cle = fact_full.groupby(["assure_id", "prestataire_id", "montant_total",
                                  "date_soin"]).cumcount()
    fact_full["est_copie"] = (rang_num > 0) | (rang_cle > 0)
    fact_full = fact_full.sort_values("facture_id").reset_index(drop=True)

    # --- Indicateurs de présence à la place des identifiants de documents
    fact_full["a_ordonnance"] = fact_full.ordonnance_id.notna()
    fact_full["a_attestation"] = fact_full.attestation_pec_id.notna()
    fact_full["a_entente"] = fact_full.entente_prealable_id.notna()
    fact_full["mois_soin"] = fact_full.date_soin.dt.month

    cles = fact_full[CLES].copy()
    audit = fact_full[["facture_id", "sexe", "age_soin", "formule", "type_facture",
                       "ville_presta"]].copy()
    audit["age_tranche"] = pd.cut(audit.age_soin, bins=[-1, 5, 17, 35, 50, 65, 120],
                                  labels=["0-5", "6-17", "18-35", "36-50", "51-65", "65+"])

    dates_brutes = list(fact_full.select_dtypes(include="datetime").columns)
    a_retirer = (COLONNES_SANTE + COLONNES_QUASI_IDENTIFIANTS + FUITE + IDENTIFIANTS
                 + PROTEGES + dates_brutes)
    X = fact_full.drop(columns=[c for c in a_retirer if c in fact_full.columns])
    X = X.replace([np.inf, -np.inf], np.nan)
    for c in X.select_dtypes(include=["object", "string"]).columns:
        X[c] = X[c].astype("category")

    # Contrôles automatiques
    verifier_sans_pii(X)
    verifier_features_ml(X)
    assert not set(FUITE) & set(X.columns), "Colonne de fuite présente dans X"
    assert not set(PROTEGES) & set(X.columns), "Attribut protégé présent dans X"
    assert X.facture_id.is_unique, "facture_id doit être unique"
    return X, cles, audit, decisions


if __name__ == "__main__":
    sortie = Path("data/processed")
    sortie.mkdir(parents=True, exist_ok=True)
    X, cles, audit, decisions = build_features()
    X.to_parquet(sortie / "features.parquet", index=False)
    cles.to_parquet(sortie / "cles.parquet", index=False)
    audit.to_parquet(sortie / "audit.parquet", index=False)
    decisions.to_parquet(sortie / "decisions.parquet", index=False)
    print("OK :", X.shape, "variables ->", sortie.resolve())
