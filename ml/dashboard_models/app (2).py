# -*- coding: utf-8 -*-
"""
PrediCare TPA — Dashboard Modèles / Scores / FinOps
3 versions (v1, v2, v3) · Ariary · Streamlit
Emplacement : ml/dashboard_models/app.py
Lancer : streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Config page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PrediCare — Modèles & FinOps",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Chemins modèles (ml/models/v1|v2|v3)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CANDIDATES = [
    ROOT.parent / "models",           # ml/models
    ROOT.parent.parent / "models",    # projet/models
    ROOT / "models",
]

def find_models_dir() -> Path | None:
    for p in CANDIDATES:
        if p.is_dir():
            return p
    return None


MODELS_DIR = find_models_dir()

# ---------------------------------------------------------------------------
# Métriques projet (fallback si pas de metadata.json)
# ---------------------------------------------------------------------------
DEFAULTS = {
    "v1": {
        "label": "v1 — Prudent",
        "seuil": 0.7156,
        "recall": 0.9247,
        "precision": 0.9290,
        "f1": 0.9268,
        "pr_auc": 0.9721,
        "roc_auc": 0.9927,
        "pct_alertes": 8.5,
        "n_test": 12689,
        "n_fraudes_test": 1089,
        "rappel_cible": 0.92,
        "color": "#2B6CB0",
    },
    "v2": {
        "label": "v2 — Recommandé",
        "seuil": 0.5429,
        "recall": 0.9467,
        "precision": 0.8737,
        "f1": 0.9088,
        "pr_auc": 0.9721,
        "roc_auc": 0.9927,
        "pct_alertes": 9.3,
        "n_test": 12689,
        "n_fraudes_test": 1089,
        "rappel_cible": 0.95,
        "color": "#38A169",
    },
    "v3": {
        "label": "v3 — Maximal",
        "seuil": 0.1457,
        "recall": 0.9798,
        "precision": 0.5340,
        "f1": 0.6913,
        "pr_auc": 0.9721,
        "roc_auc": 0.9927,
        "pct_alertes": 15.7,
        "n_test": 12689,
        "n_fraudes_test": 1089,
        "rappel_cible": 0.98,
        "color": "#C05621",
    },
}

# Rappel par type de fraude (v2 — ordre de grandeur projet)
RECALL_PAR_TYPE = {
    "SURFACTURATION": 1.00,
    "LIGNES_NON_PRESCRITES": 1.00,
    "ORDONNANCE_EXPIREE": 1.00,
    "QUANTITE_EXCESSIVE": 1.00,
    "PRESTATAIRE_RADIE": 1.00,
    "SUBSTITUTION_PRODUIT": 1.00,
    "SOINS_JOUR_FERME": 1.00,
    "HORS_COUVERTURE": 1.00,
    "SURCONSOMMATION": 0.975,
    "FRACTIONNEMENT": 0.936,
    "ACTE_FICTIF": 0.967,
    "UPCODING": 0.865,
    "DOUBLON": 0.804,
    "INCOHERENCE_PROFIL": 0.75,
}

# ---------------------------------------------------------------------------
# Hypothèses FinOps (Ariary)
# ---------------------------------------------------------------------------
COUT_FP_AR = 2500
MONTANT_MOYEN_FRAUDE_AR = 150_000
INFRA_MENSUEL_AR = 1_750_000
INFRA_24_MOIS_AR = INFRA_MENSUEL_AR * 24
N_FACTURES_PROJET = 84_593
TAUX_FRAUDE = 0.0869


def load_version_meta(version: str) -> dict:
    """Charge metadata.json si présent, sinon DEFAULTS."""
    base = DEFAULTS[version].copy()
    if MODELS_DIR is None:
        return base
    meta_path = MODELS_DIR / version / "metadata.json"
    if not meta_path.exists():
        return base
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if "seuil_alerte" in meta:
            base["seuil"] = float(meta["seuil_alerte"])
        if "rappel_cible" in meta:
            base["rappel_cible"] = float(meta["rappel_cible"])
        rt = meta.get("resultats_test") or {}
        for k_src, k_dst in [
            ("recall", "recall"),
            ("precision", "precision"),
            ("f1", "f1"),
            ("pr_auc", "pr_auc"),
            ("roc_auc", "roc_auc"),
            ("pct_alertes", "pct_alertes"),
        ]:
            if k_src in rt:
                base[k_dst] = float(rt[k_src])
        if "n_test" in rt:
            base["n_test"] = int(rt["n_test"])
    except Exception:
        pass
    return base


def finops_for(v: dict) -> dict:
    """Calcule montant économisé, coût FP, gain net, ROI (extrapolation projet)."""
    n_fraudes = N_FACTURES_PROJET * TAUX_FRAUDE
    n_vp = n_fraudes * v["recall"]
    n_alertes = N_FACTURES_PROJET * (v["pct_alertes"] / 100.0)
    # FP ≈ alertes - VP (approx)
    n_fp = max(0.0, n_alertes - n_vp)
    montant_economise = n_vp * MONTANT_MOYEN_FRAUDE_AR
    cout_fp = n_fp * COUT_FP_AR
    gain_net = montant_economise - cout_fp - INFRA_24_MOIS_AR
    roi = (gain_net / INFRA_24_MOIS_AR) * 100 if INFRA_24_MOIS_AR else 0
    return {
        "n_vp": n_vp,
        "n_fp": n_fp,
        "n_alertes": n_alertes,
        "montant_economise": montant_economise,
        "cout_fp": cout_fp,
        "infra": INFRA_24_MOIS_AR,
        "gain_net": gain_net,
        "roi": roi,
    }


def fmt_ar(x: float) -> str:
    if abs(x) >= 1_000_000_000:
        return f"{x / 1_000_000_000:.2f} Md Ar"
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.1f} M Ar"
    return f"{x:,.0f} Ar".replace(",", " ")


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f} %" if x <= 1 else f"{x:.1f} %"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("PrediCare TPA")
st.sidebar.caption("Modèles · Scores · FinOps")

version = st.sidebar.selectbox(
    "Version du modèle",
    options=["v1", "v2", "v3"],
    format_func=lambda x: DEFAULTS[x]["label"],
    index=1,  # v2 par défaut
)

v = load_version_meta(version)
fo = finops_for(v)

st.sidebar.markdown("---")
st.sidebar.markdown("**Statut modèles**")
if MODELS_DIR:
    st.sidebar.success(f"Dossier : `{MODELS_DIR}`")
    for ver in ("v1", "v2", "v3"):
        joblib = MODELS_DIR / ver / "model.joblib"
        meta = MODELS_DIR / ver / "metadata.json"
        ok = joblib.exists()
        st.sidebar.write(
            f"{'✅' if ok else '⚠️'} **{ver}** "
            f"{'model.joblib' if ok else 'pas de joblib'} · "
            f"{'metadata OK' if meta.exists() else 'metadata défaut'}"
        )
else:
    st.sidebar.warning("Aucun dossier models/ trouvé — métriques projet utilisées")

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Hypothèses FinOps : FP = {COUT_FP_AR} Ar · "
    f"fraude moy. = {MONTANT_MOYEN_FRAUDE_AR:,} Ar · "
    f"infra = {INFRA_MENSUEL_AR:,} Ar/mois"
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Dashboard Modèles & FinOps")
st.markdown(
    f"**Version active :** {v['label']}  ·  "
    f"**Seuil :** `{v['seuil']:.4f}`  ·  "
    f"**Rappel cible :** {fmt_pct(v['rappel_cible'])}"
)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Recall", fmt_pct(v["recall"]))
c2.metric("Précision", fmt_pct(v["precision"]))
c3.metric("F1", fmt_pct(v["f1"]))
c4.metric("% alertes", f"{v['pct_alertes']:.1f} %")
c5.metric("Montant économisé", fmt_ar(fo["montant_economise"]))

st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_modele, tab_alertes, tab_finops, tab_cmp = st.tabs(
    ["Modèle & scores", "Alertes", "FinOps", "Comparaison v1 / v2 / v3"]
)

# ===== TAB MODELE =====
with tab_modele:
    st.subheader("Performance sur le jeu de test")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"""
| Indicateur | Valeur |
|------------|--------|
| Seuil d'alerte | **{v['seuil']:.4f}** |
| Recall | **{fmt_pct(v['recall'])}** |
| Précision | **{fmt_pct(v['precision'])}** |
| F1-score | **{fmt_pct(v['f1'])}** |
| PR-AUC | **{v['pr_auc']:.4f}** |
| ROC-AUC | **{v['roc_auc']:.4f}** |
| Factures test | {v['n_test']:,} |
| Fraudes test | {v['n_fraudes_test']:,} |
"""
        )
    with col_b:
        fig_g = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=v["seuil"],
                title={"text": "Seuil d'alerte"},
                gauge={
                    "axis": {"range": [0, 1]},
                    "bar": {"color": v["color"]},
                    "steps": [
                        {"range": [0, 0.3], "color": "#FED7D7"},
                        {"range": [0.3, 0.6], "color": "#FEEBC8"},
                        {"range": [0.6, 1], "color": "#C6F6D5"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 2},
                        "thickness": 0.75,
                        "value": v["seuil"],
                    },
                },
                number={"valueformat": ".3f"},
            )
        )
        fig_g.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_g, use_container_width=True)

    st.subheader("Rappel par type de fraude (réf. v2)")
    df_types = pd.DataFrame(
        {"Type": list(RECALL_PAR_TYPE.keys()), "Recall": list(RECALL_PAR_TYPE.values())}
    ).sort_values("Recall", ascending=True)
    fig_t = px.bar(
        df_types,
        x="Recall",
        y="Type",
        orientation="h",
        color="Recall",
        color_continuous_scale="Teal",
        range_x=[0, 1.05],
        text=df_types["Recall"].apply(lambda x: f"{x*100:.0f}%"),
    )
    fig_t.update_traces(textposition="outside")
    fig_t.update_layout(height=420, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_t, use_container_width=True)

# ===== TAB ALERTES =====
with tab_alertes:
    st.subheader("Volume d'alertes (extrapolation projet)")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Alertes estimées", f"{fo['n_alertes']:,.0f}".replace(",", " "))
    a2.metric("% factures alertées", f"{v['pct_alertes']:.1f} %")
    a3.metric("Vrais positifs (est.)", f"{fo['n_vp']:,.0f}".replace(",", " "))
    a4.metric("Faux positifs (est.)", f"{fo['n_fp']:,.0f}".replace(",", " "))

    st.info(
        "Une **alerte** = score ≥ seuil. "
        "Les vrais positifs sont les fraudes bien détectées ; "
        "les faux positifs génèrent un coût de traitement humain (~2 500 Ar)."
    )

    # Camembert VP / FP
    fig_pie = px.pie(
        names=["Vrais positifs", "Faux positifs"],
        values=[fo["n_vp"], fo["n_fp"]],
        color_discrete_sequence=["#38A169", "#E53E3E"],
        title="Composition des alertes (estimée)",
    )
    fig_pie.update_layout(height=320)
    st.plotly_chart(fig_pie, use_container_width=True)

# ===== TAB FINOPS =====
with tab_finops:
    st.subheader("FinOps — coûts, gains et ROI (Ariary)")
    st.caption(
        f"Période : 24 mois · {N_FACTURES_PROJET:,} factures · "
        f"taux fraude {TAUX_FRAUDE*100:.2f} % · "
        f"montant moyen fraude {MONTANT_MOYEN_FRAUDE_AR:,} Ar"
    )

    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Montant économisé", fmt_ar(fo["montant_economise"]))
    f2.metric("Coût fausses alertes", fmt_ar(fo["cout_fp"]))
    f3.metric("Infra 24 mois", fmt_ar(fo["infra"]))
    f4.metric("Gain net", fmt_ar(fo["gain_net"]))

    st.markdown(f"### ROI estimé : **{fo['roi']:,.0f} %**")

    # Waterfall simplifié
    fig_w = go.Figure(
        go.Waterfall(
            name="FinOps",
            orientation="v",
            measure=["absolute", "relative", "relative", "total"],
            x=["Économisé", "− Fausses alertes", "− Infra", "Gain net"],
            y=[
                fo["montant_economise"],
                -fo["cout_fp"],
                -fo["infra"],
                fo["gain_net"],
            ],
            text=[
                fmt_ar(fo["montant_economise"]),
                fmt_ar(-fo["cout_fp"]),
                fmt_ar(-fo["infra"]),
                fmt_ar(fo["gain_net"]),
            ],
            textposition="outside",
            connector={"line": {"color": "#A0AEC0"}},
            increasing={"marker": {"color": "#38A169"}},
            decreasing={"marker": {"color": "#E53E3E"}},
            totals={"marker": {"color": "#2B6CB0"}},
        )
    )
    fig_w.update_layout(height=360, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_w, use_container_width=True)

    st.markdown(
        """
**Formules**
- Montant économisé ≈ fraudes × rappel × montant moyen
- Coût FP ≈ (alertes − VP) × 2 500 Ar
- Gain net = économisé − coût FP − infra 24 mois
- ROI = gain net / infra × 100

*Hypothèses documentées — à valider avec le métier.*
"""
    )

# ===== TAB COMPARAISON =====
with tab_cmp:
    st.subheader("Comparaison des 3 versions")
    rows = []
    for ver in ("v1", "v2", "v3"):
        vv = load_version_meta(ver)
        ff = finops_for(vv)
        rows.append(
            {
                "Version": vv["label"],
                "Seuil": round(vv["seuil"], 4),
                "Recall": f"{vv['recall']*100:.1f}%",
                "Précision": f"{vv['precision']*100:.1f}%",
                "% alertes": f"{vv['pct_alertes']:.1f}%",
                "Économisé (Ar)": fmt_ar(ff["montant_economise"]),
                "Coût FP (Ar)": fmt_ar(ff["cout_fp"]),
                "Gain net (Ar)": fmt_ar(ff["gain_net"]),
                "ROI": f"{ff['roi']:,.0f}%",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Graph recall vs précision
    df_cmp = pd.DataFrame(
        [
            {
                "Version": ver.upper(),
                "Recall": load_version_meta(ver)["recall"] * 100,
                "Précision": load_version_meta(ver)["precision"] * 100,
                "% alertes": load_version_meta(ver)["pct_alertes"],
                "Gain net (M Ar)": finops_for(load_version_meta(ver))["gain_net"] / 1e6,
            }
            for ver in ("v1", "v2", "v3")
        ]
    )
    g1, g2 = st.columns(2)
    with g1:
        fig1 = px.bar(
            df_cmp,
            x="Version",
            y=["Recall", "Précision"],
            barmode="group",
            title="Recall vs Précision (%)",
            color_discrete_sequence=["#2B6CB0", "#38A169"],
        )
        st.plotly_chart(fig1, use_container_width=True)
    with g2:
        fig2 = px.bar(
            df_cmp,
            x="Version",
            y="Gain net (M Ar)",
            title="Gain net estimé (millions Ar)",
            color="Version",
            color_discrete_sequence=["#2B6CB0", "#38A169", "#C05621"],
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.success(
        "**Recommandation :** v2 — meilleur équilibre entre fraudes attrapées, "
        "charge des gestionnaires et gain net."
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "PrediCare TPA · Dashboard M1 (modèles & FinOps) · "
    "Les coûts sont des estimations documentées · "
    "Dashboard officiel CAPSTONE = Superset + schéma bi"
)
