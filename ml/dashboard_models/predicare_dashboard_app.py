# -*- coding: utf-8 -*-
"""
PrediCare TPA — Tableau de bord modèles et analyse financière
Lancement depuis ce dossier :
    python -m streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------
# Configuration générale
# ---------------------------------------------------------------------
NAVY = "#16233F"
NAVY_LIGHT = "#243858"
CREAM = "#F8EFDF"
WHITE = "#FFFFFF"
BLUE = "#4776A8"
GREEN = "#34745A"
ORANGE = "#B56C2A"
RED = "#B44747"
MUTED = "#68758A"
BORDER = "#E6E1D8"

st.set_page_config(
    page_title="PrediCare TPA | Tableau de bord",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; }}
    .stApp {{ background: {CREAM}; color: {NAVY}; }}
    [data-testid="stSidebar"] {{ background: {NAVY}; border-right: 0; }}
    [data-testid="stSidebar"] * {{ color: #F4F1EA; }}
    [data-testid="stSidebar"] .stCaption {{ color: #C5CEDB !important; }}
    .block-container {{ max-width: 1440px; padding-top: 1.6rem; padding-bottom: 2.5rem; }}
    h1, h2, h3 {{ color: {NAVY} !important; letter-spacing: -0.025em; }}
    h1 {{ font-size: 2rem !important; font-weight: 700 !important; }}
    h2 {{ font-size: 1.45rem !important; font-weight: 700 !important; }}
    h3 {{ font-size: 1.08rem !important; font-weight: 700 !important; }}
    p, label, .stMarkdown {{ color: {NAVY}; }}
    div[data-testid="stMetric"] {{
        background: {WHITE}; border: 1px solid {BORDER}; border-radius: 14px;
        padding: 1rem 1.1rem; box-shadow: 0 3px 12px rgba(22,35,63,.035);
    }}
    div[data-testid="stMetricLabel"] p {{ color: {MUTED} !important; font-size: .82rem; }}
    div[data-testid="stMetricValue"] {{ color: {NAVY} !important; font-weight: 700; }}
    div[data-testid="stMetricDelta"] {{ font-size: .78rem; }}
    .eyebrow {{ color: {MUTED}; font-size: .73rem; letter-spacing: .12em; font-weight: 700; text-transform: uppercase; }}
    .subtitle {{ color: {MUTED}; font-size: .92rem; margin-top: -.5rem; }}
    .panel {{ background: {WHITE}; border: 1px solid {BORDER}; border-radius: 15px; padding: 1.1rem 1.25rem; }}
    .notice {{ background: #F0F3F8; border: 1px solid #D9E1EC; border-radius: 12px; padding: .9rem 1rem; color: {NAVY}; }}
    .small-muted {{ color: {MUTED}; font-size: .8rem; }}
    div[data-testid="stTabs"] button {{ color: {NAVY}; font-weight: 600; }}
    .stButton > button {{ border-radius: 9px; border: 1px solid {NAVY}; background: {NAVY}; color: white; font-weight: 600; }}
    .stButton > button:hover {{ background: {NAVY_LIGHT}; border-color: {NAVY_LIGHT}; color: white; }}
    [data-testid="stNumberInput"] input {{ border-radius: 8px; }}
    hr {{ border-color: {BORDER}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Modèles : métriques par défaut, remplaçables par metadata.json si présent
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CANDIDATES = [ROOT.parent / "models", ROOT.parent.parent / "models", ROOT / "models"]


def find_models_dir():
    for path in CANDIDATES:
        if path.is_dir():
            return path
    return None


MODELS_DIR = find_models_dir()

DEFAULTS = {
    "v1": {"label": "Version 1 — Prudente", "seuil": 0.7156, "recall": 0.9247,
           "precision": 0.9290, "f1": 0.9268, "pr_auc": 0.9721, "roc_auc": 0.9927,
           "pct_alertes": 8.5, "n_test": 12689, "n_fraudes_test": 1089, "n_alertes_test": 1084,
           "rappel_cible": 0.92, "color": BLUE},
    "v2": {"label": "Version 2 — Recommandée", "seuil": 0.5429, "recall": 0.9467,
           "precision": 0.8737, "f1": 0.9088, "pr_auc": 0.9721, "roc_auc": 0.9927,
           "pct_alertes": 9.3, "n_test": 12689, "n_fraudes_test": 1089, "n_alertes_test": 1180,
           "rappel_cible": 0.95, "color": GREEN},
    "v3": {"label": "Version 3 — Détection maximale", "seuil": 0.1457, "recall": 0.9798,
           "precision": 0.5340, "f1": 0.6913, "pr_auc": 0.9721, "roc_auc": 0.9927,
           "pct_alertes": 15.7, "n_test": 12689, "n_fraudes_test": 1089, "n_alertes_test": 1998,
           "rappel_cible": 0.98, "color": ORANGE},
}

RECALL_PAR_TYPE = {
    "SURFACTURATION": 1.00, "LIGNES_NON_PRESCRITES": 1.00, "ORDONNANCE_EXPIREE": 1.00,
    "QUANTITE_EXCESSIVE": 1.00, "PRESTATAIRE_RADIE": 1.00, "SUBSTITUTION_PRODUIT": 1.00,
    "SOINS_JOUR_FERME": 1.00, "HORS_COUVERTURE": 1.00, "SURCONSOMMATION": 0.975,
    "FRACTIONNEMENT": 0.936, "ACTE_FICTIF": 0.967, "UPCODING": 0.865,
    "DOUBLON": 0.804, "INCOHERENCE_PROFIL": 0.75,
}


def load_version_meta(version: str) -> dict:
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
        results = meta.get("resultats_test") or {}
        for key in ("recall", "precision", "f1", "pr_auc", "roc_auc", "pct_alertes"):
            if key in results:
                base[key] = float(results[key])
        for key in ("n_test", "n_alertes_test", "n_fraudes_test"):
            if key in results:
                base[key] = int(results[key])
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return base


def fmt_ar(value: float) -> str:
    return f"{value:,.0f} Ar".replace(",", " ")


def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f} %"


def financial_estimate(v: dict, n_invoices: int, fraud_rate: float,
                       cost_per_review: float, average_fraud: float,
                       monthly_infra: float, months: int) -> dict:
    """Simulation indicative; les fraudes détectées ne sont pas forcément toutes évitées."""
    n_frauds = n_invoices * fraud_rate
    n_true_positive = n_frauds * v["recall"]
    n_alerts = n_invoices * (v["pct_alertes"] / 100)
    n_false_alerts = max(0.0, n_alerts - n_true_positive)
    gross_savings = n_true_positive * average_fraud
    review_cost = n_false_alerts * cost_per_review
    infra_total = monthly_infra * months
    total_costs = review_cost + infra_total
    net_gain = gross_savings - total_costs
    roi = (net_gain / total_costs * 100) if total_costs > 0 else 0
    return {
        "n_frauds": n_frauds,
        "n_true_positive": n_true_positive,
        "n_alerts": n_alerts,
        "n_false_alerts": n_false_alerts,
        "gross_savings": gross_savings,
        "review_cost": review_cost,
        "infra_total": infra_total,
        "total_costs": total_costs,
        "net_gain": net_gain,
        "roi": roi,
        "cost_per_invoice": infra_total / n_invoices if n_invoices else 0,
        "cost_per_alert": total_costs / n_alerts if n_alerts else 0,
        "payback_months": infra_total / (gross_savings / months - review_cost / months)
        if months > 0 and gross_savings > review_cost else float("inf"),
    }


def main():
    with st.sidebar:
        st.markdown(
            f"""<div style="padding:.4rem 0 1.1rem">
            <div style="font-size:1.35rem;font-weight:700;color:{CREAM}">PrediCare</div>
            <div style="font-size:.78rem;color:#C5CEDB">Protection et contrôle des remboursements</div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.divider()
        page = st.radio(
            "NAVIGATION",
            ["Tableau de bord", "Analyse financière", "Performance des modèles", "Alertes estimées"],
            label_visibility="visible",
        )
        st.divider()
        st.caption("ENVIRONNEMENT DE DÉMONSTRATION")
        if MODELS_DIR:
            st.caption("Dossier des modèles détecté.")
        else:
            st.caption("Métriques de démonstration chargées.")
        st.markdown("---")
        st.caption("La décision finale appartient au gestionnaire.")

    st.markdown('<div class="eyebrow">ESPACE DE GESTION  /  PREDICARE TPA</div>', unsafe_allow_html=True)
    st.title(page)
    st.markdown('<div class="subtitle">Suivi des remboursements, des alertes et des hypothèses financières.</div>', unsafe_allow_html=True)
    st.write("")

    with st.sidebar:
        selected_version = st.selectbox(
            "Version du modèle",
            options=["v1", "v2", "v3"],
            index=1,
            format_func=lambda key: DEFAULTS[key]["label"],
        )

    v = load_version_meta(selected_version)

    # Paramètres de simulation modifiables par l'utilisateur
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Paramètres de simulation")
        n_invoices = st.number_input("Nombre de factures analysées", min_value=1, value=84593, step=1000)
        fraud_rate_pct = st.number_input("Taux de fraude estimé (%)", min_value=0.0, max_value=100.0, value=8.69, step=0.1)
        cost_per_review = st.number_input("Coût d'examen d'une alerte (Ar)", min_value=0, value=2500, step=500)
        average_fraud = st.number_input("Montant moyen d'une fraude (Ar)", min_value=0, value=150000, step=10000)
        monthly_infra = st.number_input("Coût mensuel d'exploitation (Ar)", min_value=0, value=1750000, step=100000)
        months = st.number_input("Période d'analyse (mois)", min_value=1, value=24, step=1)
        st.caption("Les montants préremplis sont des hypothèses modifiables, pas des coûts constatés.")

    fo = financial_estimate(
        v, int(n_invoices), float(fraud_rate_pct) / 100,
        float(cost_per_review), float(average_fraud), float(monthly_infra), int(months)
    )

    if page == "Tableau de bord":
        st.markdown("### Vue d'ensemble")
        a, b, c, d = st.columns(4)
        a.metric("Factures analysées", f"{int(n_invoices):,}".replace(",", " "))
        b.metric("Taux de fraude estimé", f"{fraud_rate_pct:.2f} %")
        c.metric("Fraudes potentiellement détectées", f"{fo['n_true_positive']:,.0f}".replace(",", " "))
        d.metric("Alertes à examiner", f"{fo['n_alerts']:,.0f}".replace(",", " "))

        st.markdown("")
        left, right = st.columns([1.5, 1])
        with left:
            trend = pd.DataFrame({
                "Version": ["Version prudente", "Version recommandée", "Détection maximale"],
                "Taux de fraudes détectées (%)": [
                    DEFAULTS["v1"]["recall"] * 100,
                    DEFAULTS["v2"]["recall"] * 100,
                    DEFAULTS["v3"]["recall"] * 100,
                ],
            })
            fig = px.bar(trend, x="Version", y="Taux de fraudes détectées (%)",
                         title="Capacité de détection par version",
                         color="Version",
                         color_discrete_sequence=[BLUE, GREEN, ORANGE],
                         range_y=[0, 105])
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              font_color=NAVY, showlegend=False, margin=dict(l=15, r=15, t=55, b=15))
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("#### État de la simulation")
            st.write(f"Version sélectionnée : **{v['label']}**")
            st.write(f"Seuil de déclenchement : **{v['seuil']:.4f}**")
            st.write(f"Factures envoyées en examen : **{v['pct_alertes']:.1f} %**")
            st.write(f"Résultats financiers calculés sur **{int(months)} mois**.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("")
        st.markdown('<div class="notice"><strong>Important :</strong> les alertes signalent des dossiers à examiner. Elles ne constituent pas une preuve de fraude et les économies affichées sont des estimations de scénario.</div>', unsafe_allow_html=True)

    elif page == "Analyse financière":
        st.markdown("### Hypothèses financières modifiables")
        st.markdown("Ajustez les valeurs dans le panneau latéral. Les indicateurs et graphiques se mettent à jour automatiquement.")
        st.caption("Les valeurs préremplies servent uniquement à illustrer le calcul. Remplacez-les dès que vous disposez de données vérifiées.")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Économies brutes estimées", fmt_ar(fo["gross_savings"]))
        k2.metric("Coût des examens des alertes", fmt_ar(fo["review_cost"]))
        k3.metric(f"Coût d'exploitation sur {int(months)} mois", fmt_ar(fo["infra_total"]))
        k4.metric("Gain net estimé", fmt_ar(fo["net_gain"]))

        k5, k6, k7 = st.columns(3)
        k5.metric("ROI estimé", f"{fo['roi']:,.1f} %".replace(",", " "))
        k6.metric("Coût total estimé", fmt_ar(fo["total_costs"]))
        payback = fo["payback_months"]
        k7.metric("Délai estimé pour couvrir les coûts", f"{payback:.1f} mois" if payback != float("inf") else "Non calculable")

        st.markdown("")
        left, right = st.columns([1.2, 1])
        with left:
            waterfall = go.Figure(go.Waterfall(
                measure=["absolute", "relative", "relative", "total"],
                x=["Économies brutes", "Coût des examens", "Coût d'exploitation", "Gain net"],
                y=[fo["gross_savings"], -fo["review_cost"], -fo["infra_total"], fo["net_gain"]],
                text=[fmt_ar(fo["gross_savings"]), "-" + fmt_ar(fo["review_cost"]),
                      "-" + fmt_ar(fo["infra_total"]), fmt_ar(fo["net_gain"])],
                textposition="outside",
                increasing={"marker": {"color": GREEN}},
                decreasing={"marker": {"color": RED}},
                totals={"marker": {"color": NAVY}},
                connector={"line": {"color": BORDER}},
            ))
            waterfall.update_layout(title="Estimation des gains et des coûts", height=390,
                                    paper_bgcolor=WHITE, plot_bgcolor=WHITE, font_color=NAVY,
                                    margin=dict(l=15, r=15, t=55, b=20), yaxis_title="Ariary")
            st.plotly_chart(waterfall, use_container_width=True)
        with right:
            breakdown = pd.DataFrame({
                "Poste": ["Coût des examens", "Coût d'exploitation"],
                "Montant (Ar)": [fo["review_cost"], fo["infra_total"]],
            })
            fig = px.pie(breakdown, names="Poste", values="Montant (Ar)", hole=.62,
                         title="Répartition des coûts", color_discrete_sequence=[ORANGE, NAVY])
            fig.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, font_color=NAVY,
                              margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Détail des indicateurs")
        details = pd.DataFrame([
            ["Nombre de factures", f"{int(n_invoices):,}".replace(",", " ")],
            ["Fraudes estimées dans le volume", f"{fo['n_frauds']:,.0f}".replace(",", " ")],
            ["Fraudes potentiellement détectées", f"{fo['n_true_positive']:,.0f}".replace(",", " ")],
            ["Alertes à examiner", f"{fo['n_alerts']:,.0f}".replace(",", " ")],
            ["Alertes probablement non frauduleuses (estimation)", f"{fo['n_false_alerts']:,.0f}".replace(",", " ")],
            ["Coût moyen par facture (exploitation)", fmt_ar(fo["cost_per_invoice"])],
            ["Coût total par alerte (estimation)", fmt_ar(fo["cost_per_alert"])],
        ], columns=["Indicateur", "Valeur"])
        st.dataframe(details, use_container_width=True, hide_index=True)
        st.markdown('<div class="notice">Les économies brutes sont calculées à partir du taux de fraude et du taux de détection supposés. Une fraude détectée n’est pas nécessairement une perte évitée : confirmez les montants réellement bloqués avant d’utiliser ces résultats comme bilan financier.</div>', unsafe_allow_html=True)

    elif page == "Performance des modèles":
        st.markdown("### Résultats du jeu de test")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fraudes détectées (rappel)", fmt_pct(v["recall"]))
        c2.metric("Alertes correctes (précision)", fmt_pct(v["precision"]))
        c3.metric("Score F1", f"{v['f1']:.3f}")
        c4.metric("Seuil de déclenchement", f"{v['seuil']:.4f}")

        perf = pd.DataFrame([
            {"Version": DEFAULTS[key]["label"], "Fraudes détectées (%)": DEFAULTS[key]["recall"] * 100,
             "Alertes correctes (%)": DEFAULTS[key]["precision"] * 100}
            for key in ("v1", "v2", "v3")
        ])
        fig = px.bar(perf, x="Version", y=["Fraudes détectées (%)", "Alertes correctes (%)"],
                     barmode="group", title="Comparaison des performances",
                     color_discrete_sequence=[BLUE, GREEN])
        fig.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, font_color=NAVY,
                          margin=dict(l=15, r=15, t=55, b=15))
        st.plotly_chart(fig, use_container_width=True)

        by_type = pd.DataFrame({"Type de fraude": list(RECALL_PAR_TYPE.keys()),
                                "Taux de détection": list(RECALL_PAR_TYPE.values())})
        by_type["Taux de détection"] *= 100
        fig2 = px.bar(by_type.sort_values("Taux de détection"), x="Taux de détection",
                      y="Type de fraude", orientation="h", title="Détection par type de fraude",
                      range_x=[0, 105], color_discrete_sequence=[BLUE])
        fig2.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, font_color=NAVY,
                           height=450, margin=dict(l=15, r=15, t=55, b=15))
        st.plotly_chart(fig2, use_container_width=True)

    elif page == "Alertes estimées":
        st.markdown("### Volume prévisionnel des alertes")
        st.caption("Estimations basées sur le volume saisi et le taux d'alertes de la version sélectionnée.")
        a, b, c, d = st.columns(4)
        a.metric("Alertes à examiner", f"{fo['n_alerts']:,.0f}".replace(",", " "))
        b.metric("Fraudes potentiellement détectées", f"{fo['n_true_positive']:,.0f}".replace(",", " "))
        c.metric("Alertes probablement non frauduleuses", f"{fo['n_false_alerts']:,.0f}".replace(",", " "))
        d.metric("Part des factures alertées", f"{v['pct_alertes']:.1f} %")

        alert_data = pd.DataFrame({
            "Catégorie": ["Fraudes potentiellement détectées", "Alertes probablement non frauduleuses"],
            "Nombre estimé": [fo["n_true_positive"], fo["n_false_alerts"]],
        })
        fig = px.pie(alert_data, names="Catégorie", values="Nombre estimé", hole=.5,
                     title="Répartition estimée des alertes", color_discrete_sequence=[GREEN, ORANGE])
        fig.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, font_color=NAVY,
                          margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="notice">Une alerte est un signal à vérifier, pas une conclusion automatique. Les faux positifs sont estimés à partir du taux d’alertes et du rappel du modèle ; ils doivent être remplacés par les résultats réels des vérifications humaines lorsqu’ils seront disponibles.</div>', unsafe_allow_html=True)

    st.divider()
    st.caption("PrediCare TPA · Tableau de bord de démonstration · Les hypothèses financières sont modifiables.")


if __name__ == "__main__":
    main()
