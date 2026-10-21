# -*- coding: utf-8 -*-
"""
PrediCare TPA — Tableau de bord modèles et analyse financière
Lancement : python -m streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------
# Couleurs
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
    [data-testid="stSidebar"] {{ background: {NAVY}; }}
    [data-testid="stSidebar"] * {{ color: #F4F1EA; }}
    [data-testid="stSidebar"] .stCaption {{ color: #C5CEDB !important; }}
    .block-container {{ max-width: 1180px; padding-top: 1.4rem; padding-bottom: 2rem; }}
    h1, h2, h3 {{ color: {NAVY} !important; letter-spacing: -0.02em; }}
    h1 {{ font-size: 1.85rem !important; font-weight: 700 !important; }}
    h2 {{ font-size: 1.2rem !important; font-weight: 700 !important; }}
    div[data-testid="stMetric"] {{
        background: {WHITE}; border: 1px solid {BORDER}; border-radius: 14px;
        padding: 1rem 1.1rem; box-shadow: 0 2px 10px rgba(22,35,63,.04);
    }}
    div[data-testid="stMetricLabel"] p {{ color: {MUTED} !important; font-size: .8rem; }}
    div[data-testid="stMetricValue"] {{ color: {NAVY} !important; font-weight: 700; }}
    .eyebrow {{ color: {MUTED}; font-size: .72rem; letter-spacing: .14em; font-weight: 700; text-transform: uppercase; }}
    .subtitle {{ color: {MUTED}; font-size: .9rem; margin: -.35rem 0 0.8rem 0; }}
    .card {{
        background: {WHITE}; border: 1px solid {BORDER}; border-radius: 16px;
        padding: 1.25rem 1.4rem; box-shadow: 0 4px 16px rgba(22,35,63,.05);
        margin-bottom: .85rem;
    }}
    .form-shell {{
        background: {WHITE}; border: 1px solid {BORDER}; border-radius: 18px;
        padding: 1.5rem 1.6rem 1.2rem; box-shadow: 0 8px 28px rgba(22,35,63,.08);
        margin: .5rem 0 1.25rem 0;
    }}
    .form-title {{ font-size: 1.1rem; font-weight: 700; color: {NAVY}; margin-bottom: .25rem; }}
    .form-sub {{ color: {MUTED}; font-size: .85rem; margin-bottom: 1rem; }}
    .badge {{
        display: inline-block; background: {NAVY}; color: #F8EFDF; font-size: .72rem;
        font-weight: 700; padding: .28rem .65rem; border-radius: 999px;
    }}
    .hint {{ color: {MUTED}; font-size: .82rem; line-height: 1.45; }}
    .stButton > button {{
        border-radius: 10px; border: 1px solid {NAVY}; background: {NAVY};
        color: white; font-weight: 600; padding: .45rem 1rem;
    }}
    .stButton > button:hover {{ background: {NAVY_LIGHT}; border-color: {NAVY_LIGHT}; color: white; }}
    /* Inputs clairs dans le formulaire */
    .form-shell [data-testid="stNumberInput"] input,
    .form-shell [data-baseweb="select"] > div {{
        background: #FAFBFC !important; border-radius: 10px;
    }}
    hr {{ border-color: {BORDER}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Modèles
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CANDIDATES = [ROOT.parent / "models", ROOT.parent.parent / "models", ROOT / "models"]


def find_models_dir() -> Path | None:
    for p in CANDIDATES:
        if p.is_dir():
            return p
    return None


MODELS_DIR = find_models_dir()

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
        "color": BLUE,
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
        "color": GREEN,
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
        "color": ORANGE,
    },
}

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


def fmt_ar(x: float) -> str:
    ax = abs(x)
    if ax >= 1_000_000_000:
        return f"{x / 1_000_000_000:.2f} Md Ar"
    if ax >= 1_000_000:
        return f"{x / 1_000_000:.1f} M Ar"
    return f"{x:,.0f} Ar".replace(",", " ")


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f} %" if x <= 1 else f"{x:.1f} %"


# Session
defaults_state = {
    "params_open": False,
    "version": "v2",
    "n_factures": 84_593,
    "taux_fraude_pct": 8.69,
    "minutes_dossier": 15.0,
    "cout_horaire": 10_000.0,
    "montant_moyen": 150_000.0,
    "cout_mensuel": 1_750_000.0,
    "n_mois": 24,
}
for k, val in defaults_state.items():
    if k not in st.session_state:
        st.session_state[k] = val

# ---------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------
st.sidebar.markdown("### PrediCare TPA")
st.sidebar.caption("Pilotage · Finance · Modèles")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Tableau de bord", "Analyse financière", "Performance", "Alertes"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption("La décision finale appartient au gestionnaire.")

# ---------------------------------------------------------------------
# En-tête : titre + choix modèle + bouton paramètres
# ---------------------------------------------------------------------
st.markdown('<p class="eyebrow">PrediCare TPA</p>', unsafe_allow_html=True)
st.title("Votre activité en un regard")
st.markdown(
    '<p class="subtitle">Modèles, alertes et analyse financière · montants en Ariary</p>',
    unsafe_allow_html=True,
)

bar1, bar2, bar3 = st.columns([1.4, 1.1, 1.0])
with bar1:
    version_ui = st.selectbox(
        "Modèle",
        ["v1", "v2", "v3"],
        index=["v1", "v2", "v3"].index(st.session_state.version),
        format_func=lambda x: DEFAULTS[x]["label"],
        label_visibility="collapsed",
        key="version_header",
    )
    if version_ui != st.session_state.version:
        st.session_state.version = version_ui
        st.rerun()
with bar2:
    if st.button("Paramètres de calcul", use_container_width=True):
        st.session_state.params_open = not st.session_state.params_open
with bar3:
    if st.session_state.params_open:
        if st.button("Fermer", use_container_width=True):
            st.session_state.params_open = False
            st.rerun()

# ---------------------------------------------------------------------
# Formulaire blanc (uniquement si ouvert)
# ---------------------------------------------------------------------
if st.session_state.params_open:
    st.markdown(
        """
        <div class="form-shell">
          <div class="form-title">Paramètres de calcul</div>
          <div class="form-sub">
            Ajustez les hypothèses métier. Le modèle se choisit dans la barre du haut.
            Coût d’examen = (minutes ÷ 60) × coût horaire.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("form_params"):
        c1, c2 = st.columns(2)
        with c1:
            n_factures = st.number_input(
                "Nombre de factures",
                min_value=1,
                value=int(st.session_state.n_factures),
                step=1000,
            )
            minutes_dossier = st.number_input(
                "Minutes par dossier",
                min_value=1.0,
                value=float(st.session_state.minutes_dossier),
                step=1.0,
            )
            montant_moyen = st.number_input(
                "Montant moyen d’une fraude (Ar)",
                min_value=0.0,
                value=float(st.session_state.montant_moyen),
                step=5000.0,
            )
            cout_mensuel = st.number_input(
                "Coût mensuel d’exploitation (Ar)",
                min_value=0.0,
                value=float(st.session_state.cout_mensuel),
                step=50000.0,
            )
        with c2:
            taux_fraude_pct = st.number_input(
                "Taux de fraude (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(st.session_state.taux_fraude_pct),
                step=0.01,
            )
            cout_horaire = st.number_input(
                "Coût horaire chargé (Ar)",
                min_value=0.0,
                value=float(st.session_state.cout_horaire),
                step=500.0,
            )
            n_mois = st.number_input(
                "Période (mois)",
                min_value=1,
                value=int(st.session_state.n_mois),
                step=1,
            )
            cout_prev = (minutes_dossier / 60.0) * cout_horaire
            st.metric("Coût d’examen calculé", f"{cout_prev:,.0f} Ar".replace(",", " "))

        applied = st.form_submit_button("Appliquer", use_container_width=True)
        if applied:
            st.session_state.n_factures = int(n_factures)
            st.session_state.taux_fraude_pct = float(taux_fraude_pct)
            st.session_state.minutes_dossier = float(minutes_dossier)
            st.session_state.cout_horaire = float(cout_horaire)
            st.session_state.montant_moyen = float(montant_moyen)
            st.session_state.cout_mensuel = float(cout_mensuel)
            st.session_state.n_mois = int(n_mois)
            st.session_state.params_open = False
            st.rerun()

# ---------------------------------------------------------------------
# Calculs
# ---------------------------------------------------------------------
version = st.session_state.version
v = load_version_meta(version)
n_factures = st.session_state.n_factures
taux_fraude = st.session_state.taux_fraude_pct / 100.0
minutes_dossier = st.session_state.minutes_dossier
cout_horaire = st.session_state.cout_horaire
montant_moyen = st.session_state.montant_moyen
cout_mensuel = st.session_state.cout_mensuel
n_mois = st.session_state.n_mois

cout_examen = (minutes_dossier / 60.0) * cout_horaire
n_fraudes = n_factures * taux_fraude
n_vp = n_fraudes * v["recall"]
n_alertes = n_factures * (v["pct_alertes"] / 100.0)
n_fp = max(0.0, n_alertes - n_vp)

economies = n_vp * montant_moyen
cout_examens = n_fp * cout_examen
cout_exploitation = cout_mensuel * n_mois
cout_total = cout_examens + cout_exploitation
gain_net = economies - cout_total
roi = (gain_net / cout_total * 100) if cout_total > 0 else 0.0
gain_mensuel = max(0.0, (economies - cout_examens) / max(n_mois, 1))
payback = (cout_exploitation / gain_mensuel) if gain_mensuel > 0 else None

st.markdown(
    f'<span class="badge">{v["label"]}</span>'
    f'&nbsp; <span class="hint">Seuil {v["seuil"]:.4f} · '
    f"Examen {fmt_ar(cout_examen)} · {n_mois} mois</span>",
    unsafe_allow_html=True,
)
st.write("")

# ---------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------
if page == "Tableau de bord":
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Argent économisé", fmt_ar(economies))
    k2.metric("Gain net", fmt_ar(gain_net), f"ROI {roi:,.0f} %")
    k3.metric("Coût total", fmt_ar(cout_total))
    k4.metric("Alertes", f"{n_alertes:,.0f}".replace(",", " "))

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("### Lecture rapide")
        ratio = (economies / cout_exploitation) if cout_exploitation > 0 else 0
        st.markdown(
            f"""
            <div class="card">
              <p><b>Gain vs exploitation</b><br/>
              Économies <b>{fmt_ar(economies)}</b> pour un coût d’exploitation
              de <b>{fmt_ar(cout_exploitation)}</b>
              (environ <b>{ratio:.0f}×</b> l’infra sur la période).</p>
              <p><b>Version recommandée : v2</b><br/>
              Meilleur équilibre entre fraudes attrapées, charge des équipes et ROI.
              v1 est plus prudent ; v3 augmente les alertes et les coûts de traitement.</p>
              <p class="hint">Le gestionnaire valide chaque alerte.
              Les montants affichés dépendent des paramètres et des performances du modèle.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("### Composition des alertes")
        fig_pie = px.pie(
            names=["Fraudes détectées", "Fausses alertes"],
            values=[max(n_vp, 0.01), max(n_fp, 0.01)],
            color_discrete_sequence=[GREEN, RED],
        )
        fig_pie.update_layout(
            height=290,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor=CREAM,
            font_color=NAVY,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

elif page == "Analyse financière":
    st.markdown("### Analyse financière")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Économies brutes", fmt_ar(economies))
    a2.metric("Coût examens (FP)", fmt_ar(cout_examens))
    a3.metric(f"Exploitation ({n_mois} mois)", fmt_ar(cout_exploitation))
    a4.metric("Gain net · ROI", f"{fmt_ar(gain_net)} · {roi:,.0f} %")

    fig_w = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "total"],
            x=["Économisé", "− Examens", "− Exploitation", "Gain net"],
            y=[economies, -cout_examens, -cout_exploitation, gain_net],
            text=[fmt_ar(economies), fmt_ar(-cout_examens), fmt_ar(-cout_exploitation), fmt_ar(gain_net)],
            textposition="outside",
            connector={"line": {"color": BORDER}},
            increasing={"marker": {"color": GREEN}},
            decreasing={"marker": {"color": RED}},
            totals={"marker": {"color": NAVY}},
        )
    )
    fig_w.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor=CREAM,
        plot_bgcolor=WHITE,
        font_color=NAVY,
    )
    st.plotly_chart(fig_w, use_container_width=True)

    b1, b2, b3 = st.columns(3)
    b1.metric("Payback", f"{payback:.1f} mois" if payback else "—")
    b2.metric("Coût / facture", fmt_ar(cout_exploitation / n_factures if n_factures else 0))
    b3.metric("Coût / alerte", fmt_ar(cout_examen))

    st.markdown("### Comparaison des 3 versions")
    rows = []
    for ver in ("v1", "v2", "v3"):
        vv = load_version_meta(ver)
        n_fra = n_factures * taux_fraude
        vp = n_fra * vv["recall"]
        al = n_factures * (vv["pct_alertes"] / 100.0)
        fp = max(0.0, al - vp)
        eco = vp * montant_moyen
        cex = fp * cout_examen
        ctot = cex + cout_exploitation
        gn = eco - ctot
        r = (gn / ctot * 100) if ctot > 0 else 0
        rows.append(
            {
                "Version": vv["label"],
                "Seuil": round(vv["seuil"], 4),
                "Rappel": f"{vv['recall']*100:.1f} %",
                "Précision": f"{vv['precision']*100:.1f} %",
                "% alertes": f"{vv['pct_alertes']:.1f} %",
                "Économisé": fmt_ar(eco),
                "Coût FP": fmt_ar(cex),
                "Gain net": fmt_ar(gn),
                "ROI": f"{r:,.0f} %",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.success(
        "**v2 recommandée** — meilleur équilibre fraudes / charge / ROI. "
        "Dans les trois cas, le gain net reste très supérieur au coût d’exploitation."
    )

elif page == "Performance":
    st.markdown(f"### Performance — {v['label']}")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Rappel", fmt_pct(v["recall"]))
    p2.metric("Précision", fmt_pct(v["precision"]))
    p3.metric("F1", fmt_pct(v["f1"]))
    p4.metric("Seuil", f"{v['seuil']:.4f}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"""
            | Indicateur | Valeur |
            |---|---|
            | Rappel cible | {fmt_pct(v['rappel_cible'])} |
            | PR-AUC | {v['pr_auc']:.4f} |
            | ROC-AUC | {v['roc_auc']:.4f} |
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
                        {"range": [0, 0.3], "color": "#F4D6D6"},
                        {"range": [0.3, 0.6], "color": "#F3E6D0"},
                        {"range": [0.6, 1], "color": "#D5EADF"},
                    ],
                },
                number={"valueformat": ".3f"},
            )
        )
        fig_g.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor=CREAM,
            font_color=NAVY,
        )
        st.plotly_chart(fig_g, use_container_width=True)

    st.markdown("### Rappel par type de fraude")
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
    fig_t.update_layout(
        height=400,
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=CREAM,
        plot_bgcolor=WHITE,
        font_color=NAVY,
    )
    st.plotly_chart(fig_t, use_container_width=True)

else:
    st.markdown("### Alertes")
    x1, x2, x3, x4 = st.columns(4)
    x1.metric("Alertes", f"{n_alertes:,.0f}".replace(",", " "))
    x2.metric("% factures alertées", f"{v['pct_alertes']:.1f} %")
    x3.metric("Fraudes détectées", f"{n_vp:,.0f}".replace(",", " "))
    x4.metric("Fausses alertes", f"{n_fp:,.0f}".replace(",", " "))

    st.markdown(
        f"""
        <div class="card hint">
        Une alerte = score ≥ seuil (<b>{v['seuil']:.4f}</b>).
        Le gestionnaire confirme ou infirme chaque dossier.
        </div>
        """,
        unsafe_allow_html=True,
    )
    fig_pie = px.pie(
        names=["Vrais positifs", "Faux positifs"],
        values=[max(n_vp, 0.01), max(n_fp, 0.01)],
        color_discrete_sequence=[GREEN, RED],
    )
    fig_pie.update_layout(height=300, paper_bgcolor=CREAM, font_color=NAVY)
    st.plotly_chart(fig_pie, use_container_width=True)