# ========================================================
# DASHBOARD STREAMLIT RH – PRÉDICTION ATTRITION HUMANFORYOU
# Version FIX - compatible avec modèle XGBClassifier encodé
# ========================================================

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# --- Configuration page ---
st.set_page_config(
    page_title="HumanForYou - Prédiction Attrition",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS pour un design moderne ---
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .metric-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center; margin: 10px 0; }
    .metric-value { font-size: 2.5rem; font-weight: bold; color: #1e3a8a; margin: 10px 0; }
    .metric-label { font-size: 1rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }
    .section-header { color: #1e3a8a; font-size: 1.5rem; font-weight: 600; margin: 30px 0 20px 0; padding-bottom: 10px; border-bottom: 3px solid #3b82f6; }
    .chart-container { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 10px 0; }
    .streamlit-expanderHeader { background-color: #f1f5f9; border-radius: 8px; font-weight: 600; }
    .stButton>button { width: 100%; background-color: #1e3a8a; color: white; border-radius: 8px; padding: 12px 24px; font-weight: 600; border: none; box-shadow: 0 2px 4px rgba(0,0,0,0.1); transition: all 0.3s ease; }
    .stButton>button:hover { background-color: #3b82f6; box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: white; border-radius: 8px 8px 0 0; padding: 12px 24px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ========================================================
# 1) Chargement modèle (XGBClassifier)
# ========================================================
@st.cache_resource
def load_model():
    try:
        return joblib.load("best_attrition_model_pipeline.pkl")
    except FileNotFoundError:
        st.error("⚠️ Modèle introuvable : 'best_attrition_model_pipeline.pkl' (mets-le dans le même dossier).")
        return None

model = load_model()
if model is None:
    st.stop()

# ========================================================
# 2) Colonnes attendues par TON modèle (encodées)
# ========================================================
EXPECTED_COLS = [
    "Unnamed: 0",
    "Friday_Evening_Presence_Rate",
    "Overtime_Days_8.5h",
    "Avg_Daily_Hours",
    "TotalWorkingYears",
    "YearsWithCurrManager",
    "Overtime_Days_10h",
    "YearsAtCompany",
    "Early_Leave_Rate",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "BusinessTravel",
    "Age_Group",
    "JobRole_Human Resources",
    "JobRole_Laboratory Technician",
    "JobRole_Manager",
    "JobRole_Manufacturing Director",
    "JobRole_Research Director",
    "JobRole_Research Scientist",
    "JobRole_Sales Executive",
    "JobRole_Sales Representative",
    "Department_Research & Development",
    "Department_Sales",
]

JOB_ROLES = [
    "Human Resources",
    "Laboratory Technician",
    "Manager",
    "Manufacturing Director",
    "Research Director",
    "Research Scientist",
    "Sales Executive",
    "Sales Representative",
]

DEPARTMENTS = ["Research & Development", "Sales"]

BUSINESS_TRAVEL_MAP = {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}
AGE_GROUP_MAP = {"<30": 0, "30-40": 1, "40-50": 2, "50+": 3}

# ========================================================
# 3) Chargement data exploration (OPTIONNEL) - ton fichier existe: df_model.csv
# ========================================================
@st.cache_data
def load_data_optional():
    for fname in ["df_model_with_attrition.csv", "df_model.csv", "data_model.csv", "data_final.csv"]:
        try:
            df = pd.read_csv(fname)
            return df, fname
        except FileNotFoundError:
            continue
    return None, None

df_full, df_name = load_data_optional()

# --- En-tête principal ---
st.markdown("""
<div style='background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); padding: 40px; border-radius: 15px; margin-bottom: 30px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'>
    <h1 style='color: white; margin: 0; font-size: 2.5rem;'>🔍 HumanForYou - Dashboard RH</h1>
    <p style='color: #e0e7ff; margin: 10px 0 0 0; font-size: 1.1rem;'>Prédiction et Analyse du Risque d'Attrition</p>
</div>
""", unsafe_allow_html=True)

with st.container():
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("📊 **Taux historique** : ~16% d'attrition")
    with c2:
        st.info("🤖 **Modèle** : XGBoost optimisé")
    with c3:
        st.info("✅ **Éthique** : Version fair, sans variables sensibles")

st.markdown("---")

tab1, tab2 = st.tabs(["📊 Exploration Globale", "🔮 Prédiction Individuelle"])

# ========================================================
# TAB 1 : EXPLORATION (si data dispo)
# ========================================================
with tab1:
    if df_full is None:
        st.warning("📌 Exploration désactivée : aucun CSV trouvé (df_model.csv / data_model.csv / data_final.csv).")
    else:
        st.success(f"✅ Données chargées pour exploration : {df_name} ({df_full.shape[0]} lignes, {df_full.shape[1]} colonnes)")

        # Si la colonne Attrition existe en Yes/No
        if "Attrition" in df_full.columns:
            st.markdown("<div class='section-header'>📈 Indicateurs Clés</div>", unsafe_allow_html=True)

            # Cas Attrition 'Yes'/'No' OU 0/1
            if df_full["Attrition"].dtype == object:
                attrition_rate = (df_full["Attrition"] == "Yes").mean() * 100
                employees_left = (df_full["Attrition"] == "Yes").sum()
            else:
                attrition_rate = (df_full["Attrition"] == 1).mean() * 100
                employees_left = (df_full["Attrition"] == 1).sum()

            total_employees = len(df_full)
            avg_tenure = df_full["YearsAtCompany"].mean() if "YearsAtCompany" in df_full.columns else float("nan")

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Taux d'Attrition</div>
                    <div class='metric-value' style='color:#dc2626;'>{attrition_rate:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            with k2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Employés Partis</div>
                    <div class='metric-value' style='color:#ea580c;'>{employees_left}</div>
                </div>""", unsafe_allow_html=True)
            with k3:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Total Employés</div>
                    <div class='metric-value' style='color:#16a34a;'>{total_employees}</div>
                </div>""", unsafe_allow_html=True)
            with k4:
                val = "-" if pd.isna(avg_tenure) else f"{avg_tenure:.1f} ans"
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Ancienneté Moyenne</div>
                    <div class='metric-value' style='color:#2563eb;'>{val}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<div class='section-header'>📊 Analyses</div>", unsafe_allow_html=True)

            # Quelques graphes si colonnes présentes
            if {"Overtime_Days_10h", "YearsWithCurrManager"}.issubset(df_full.columns) and "Attrition" in df_full.columns:
                colA, colB = st.columns(2)
                with colA:
                    st.plotly_chart(px.box(df_full, x="Attrition", y="Overtime_Days_10h", title="Jours >10h vs Attrition"),
                                    use_container_width=True)
                with colB:
                    st.plotly_chart(px.box(df_full, x="Attrition", y="YearsWithCurrManager", title="Années avec manager vs Attrition"),
                                    use_container_width=True)
        else:
            st.info("ℹ️ La colonne 'Attrition' n’est pas dans ce CSV → j’affiche juste un aperçu.")
            st.dataframe(df_full.head(30))

# ========================================================
# TAB 2 : PRÉDICTION (corrigée pour colonnes encodées)
# ========================================================
with tab2:
    st.markdown("<div class='section-header'>🎯 Saisie des Informations Employé</div>", unsafe_allow_html=True)

    with st.expander("👤 Profil & Poste", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            job_role = st.selectbox("Poste", JOB_ROLES)
            department = st.selectbox("Département", DEPARTMENTS)
        with col2:
            age_group = st.selectbox("Tranche d'Âge", list(AGE_GROUP_MAP.keys()))
            business_travel = st.selectbox("Fréquence Déplacements", list(BUSINESS_TRAVEL_MAP.keys()))
        with col3:
            total_years = st.number_input("Années Expérience Totale", 0, 40, 10)
            years_company = st.number_input("Ancienneté Entreprise (années)", 0, 40, 5)

    with st.expander("⏱️ Temps de Travail & Horaires", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_hours = st.slider("Heures Moyennes Journalières", 5.0, 12.0, 8.0, 0.1)
            overtime_8_5 = st.number_input("Jours Overtime >8.5h", 0, 250, 50)
        with col2:
            overtime_10 = st.number_input("Jours Overtime >10h", 0, 200, 10)
            early_leave = st.slider("Taux Départ Anticipé (0 à 1)", 0.0, 1.0, 0.10, 0.01)
        with col3:
            friday_presence = st.slider("Taux Présence Vendredi Soir (0 à 1)", 0.0, 1.0, 0.20, 0.01)

    with st.expander("😊 Satisfaction & Management", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            job_sat = st.select_slider("Satisfaction Travail", [1, 2, 3, 4], value=3)
        with col2:
            env_sat = st.select_slider("Satisfaction Environnement", [1, 2, 3, 4], value=3)
        with col3:
            years_manager = st.number_input("Années avec Manager Actuel", 0, 30, 4)

    st.markdown("<br>", unsafe_allow_html=True)
    colL, colC, colR = st.columns([1, 2, 1])
    with colC:
        predict_button = st.button("🔮 PRÉDIRE LE RISQUE D'ATTRITION", use_container_width=True)
           
    if predict_button:
        row = {c: 0 for c in EXPECTED_COLS}

        # Numériques
        row["Unnamed: 0"] = 0
        row["Friday_Evening_Presence_Rate"] = friday_presence
        row["Overtime_Days_8.5h"] = overtime_8_5
        row["Avg_Daily_Hours"] = avg_hours
        row["TotalWorkingYears"] = total_years
        row["YearsWithCurrManager"] = years_manager
        row["Overtime_Days_10h"] = overtime_10
        row["YearsAtCompany"] = years_company
        row["Early_Leave_Rate"] = early_leave
        row["JobSatisfaction"] = job_sat
        row["EnvironmentSatisfaction"] = env_sat

        # Catégories encodées numériquement
        row["BusinessTravel"] = BUSINESS_TRAVEL_MAP[business_travel]
        row["Age_Group"] = AGE_GROUP_MAP[age_group]

        # One-hot JobRole
        role_col = f"JobRole_{job_role}"
        if role_col in row:
            row[role_col] = 1

        # One-hot Department
        if department == "Sales":
            row["Department_Sales"] = 1
        else:
            row["Department_Research & Development"] = 1

        input_df = pd.DataFrame([row], columns=EXPECTED_COLS)
        proba = float(model.predict_proba(input_df)[0][1])
        SEUIL_RISQUE = 0.35   # 👈 seuil RH réaliste
        prediction = "RISQUE" if proba >= SEUIL_RISQUE else "PAS RISQUE"


        # ✅ Affichage
        st.markdown("<div class='section-header'>📊 Résultat de la Prédiction</div>", unsafe_allow_html=True)

        if prediction == "RISQUE":
            risk_color = "#ef4444"
            emoji = "🚨"
            bg = "#fee2e2"
        else:
            risk_color = "#10b981"
            emoji = "✅"
            bg = "#d1fae5"

        st.markdown(f"""
        <div style='background-color:{bg}; padding: 60px; border-radius: 15px; 
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15); text-align: center; margin: 20px 0;'>
            <div style='font-size: 6rem; margin-bottom: 10px;'>{emoji}</div>
            <div style='font-size: 3rem; font-weight: bold; color: {risk_color};'>
                {prediction}
            </div>
            <div style='font-size: 1.1rem; color: #64748b; margin-top: 10px;'>
                Prédiction basée sur XGBoost
            </div>
        </div>
        """, unsafe_allow_html=True)

        if prediction == "RISQUE":
            st.error("""
        🚨 **Action Urgente Requise**
        - Entretien individuel avec le manager
        - Réduction des heures supplémentaires
        - Plan de développement / mobilité interne
        """)
        else:
            st.success("""
        ✅ **Situation Favorable**
        - Profil stable
        - Continuer le suivi et les échanges réguliers
        """)
