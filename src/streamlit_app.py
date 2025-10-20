import streamlit as st
import numpy as np
import pickle
import pandas as pd


with open("src/model_xgb.pkl", "rb") as f:
    model = pickle.load(f)


st.title("Prédiction de l'employé avec XGBoost")

# Inputs utilisateur (ajuste types / valeurs selon tes données)
age = st.number_input("Âge", min_value=18, max_value=100, value=30)
genre = st.selectbox("Genre", options=["H", "F"])  
revenu_mensuel = st.number_input("Revenu mensuel", min_value=0)
statut_marital = st.selectbox("Statut marital", options=["Célibataire", "Marié", "Divorcé", "Autre"])
departement = st.text_input("Département")  
nombre_experiences_precedentes = st.number_input("Nombre d'expériences précédentes", min_value=0)
annees_dans_l_entreprise = st.number_input("Années dans l'entreprise", min_value=0)
nombre_participation_pee = st.number_input("Nombre de participations PEE", min_value=0)
nb_formations_suivies = st.number_input("Nombre de formations suivies", min_value=0)
distance_domicile_travail = st.number_input("Distance domicile-travail (km)", min_value=0)
niveau_education = st.selectbox("Niveau d'éducation (1-5)", [1, 2, 3, 4, 5], index=0)
domaine_etude = st.selectbox("Domaine d'étude", options=["Infra & Cloud", "Transformation Digitale", "Marketing", "Entrepreunariat", "Autre"])
frequence_deplacement = st.selectbox("Fréquence déplacement", options=["Occasionnel", "Frequent", "Aucun"])
annees_depuis_la_derniere_promotion = st.number_input("Années depuis la dernière promotion", min_value=0)
satisfaction_employee_environnement = st.slider("Satisfaction environnement (1-10)", 1, 10, 1)
note_evaluation_precedente = st.slider("Note évaluation précédente (1-10)", 1, 10, 1)
satisfaction_employee_nature_travail = st.slider("Satisfaction nature travail (1-10)", 1, 10, 1)
satisfaction_employee_equipe = st.slider("Satisfaction équipe (1-10)", 1, 10, 1)
satisfaction_employee_equilibre_pro_perso = st.slider("Satisfaction équilibre pro/perso (1-10)", 1, 10, 1)
note_evaluation_actuelle = st.slider("Note évaluation actuelle (1-10)", 1, 10, 1)
augementation_salaire_precedente = st.selectbox("Augmentation salaire précédente (%)", list(range(11, 26)), index=0)


# Bouton prédiction
if st.button("Prédire"):
    genre_num = 1 if genre == "H" else 0
    statut_marital_map = {"Célibataire": 0, "Marié": 1, "Divorcé": 2, "Autre": 3}
    statut_marital_num = statut_marital_map.get(statut_marital, 3)

    niveau_education_num = niveau_education
    frequence_deplacement_map = {"Aucun": 0, "Occasionnel": 1, "Frequent": 2}
    frequence_deplacement_num = frequence_deplacement_map.get(frequence_deplacement, 0)

    departement_num = 0
    domaine_etude_num = 0
    augementation_num = augementation_salaire_precedente

    features = [
        'age',
        'genre',
        'revenu_mensuel',
        'statut_marital',
        'departement',
        'nombre_experiences_precedentes',
        'annees_dans_l_entreprise',
        'nombre_participation_pee',
        'nb_formations_suivies',
        'distance_domicile_travail',
        'niveau_education',
        'domaine_etude',
        'frequence_deplacement',
        'annees_depuis_la_derniere_promotion',
        'satisfaction_employee_environnement',
        'note_evaluation_precedente',
        'satisfaction_employee_nature_travail',
        'satisfaction_employee_equipe',
        'satisfaction_employee_equilibre_pro_perso',
        'note_evaluation_actuelle',
        'augementation_salaire_precedente'
]

    input_values = [
        age,
        genre,
        revenu_mensuel,
        statut_marital,
        departement,
        nombre_experiences_precedentes,
        annees_dans_l_entreprise,
        nombre_participation_pee,
        nb_formations_suivies,
        distance_domicile_travail,
        niveau_education,
        domaine_etude,
        frequence_deplacement,
        annees_depuis_la_derniere_promotion,
        satisfaction_employee_environnement,
        note_evaluation_precedente,
        satisfaction_employee_nature_travail,
        satisfaction_employee_equipe,
        satisfaction_employee_equilibre_pro_perso,
        note_evaluation_actuelle,
        augementation_salaire_precedente
]

    X = pd.DataFrame([input_values], columns=features)

    # --- bloc prédiction ---
    pred = model.predict(X)
    try:
        proba = model.predict_proba(X)
    except AttributeError:
        proba = None

    st.write(f"**Prédiction :** {pred[0]}")
    if proba is not None:
        st.write(f"**Probabilités :** {proba[0]}")