import pickle
import pandas as pd

def main():
    # Charger le modèle
    with open('model_xgb.pkl', 'rb') as f:
        model = pickle.load(f)

    # Exemple de données de test (adapter les colonnes selon ton entraînement)
    X_test = pd.DataFrame([{
        "age": 30,
        "genre": "H",
        "revenu_mensuel": 3200,
        "statut_marital": "Célibataire",
        "departement": "Marketing",
        "nombre_experiences_precedentes": 2,
        "annees_dans_l_entreprise": 5,
        "nombre_participation_pee": 1,
        "nb_formations_suivies": 3,
        "distance_domicile_travail": 15,
        "niveau_education": 3,
        "domaine_etude": "Transformation Digitale",
        "frequence_deplacement": "Occasionnel",
        "annees_depuis_la_derniere_promotion": 3,
        "satisfaction_employee_environnement": 7,
        "note_evaluation_precedente": 6,
        "satisfaction_employee_nature_travail": 8,
        "satisfaction_employee_equipe": 7,
        "satisfaction_employee_equilibre_pro_perso": 6,
        "note_evaluation_actuelle": 7,
        "augementation_salaire_precedente": 12
    }])

    # Prédiction
    preds = model.predict(X_test)
    print("✅ Prédiction :", preds)

    # Vérification simple
    assert preds.shape[0] == X_test.shape[0], 

if __name__ == "__main__":
    main()

