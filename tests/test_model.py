import pickle
import pandas as pd

def main():
    with open('model_xgb.pkl', 'rb') as f:
        model = pickle.load(f)

    print("✅ Modèle chargé avec succès !")

    # Récupérer les noms attendus (si pipeline scikit-learn)
    try:
        expected_features = model.feature_names_in_
        print("🧩 Le modèle attend ces features :", list(expected_features))
    except AttributeError:
        print("⚠️ Impossible de détecter automatiquement les noms de features.")
        expected_features = [
            'age', 'genre', 'revenu_mensuel', 'statut_marital', 'departement',
            'nombre_experiences_precedentes', 'annees_dans_l_entreprise',
            'nombre_participation_pee', 'nb_formations_suivies',
            'distance_domicile_travail', 'niveau_education', 'domaine_etude',
            'frequence_deplacement', 'annees_depuis_la_derniere_promotion',
            'satisfaction_employee_environnement', 'note_evaluation_precedente',
            'satisfaction_employee_nature_travail', 'satisfaction_employee_equipe',
            'satisfaction_employee_equilibre_pro_perso', 'note_evaluation_actuelle',
            'augementation_salaire_precedente',
        ]

    # Exemple d'entrée (doit respecter l’ordre exact des features)
    X_test = pd.DataFrame([{
        'age': 35,
        'genre': 'F',
        'revenu_mensuel': 3500,
        'statut_marital': 'Célibataire',
        'departement': 'IT',
        'nombre_experiences_precedentes': 3,
        'annees_dans_l_entreprise': 5,
        'nombre_participation_pee': 2,
        'nb_formations_suivies': 1,
        'distance_domicile_travail': 15.0,
        'niveau_education': 'Master',
        'domaine_etude': 'Informatique',
        'frequence_deplacement': 'Rarement',
        'annees_depuis_la_derniere_promotion': 2,
        'satisfaction_employee_environnement': 4,
        'note_evaluation_precedente': 3.8,
        'satisfaction_employee_nature_travail': 4,
        'satisfaction_employee_equipe': 5,
        'satisfaction_employee_equilibre_pro_perso': 4,
        'note_evaluation_actuelle': 4.2,
        'augementation_salaire_precedente': 0.05,
    }])[expected_features]

<<<<<<< HEAD
      - name: Installer les dépendances
        run: |
          python -m pip install --upgrade pip
          pip install -r ../../requirements.txt   # si requirements.txt est à la racine

      - name: Lancer le test modèle
        run: python .github/workflows/test_model.py

=======
    print("🧠 Colonnes envoyées au modèle :", list(X_test.columns))
    preds = model.predict(X_test)
    print("✅ Prédiction :", preds)

if __name__ == "__main__":
    main()
>>>>>>> 70baeb5 (Save workflow updates before rebase)
