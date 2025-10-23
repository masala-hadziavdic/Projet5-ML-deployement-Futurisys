import pickle
import numpy as np

def main():
    # Charger le modèle
    with open('model_xgb.pkl', 'rb') as f:
        model = pickle.load(f)

    # Exemple de donnée dummy (adapter les features)
    X_test = np.array([[0.5, 1.2, 3.3, 4.0]])

    # Prédiction
    preds = model.predict(X_test)

    print("Prédiction :", preds)

    assert preds.shape[0] == X_test.shape[0], "Nombre de prédictions incorrect"

if __name__ == "__main__":
    main()
