import joblib


def save_model(model, filepath="classifier.joblib"):
    with open(filepath, "wb") as f:
        joblib.dump(model, f)
    print(f"💾 Modèle sauvegardé dans {filepath}")


def load_model(filepath="classifier.joblib"):
    model = joblib.load(filepath)
    print(f"📂 Modèle chargé depuis {filepath}")
    return model
