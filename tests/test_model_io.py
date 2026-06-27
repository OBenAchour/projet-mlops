from model_io import save_model, load_model
from sklearn.ensemble import RandomForestClassifier


def test_save_load_model(tmp_path):

    model = RandomForestClassifier(n_estimators=10, random_state=42)

    filepath = tmp_path / "model.joblib"

    save_model(model, filepath)

    loaded_model = load_model(filepath)

    assert loaded_model is not None

    assert hasattr(loaded_model, "predict")
