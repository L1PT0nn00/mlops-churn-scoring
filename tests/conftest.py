import pytest

from churn.config import Settings
from churn.data import generate, save_csv


@pytest.fixture
def settings(tmp_path, monkeypatch) -> Settings:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    return Settings(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        train_iterations=80,
        model_reload_seconds=3600,
        drift_window=200,
        drift_every=50,
    )


@pytest.fixture
def data_path(settings):
    return save_csv(generate(n=5000, seed=7), settings.default_data_path)
