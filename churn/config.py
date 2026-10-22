"""Настройки из переменных окружения (префикс CHURN_), без хардкода в коде."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CHURN_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    data_dir: Path = Path("data")
    model_dir: Path = Path("models")

    # Проверки данных
    min_rows: int = 1000
    max_null_share: float = 0.05

    # Обучение и ворота качества
    test_share: float = 0.2
    train_iterations: int = 300
    threshold: float = 0.5
    min_auc: float = 0.70
    min_improvement: float = 0.0

    # Сервис
    model_reload_seconds: int = 30
    drift_window: int = 500
    drift_every: int = 50
    log_level: str = "INFO"

    # MLflow (необязателен: без адреса трекинг просто выключен)
    mlflow_tracking_uri: Optional[str] = Field(default=None, validation_alias="MLFLOW_TRACKING_URI")
    mlflow_experiment: str = "churn"
    mlflow_model_name: str = "churn-model"

    @property
    def default_data_path(self) -> Path:
        return self.data_dir / "raw" / "customers.csv"


def get_settings() -> Settings:
    return Settings()
