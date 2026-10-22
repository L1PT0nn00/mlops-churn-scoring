"""Интеграция с MLflow: журнал экспериментов и регистрация прод-модели.

Если MLFLOW_TRACKING_URI не задан или mlflow не установлен, функции
ничего не делают: проект работает и без трекинг-сервера.
"""

import logging
from typing import Optional

from churn.config import Settings

log = logging.getLogger(__name__)


def _mlflow(settings: Settings):
    if not settings.mlflow_tracking_uri:
        return None
    try:
        import mlflow
        import mlflow.catboost  # noqa: F401
    except ImportError:
        log.warning("MLFLOW_TRACKING_URI задан, но mlflow не установлен, трекинг пропущен")
        return None
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return mlflow


def log_training(settings: Settings, version: str, meta: dict, model) -> Optional[str]:
    mlflow = _mlflow(settings)
    if mlflow is None:
        return None
    try:
        mlflow.set_experiment(settings.mlflow_experiment)
        with mlflow.start_run(run_name=version) as run:
            mlflow.log_params(meta["params"])
            mlflow.log_metrics({k: float(v) for k, v in meta["metrics"].items()})
            mlflow.set_tags(
                {
                    "churn_version": version,
                    "train_period": " .. ".join(meta["train_period"]),
                    "test_period": " .. ".join(meta["test_period"]),
                }
            )
            mlflow.catboost.log_model(model, artifact_path="model")
            return run.info.run_id
    except Exception:
        log.exception("не удалось записать запуск в MLflow, продолжаем без него")
        return None


def register_production(settings: Settings, version: str, run_id: Optional[str]) -> None:
    mlflow = _mlflow(settings)
    if mlflow is None or not run_id:
        return
    try:
        from mlflow.tracking import MlflowClient

        mv = mlflow.register_model(f"runs:/{run_id}/model", settings.mlflow_model_name)
        client = MlflowClient()
        client.set_registered_model_alias(settings.mlflow_model_name, "champion", mv.version)
        client.set_model_version_tag(settings.mlflow_model_name, mv.version, "churn_version", version)
        log.info("registered in MLflow", extra={"version": version, "mlflow_version": mv.version})
    except Exception:
        log.exception("не удалось зарегистрировать модель в MLflow")
