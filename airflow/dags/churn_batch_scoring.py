"""Ночной батч-скоринг всех клиентов: результат забирает CRM.

Для оттока это основной режим: клиент не уходит за миллисекунды, поэтому
онлайн-API не нужен, достаточно раз в сутки пересчитать всех.
"""

from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.sensors.python import PythonSensor


def _has_production_model() -> bool:
    from churn.config import get_settings
    from churn.registry import Registry

    return Registry(get_settings().model_dir).production_version() is not None


@dag(
    dag_id="churn_batch_scoring",
    schedule="0 5 * * *",  # после переобучения в 03:00
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "mlops", "retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["churn", "scoring"],
)
def churn_batch_scoring():
    wait_for_model = PythonSensor(
        task_id="wait_for_production_model",
        python_callable=_has_production_model,
        poke_interval=60,
        timeout=3600,
        mode="reschedule",  # не держит слот воркера, пока ждёт
    )

    @task
    def score(ds=None) -> dict:
        from churn.config import get_settings
        from churn.pipeline import run_batch_score

        settings = get_settings()
        source = settings.data_dir / "raw" / f"customers_{ds}.csv"
        if not source.exists():
            source = settings.default_data_path
        out = settings.data_dir / "scores" / f"churn_scores_{ds}.csv"
        # В проде: запись в таблицу Postgres за дату (DELETE партиции + INSERT), чтобы перезапуск не задвоил
        return {"rows": run_batch_score(source, out, settings), "out": str(out)}

    wait_for_model >> score()


churn_batch_scoring()
