"""Ночное переобучение с воротами качества.

    extract -> validate_data -> train_challenger -> quality_gate -> promote
                                                                \-> keep_champion

Зелёная задача в Airflow значит только «функция не упала». Поэтому
плохие данные и слабая модель отсекаются явно: validate_data падает на
кривых данных, quality_gate не пускает в прод модель хуже текущей.

Тяжёлые импорты (pandas, catboost) внутри задач: файл DAG парсится
планировщиком каждые ~30 секунд и должен импортироваться быстро.
"""

import logging
import os
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException

log = logging.getLogger(__name__)


def notify_failure(context) -> None:
    ti = context["task_instance"]
    # В реальном контуре здесь уведомление в Telegram/почту дежурному
    log.error("task failed: dag=%s task=%s try=%s", ti.dag_id, ti.task_id, ti.try_number)


DEFAULT_ARGS = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": notify_failure,
}


@dag(
    dag_id="churn_retrain",
    schedule="0 3 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["churn", "training"],
)
def churn_retrain():
    @task
    def extract(ds=None) -> str:
        """Снимок клиентов за дату. В проде это SQL из DWH, здесь генератор.

        Идемпотентно: файл за дату перезаписывается, повторный запуск не плодит дублей.
        """
        import pandas as pd

        from churn.config import get_settings
        from churn.pipeline import run_generate

        settings = get_settings()
        out = settings.data_dir / "raw" / f"customers_{ds}.csv"
        start = (pd.Timestamp(ds) - pd.Timedelta(days=730)).date().isoformat()
        shift = float(os.environ.get("CHURN_SIM_SHIFT", "0"))  # имитация дрейфа для демо
        run_generate(out, n=20_000, start=start, end=ds, seed=int(ds.replace("-", "")) % 10_000, shift=shift)
        return str(out)

    @task
    def validate_data(path: str) -> dict:
        from churn.config import get_settings
        from churn.pipeline import run_validate
        from churn.validation import DataValidationError

        try:
            return run_validate(path, get_settings())
        except DataValidationError as e:
            # Без ретраев: кривые данные сами не починятся, нужен человек
            raise AirflowFailException(f"данные не прошли проверку: {e}") from e

    @task
    def train_challenger(path: str) -> str:
        from churn.config import get_settings
        from churn.pipeline import run_train

        return run_train(path, get_settings())

    @task.branch
    def quality_gate(version: str) -> str:
        from churn.config import get_settings
        from churn.pipeline import run_decide

        decision = run_decide(version, get_settings())
        return "promote" if decision["promote"] else "keep_champion"

    @task
    def promote(version: str) -> None:
        from churn.config import get_settings
        from churn.pipeline import run_promote

        run_promote(version, get_settings())

    @task
    def keep_champion(version: str) -> None:
        from churn.config import get_settings
        from churn.pipeline import run_reject

        run_reject(version, get_settings(), reason="не прошла ворота качества")
        # В проде: уведомление команде, что модель обучилась, но в прод не пошла

    path = extract()
    checked = validate_data(path)
    version = train_challenger(path)
    checked >> version
    quality_gate(version) >> [promote(version), keep_champion(version)]


churn_retrain()
