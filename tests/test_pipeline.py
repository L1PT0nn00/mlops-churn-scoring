import pandas as pd
import pytest

from churn.data import TIME_COL, generate, load_csv, save_csv
from churn.pipeline import run_batch_score, run_bootstrap, run_retrain, run_rollback
from churn.registry import Registry
from churn.train import time_split
from churn.validation import DataValidationError


def test_time_split_has_no_future_in_train():
    train, test, cutoff = time_split(generate(n=3000, seed=5), test_share=0.2)
    assert train[TIME_COL].max() < test[TIME_COL].min()
    assert 0.15 < len(test) / (len(train) + len(test)) < 0.25


def test_first_model_goes_to_production(settings, data_path):
    decision = run_retrain(data_path, settings)
    registry = Registry(settings.model_dir)
    assert decision["promote"] is True
    assert registry.production_version() == decision["challenger_version"]
    assert registry.read_meta(decision["challenger_version"])["metrics"]["roc_auc"] > 0.7


def test_worse_model_is_rejected(settings, data_path, monkeypatch):
    """Всё зелёное, но модель хуже: ворота не пускают её в прод."""
    first = run_retrain(data_path, settings)
    weak = settings.model_copy(update={"train_iterations": 1, "min_auc": 0.0, "min_improvement": 0.01})
    second = run_retrain(data_path, weak)
    registry = Registry(settings.model_dir)
    assert second["promote"] is False
    assert registry.production_version() == first["challenger_version"]
    assert registry.read_meta(second["challenger_version"])["status"] == "rejected"


def test_bad_data_stops_pipeline(settings):
    df = generate(n=3000, seed=9)
    df["monthly_charges"] = None
    path = save_csv(df, settings.data_dir / "raw" / "broken.csv")
    with pytest.raises(DataValidationError):
        run_retrain(path, settings)
    assert Registry(settings.model_dir).list_versions() == []


def test_rollback_returns_previous_version(settings, data_path):
    first = run_retrain(data_path, settings)["challenger_version"]
    second_settings = settings.model_copy(update={"min_improvement": -1.0})  # пропустить любую
    second = run_retrain(data_path, second_settings)["challenger_version"]
    registry = Registry(settings.model_dir)
    assert registry.production_version() == second
    assert run_rollback(settings) == first
    assert registry.production_version() == first


def test_bootstrap_is_idempotent(settings):
    assert run_bootstrap(settings)["promote"] is True
    assert run_bootstrap(settings) is None
    assert len(Registry(settings.model_dir).list_versions()) == 1


def test_batch_score_writes_all_customers(settings, data_path):
    run_retrain(data_path, settings)
    out = settings.data_dir / "scores" / "s.csv"
    rows = run_batch_score(data_path, out, settings)
    scores = load_csv(out)
    assert rows == len(pd.read_csv(data_path)) == len(scores)
    assert scores["churn_probability"].between(0, 1).all()
