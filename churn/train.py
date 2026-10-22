"""Обучение challenger-модели со сплитом по времени."""

import logging
from datetime import datetime, timezone
from typing import Dict, Tuple

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from churn import tracking
from churn.config import Settings
from churn.data import TARGET, TIME_COL
from churn.drift import build_reference
from churn.features import CAT_FEATURES, FEATURES, NUMERIC_FEATURES, build_features
from churn.registry import Registry, write_json

log = logging.getLogger(__name__)


def time_split(df: pd.DataFrame, test_share: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Учимся на прошлом, проверяем на более позднем периоде.

    Случайный сплит на данных со временем даёт утечку: модель видит
    будущее относительно теста, и метрики получаются завышенными.
    """
    cutoff = df[TIME_COL].quantile(1 - test_share)
    train = df[df[TIME_COL] < cutoff]
    test = df[df[TIME_COL] >= cutoff]
    if train.empty or test.empty:
        raise ValueError("сплит по времени дал пустую часть, проверь даты")
    return train, test, cutoff


def fit(train_df: pd.DataFrame, iterations: int = 300, seed: int = 42) -> CatBoostClassifier:
    model = CatBoostClassifier(
        iterations=iterations,
        depth=6,
        learning_rate=0.08,
        loss_function="Logloss",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(build_features(train_df), train_df[TARGET], cat_features=CAT_FEATURES)
    return model


def evaluate(model: CatBoostClassifier, df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, float]:
    y = df[TARGET].to_numpy()
    proba = model.predict_proba(build_features(df))[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "pr_auc": round(float(average_precision_score(y, proba)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "churn_rate": round(float(y.mean()), 4),
        "rows": int(len(df)),
    }


def train_challenger(df: pd.DataFrame, registry: Registry, settings: Settings) -> str:
    train_df, test_df, cutoff = time_split(df, settings.test_share)
    log.info("training", extra={"rows_train": len(train_df), "rows_test": len(test_df), "cutoff": str(cutoff.date())})

    model = fit(train_df, iterations=settings.train_iterations)
    metrics = evaluate(model, test_df, settings.threshold)

    version = registry.new_version()
    d = registry.version_dir(version)
    model.save_model(str(d / "model.cbm"))
    test_df.to_csv(d / "holdout.csv", index=False)
    write_json(d / "reference.json", build_reference(build_features(train_df), NUMERIC_FEATURES, CAT_FEATURES))

    meta = {
        "version": version,
        "status": "challenger",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_period": [str(train_df[TIME_COL].min().date()), str(train_df[TIME_COL].max().date())],
        "test_period": [str(test_df[TIME_COL].min().date()), str(test_df[TIME_COL].max().date())],
        "features": FEATURES,
        "cat_features": CAT_FEATURES,
        "threshold": settings.threshold,
        "params": {"iterations": settings.train_iterations, "depth": 6, "learning_rate": 0.08},
        "metrics": metrics,
    }
    run_id = tracking.log_training(settings, version, meta, model)
    if run_id:
        meta["mlflow_run_id"] = run_id
    write_json(d / "meta.json", meta)
    log.info("challenger trained", extra={"version": version, **metrics})
    return version
