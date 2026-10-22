"""Шаги пайплайна. Одни и те же функции вызывают CLI, Airflow и CronJob в Kubernetes.

    python -m churn.pipeline generate      сгенерировать данные
    python -m churn.pipeline validate      проверить данные
    python -m churn.pipeline retrain       проверка -> обучение -> ворота -> выкатка
    python -m churn.pipeline rollback      вернуть предыдущую прод-модель
    python -m churn.pipeline batch-score   ночной скоринг всех клиентов
    python -m churn.pipeline bootstrap     данные + первая модель, если их ещё нет
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from churn import tracking
from churn.config import Settings, get_settings
from churn.data import ID_COL, generate, load_csv, save_csv
from churn.gates import decide
from churn.logging_utils import setup_logging
from churn.registry import Registry
from churn.train import train_challenger
from churn.validation import DataValidationError, validate

log = logging.getLogger("churn.pipeline")


def run_generate(out: Path, n: int = 20_000, start: str = "2024-01-01", end: str = "2025-12-31",
                 seed: int = 42, shift: float = 0.0) -> Path:
    path = save_csv(generate(n=n, start=start, end=end, seed=seed, shift=shift), out)
    log.info("data generated", extra={"path": str(path), "rows": n, "shift": shift})
    return path


def run_validate(path: Path, settings: Settings, require_target: bool = True) -> dict:
    report = validate(load_csv(path), settings.min_rows, settings.max_null_share, require_target=require_target)
    for warning in report.warnings:
        log.warning(warning)
    report.raise_if_failed()
    log.info("data valid", extra={"path": str(path), **report.stats})
    return report.stats


def run_train(path: Path, settings: Settings) -> str:
    return train_challenger(load_csv(path), Registry(settings.model_dir), settings)


def run_decide(version: str, settings: Settings) -> dict:
    registry = Registry(settings.model_dir)
    decision = decide(registry, version, settings.min_auc, settings.min_improvement)
    registry.update_meta(version, gate=decision.to_dict())
    log.info("quality gate", extra=decision.to_dict())
    return decision.to_dict()


def run_promote(version: str, settings: Settings, reason: str = "passed quality gate") -> None:
    registry = Registry(settings.model_dir)
    registry.promote(version, reason=reason)
    tracking.register_production(settings, version, registry.read_meta(version).get("mlflow_run_id"))
    log.info("promoted", extra={"version": version})


def run_reject(version: str, settings: Settings, reason: str) -> None:
    Registry(settings.model_dir).update_meta(version, status="rejected")
    log.warning("challenger rejected, champion stays", extra={"version": version, "reason": reason})


def run_retrain(path: Path, settings: Settings) -> dict:
    run_validate(path, settings)
    version = run_train(path, settings)
    decision = run_decide(version, settings)
    if decision["promote"]:
        run_promote(version, settings, reason=decision["reason"])
    else:
        run_reject(version, settings, decision["reason"])
    return decision


def run_rollback(settings: Settings) -> str:
    version = Registry(settings.model_dir).rollback()
    log.warning("rolled back", extra={"version": version})
    return version


def run_batch_score(in_path: Path, out_path: Path, settings: Settings) -> int:
    """В реальном контуре результат пишется в таблицу Postgres за дату (идемпотентно)."""
    registry = Registry(settings.model_dir)
    version = registry.production_version()
    if version is None:
        raise RuntimeError("в проде нет модели")
    df = load_csv(in_path)
    validate(df, min_rows=1, max_null_share=settings.max_null_share, require_target=False).raise_if_failed()
    model = registry.load(version)
    proba = model.predict_proba(df)
    scores = df[[ID_COL]].copy()
    scores["churn_probability"] = proba.round(4)
    scores["will_churn"] = (proba >= model.threshold).astype(int)
    scores["model_version"] = version
    scores["scored_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_csv(scores, out_path)
    log.info("batch scored", extra={"rows": len(scores), "out": str(out_path), "version": version})
    return len(scores)


def run_bootstrap(settings: Settings) -> Optional[dict]:
    path = settings.default_data_path
    if not path.exists():
        run_generate(path)
    if Registry(settings.model_dir).production_version():
        log.info("production model exists, bootstrap skipped")
        return None
    return run_retrain(path, settings)


def main(argv=None) -> int:
    settings = get_settings()
    setup_logging(settings.log_level)
    p = argparse.ArgumentParser(prog="churn.pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--out", type=Path, default=settings.default_data_path)
    g.add_argument("--n", type=int, default=20_000)
    g.add_argument("--start", default="2024-01-01")
    g.add_argument("--end", default="2025-12-31")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--shift", type=float, default=0.0, help="сила дрейфа, 0 = без дрейфа")

    for name in ("validate", "train", "retrain"):
        sp = sub.add_parser(name)
        sp.add_argument("--data", type=Path, default=settings.default_data_path)
    sub.add_parser("rollback")
    sub.add_parser("bootstrap")
    b = sub.add_parser("batch-score")
    b.add_argument("--data", type=Path, default=settings.default_data_path)
    b.add_argument("--out", type=Path, default=settings.data_dir / "scores" / "churn_scores.csv")

    args = p.parse_args(argv)
    try:
        if args.cmd == "generate":
            result = {"path": str(run_generate(args.out, args.n, args.start, args.end, args.seed, args.shift))}
        elif args.cmd == "validate":
            result = run_validate(args.data, settings)
        elif args.cmd == "train":
            result = {"version": run_train(args.data, settings)}
        elif args.cmd == "retrain":
            result = run_retrain(args.data, settings)
        elif args.cmd == "rollback":
            result = {"version": run_rollback(settings)}
        elif args.cmd == "batch-score":
            result = {"rows": run_batch_score(args.data, args.out, settings), "out": str(args.out)}
        else:
            result = run_bootstrap(settings) or {"status": "skipped"}
    except DataValidationError as e:
        log.error("data validation failed", extra={"errors": e.report.errors})
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
