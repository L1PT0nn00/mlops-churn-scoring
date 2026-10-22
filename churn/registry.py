"""Файловый реестр моделей: версии, прод-указатель, история, откат.

Это лёгкая замена MLflow Model Registry для локального запуска. Когда
задан MLFLOW_TRACKING_URI, прод-версия дополнительно регистрируется
в MLflow с алиасом champion (см. churn/tracking.py).

Структура:
    models/versions/<version>/model.cbm      модель
    models/versions/<version>/meta.json      метрики, периоды, параметры
    models/versions/<version>/reference.json распределения для дрейфа
    models/versions/<version>/holdout.csv    отложенный период для сравнения
    models/production.json                   какая версия сейчас в проде
    models/history.jsonl                     журнал выкаток и откатов
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from churn.features import build_features


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    os.replace(tmp, path)  # атомарная замена: сервис не прочитает полфайла


@dataclass
class LoadedModel:
    version: str
    model: CatBoostClassifier
    meta: dict
    reference: dict

    @property
    def threshold(self) -> float:
        return float(self.meta.get("threshold", 0.5))

    def predict_proba(self, raw: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(build_features(raw))[:, 1]


class Registry:
    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)
        self.versions_dir = self.root / "versions"
        self.production_file = self.root / "production.json"
        self.history_file = self.root / "history.jsonl"

    def new_version(self) -> str:
        base = "v" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        version, i = base, 1
        while self.version_dir(version).exists():
            i += 1
            version = f"{base}-{i}"
        self.version_dir(version).mkdir(parents=True)
        return version

    def version_dir(self, version: str) -> Path:
        return self.versions_dir / version

    def list_versions(self) -> List[str]:
        if not self.versions_dir.exists():
            return []
        return sorted(p.name for p in self.versions_dir.iterdir() if (p / "model.cbm").exists())

    def read_meta(self, version: str) -> dict:
        return json.loads((self.version_dir(version) / "meta.json").read_text())

    def update_meta(self, version: str, **fields) -> dict:
        meta = self.read_meta(version)
        meta.update(fields)
        write_json(self.version_dir(version) / "meta.json", meta)
        return meta

    def production_version(self) -> Optional[str]:
        if not self.production_file.exists():
            return None
        return json.loads(self.production_file.read_text()).get("version")

    def promote(self, version: str, reason: str = "") -> None:
        if version not in self.list_versions():
            raise ValueError(f"нет такой версии: {version}")
        previous = self.production_version()
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(self.production_file, {"version": version, "promoted_at": _now(), "reason": reason})
        self._log({"event": "promote", "version": version, "previous": previous, "reason": reason})
        self.update_meta(version, status="production")
        if previous and previous != version:
            self.update_meta(previous, status="archived")

    def rollback(self) -> str:
        current = self.production_version()
        if current is None:
            raise RuntimeError("в проде нет модели, откатывать нечего")
        previous = None
        for event in self._history():
            if event.get("event") == "promote" and event.get("version") == current:
                previous = event.get("previous")
        if not previous:
            raise RuntimeError(f"у {current} нет предыдущей версии")
        self.promote(previous, reason=f"rollback from {current}")
        return previous

    def load(self, version: str) -> LoadedModel:
        d = self.version_dir(version)
        model = CatBoostClassifier()
        model.load_model(str(d / "model.cbm"))
        return LoadedModel(
            version=version,
            model=model,
            meta=json.loads((d / "meta.json").read_text()),
            reference=json.loads((d / "reference.json").read_text()),
        )

    def _log(self, event: dict) -> None:
        event = {"at": _now(), **event}
        with self.history_file.open("a") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _history(self) -> List[dict]:
        if not self.history_file.exists():
            return []
        return [json.loads(line) for line in self.history_file.read_text().splitlines() if line.strip()]
