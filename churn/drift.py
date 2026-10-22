"""Мониторинг дрейфа: PSI между данными в проде и обучающей выборкой.

PSI (Population Stability Index): < 0.1 стабильно, 0.1-0.2 стоит
присмотреться, > 0.2 распределение заметно уехало.
"""

import threading
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

EPS = 1e-4


def _key(value) -> str:
    try:
        return format(float(value), "g")
    except (TypeError, ValueError):
        return str(value)


def build_reference(features: pd.DataFrame, numeric: List[str], categorical: List[str], bins: int = 10) -> dict:
    ref: dict = {"numeric": {}, "discrete": {}}
    for col in numeric:
        values = features[col].astype(float).to_numpy()
        if len(np.unique(values)) <= bins:
            ref["discrete"][col] = _frequencies(values)
            continue
        inner = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1))[1:-1])
        ref["numeric"][col] = {"edges": inner.tolist(), "expected": _binned(values, inner).tolist()}
    for col in categorical:
        ref["discrete"][col] = _frequencies(features[col].to_numpy())
    return ref


def _binned(values: np.ndarray, inner_edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(inner_edges, values, side="right")
    counts = np.bincount(idx, minlength=len(inner_edges) + 1).astype(float)
    return counts / max(counts.sum(), 1.0)


def _frequencies(values) -> Dict[str, float]:
    keys = pd.Series([_key(v) for v in values])
    return {k: float(v) for k, v in keys.value_counts(normalize=True).items()}


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def compute_psi(reference: dict, features: pd.DataFrame) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for col, spec in reference.get("numeric", {}).items():
        actual = _binned(features[col].astype(float).to_numpy(), np.asarray(spec["edges"]))
        result[col] = round(psi(spec["expected"], actual), 4)
    for col, expected in reference.get("discrete", {}).items():
        actual = _frequencies(features[col].to_numpy())
        keys = sorted(set(expected) | set(actual))
        result[col] = round(psi([expected.get(k, 0.0) for k in keys], [actual.get(k, 0.0) for k in keys]), 4)
    return result


class DriftMonitor:
    """Скользящее окно последних запросов; PSI пересчитывается раз в `every` строк."""

    def __init__(self, reference: dict, window: int = 500, every: int = 50, min_rows: int = 100):
        self.reference = reference
        self.rows: deque = deque(maxlen=window)
        self.every = max(every, 1)
        self.min_rows = min(min_rows, window)
        self._since = 0
        self._lock = threading.Lock()

    def update(self, features: pd.DataFrame) -> Optional[Dict[str, float]]:
        with self._lock:
            self.rows.extend(features.to_dict("records"))
            self._since += len(features)
            if self._since < self.every or len(self.rows) < self.min_rows:
                return None
            self._since = 0
            window = pd.DataFrame(list(self.rows))
        return compute_psi(self.reference, window)
