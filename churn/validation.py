"""Проверки данных перед обучением и скорингом.

Airflow считает задачу успешной, если она не упала. Поэтому всё, что
должно остановить пайплайн на плохих данных, проверяется здесь явно.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from churn.data import ALLOWED_CATEGORIES, ID_COL, RAW_FEATURES, TARGET, TIME_COL

RANGES = {
    "tenure_months": (0, 600),
    "monthly_charges": (0.01, 10_000),
    "total_charges": (0, 10_000_000),
    "support_calls_90d": (0, 1_000),
    "num_products": (1, 50),
    "age": (18, 120),
}


class DataValidationError(Exception):
    def __init__(self, report: "ValidationReport"):
        self.report = report
        super().__init__("; ".join(report.errors))


@dataclass
class ValidationReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> "ValidationReport":
        if not self.ok:
            raise DataValidationError(self)
        return self


def validate(
    df: pd.DataFrame,
    min_rows: int = 1000,
    max_null_share: float = 0.05,
    require_target: bool = True,
    target_rate_bounds=(0.02, 0.60),
) -> ValidationReport:
    rep = ValidationReport()
    required = [ID_COL, TIME_COL] + RAW_FEATURES + ([TARGET] if require_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        rep.errors.append(f"нет колонок: {missing}")
        return rep

    rep.stats["rows"] = float(len(df))
    if len(df) < min_rows:
        rep.errors.append(f"слишком мало строк: {len(df)} < {min_rows}")

    for col in required:
        share = float(df[col].isna().mean())
        if share > max_null_share:
            rep.errors.append(f"{col}: пропусков {share:.1%} > {max_null_share:.0%}")

    for col, (lo, hi) in RANGES.items():
        values = pd.to_numeric(df[col], errors="coerce")
        bad = float(((values < lo) | (values > hi)).mean())
        if bad > 0.01:
            rep.errors.append(f"{col}: {bad:.1%} значений вне диапазона [{lo}, {hi}]")

    for col, allowed in ALLOWED_CATEGORIES.items():
        values = df[col].dropna()
        unknown = float((~values.isin(allowed)).mean()) if len(values) else 0.0
        if unknown > 0.01:
            rep.errors.append(f"{col}: {unknown:.1%} неизвестных категорий")

    # Одинаковые значения в числовой колонке почти всегда значат сломанный источник
    for col in ("monthly_charges", "age"):
        counts = df[col].value_counts(normalize=True, dropna=True)
        top_share = float(counts.iloc[0]) if len(counts) else 0.0
        if top_share > 0.5:
            rep.errors.append(f"{col}: одно значение у {top_share:.0%} строк, похоже на сломанный источник")

    if require_target:
        rate = float(df[TARGET].mean())
        rep.stats["target_rate"] = rate
        lo, hi = target_rate_bounds
        if not lo <= rate <= hi:
            rep.errors.append(f"доля оттока {rate:.1%} вне ожидаемого [{lo:.0%}, {hi:.0%}]")

    dup = float(df.duplicated(subset=[ID_COL, TIME_COL]).mean())
    if dup > 0:
        rep.warnings.append(f"дубликатов (id, дата): {dup:.2%}")

    if len(df):
        rep.stats["date_min"] = pd.Timestamp(df[TIME_COL].min()).timestamp()
        rep.stats["date_max"] = pd.Timestamp(df[TIME_COL].max()).timestamp()
    return rep
