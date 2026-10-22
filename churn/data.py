"""Схема данных и генератор синтетических клиентов телеком-оператора.

В реальном контуре данные выгружаются из DWH/Postgres. Здесь генератор,
чтобы проект запускался без внешних источников. Параметр shift имитирует
дрейф: клиенты становятся моложе, тарифы дороже, звонков в поддержку больше.
"""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

ID_COL = "customer_id"
TIME_COL = "snapshot_date"
TARGET = "churned"

CONTRACTS = ["month-to-month", "one_year", "two_year"]
PAYMENTS = ["card", "bank_transfer", "cash"]
INTERNET = ["fiber", "dsl", "none"]

NUMERIC_RAW = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_calls_90d",
    "num_products",
    "age",
]
CATEGORICAL = ["contract", "payment_method", "internet_service"]
RAW_FEATURES = NUMERIC_RAW + CATEGORICAL

ALLOWED_CATEGORIES = {
    "contract": CONTRACTS,
    "payment_method": PAYMENTS,
    "internet_service": INTERNET,
}


def generate(
    n: int = 20_000,
    start: str = "2024-01-01",
    end: str = "2025-12-31",
    seed: int = 42,
    shift: float = 0.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    days = max((end_ts - start_ts).days, 1)
    offsets = rng.integers(0, days + 1, n)
    dates = start_ts + pd.to_timedelta(offsets, unit="D")
    progress = offsets / days

    contract = rng.choice(CONTRACTS, n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(INTERNET, n, p=[0.45, 0.40, 0.15])
    payment = rng.choice(PAYMENTS, n, p=[0.50, 0.35, 0.15])

    tenure = np.clip(np.round(rng.gamma(2.0, 12.0, n)), 0, 120)
    internet_add = np.select([internet == "fiber", internet == "none"], [20.0, -15.0], 0.0)
    monthly = np.round(np.clip(rng.normal(55 + 15 * shift, 18, n) + internet_add, 10, 250), 2)
    total = np.round(np.clip(monthly * np.maximum(tenure, 1) * rng.normal(1.0, 0.05, n), 0, None), 2)
    support = rng.poisson(1.0 + 2.0 * shift, n)
    products = rng.integers(1, 5, n)
    age = np.clip(np.round(rng.normal(40 - 8 * shift, 12, n)), 18, 85)

    logit = (
        -1.0
        + 1.4 * (contract == "month-to-month")
        - 0.9 * (contract == "two_year")
        - 0.035 * tenure
        + 0.018 * (monthly - 55)
        + 0.35 * support
        - 0.25 * (products - 1)
        + 0.45 * (payment == "cash")
        + 0.30 * (internet == "fiber")
        + 0.40 * progress  # закономерность слегка меняется со временем
    )
    churned = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame(
        {
            ID_COL: [f"C{seed % 100:02d}-{i:07d}" for i in range(n)],
            TIME_COL: dates,
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "support_calls_90d": support,
            "num_products": products,
            "age": age,
            "contract": contract,
            "payment_method": payment,
            "internet_service": internet,
            TARGET: churned,
        }
    )
    return df.sort_values(TIME_COL).reset_index(drop=True)


def save_csv(df: pd.DataFrame, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)  # атомарно: читатель не увидит недописанный файл
    return path


def load_csv(path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    return df
