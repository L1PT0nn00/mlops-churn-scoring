"""Единый расчёт признаков для обучения и для сервиса.

Одна функция для обоих мест защищает от training-serving skew: признак
нельзя случайно посчитать по-разному в ноутбуке и в API.
"""

import numpy as np
import pandas as pd

from churn.data import CATEGORICAL, NUMERIC_RAW, RAW_FEATURES

DERIVED = ["avg_charge_per_month", "is_new_customer", "calls_per_product"]
NUMERIC_FEATURES = NUMERIC_RAW + DERIVED
CAT_FEATURES = list(CATEGORICAL)
FEATURES = NUMERIC_FEATURES + CAT_FEATURES


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"нет колонок: {missing}")

    out = df[RAW_FEATURES].copy()
    tenure = np.maximum(out["tenure_months"].astype(float), 1.0)
    out["avg_charge_per_month"] = out["total_charges"].astype(float) / tenure
    out["is_new_customer"] = (out["tenure_months"] < 6).astype(int)
    out["calls_per_product"] = out["support_calls_90d"] / np.maximum(out["num_products"], 1)
    for col in CAT_FEATURES:
        out[col] = out[col].astype(str)
    return out[FEATURES]
