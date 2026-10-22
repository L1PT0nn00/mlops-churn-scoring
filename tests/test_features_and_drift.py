import pandas as pd

from churn.data import generate
from churn.drift import DriftMonitor, build_reference, compute_psi
from churn.features import CAT_FEATURES, FEATURES, NUMERIC_FEATURES, build_features


def test_features_same_for_batch_and_single_row():
    """Один расчёт признаков для обучения и для API: строка по отдельности = строка в пачке."""
    df = generate(n=50, seed=3)
    batch = build_features(df)
    single = pd.concat([build_features(df.iloc[[i]]) for i in range(len(df))])
    pd.testing.assert_frame_equal(batch.reset_index(drop=True), single.reset_index(drop=True))
    assert list(batch.columns) == FEATURES


def _reference():
    return build_reference(build_features(generate(n=5000, seed=1)), NUMERIC_FEATURES, CAT_FEATURES)


def test_psi_low_on_same_distribution():
    psi = compute_psi(_reference(), build_features(generate(n=5000, seed=2)))
    assert max(psi.values()) < 0.1


def test_psi_high_on_shifted_distribution():
    psi = compute_psi(_reference(), build_features(generate(n=5000, seed=2, shift=1.0)))
    assert psi["monthly_charges"] > 0.2
    assert psi["support_calls_90d"] > 0.2


def test_monitor_reports_every_n_rows():
    monitor = DriftMonitor(_reference(), window=200, every=50, min_rows=100)
    feats = build_features(generate(n=300, seed=4))
    results = [monitor.update(feats.iloc[i : i + 25]) for i in range(0, 300, 25)]
    assert results[0] is None  # окно ещё не набралось
    assert any(r is not None for r in results)
