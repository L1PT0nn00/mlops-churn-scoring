from churn.data import generate
from churn.validation import validate


def test_clean_data_passes():
    assert validate(generate(n=3000, seed=1)).ok


def test_too_few_rows_fails():
    report = validate(generate(n=100, seed=1), min_rows=1000)
    assert not report.ok
    assert any("мало строк" in e for e in report.errors)


def test_missing_column_fails():
    df = generate(n=2000, seed=1).drop(columns=["age"])
    assert not validate(df).ok


def test_nulls_fail():
    df = generate(n=2000, seed=1)
    df.loc[: len(df) // 5, "monthly_charges"] = None
    assert any("пропусков" in e for e in validate(df).errors)


def test_broken_source_all_zeros_fails():
    """Источник сломался и шлёт одно значение: задача не должна стать зелёной."""
    df = generate(n=2000, seed=1)
    df["age"] = 18
    assert any("сломанный источник" in e for e in validate(df).errors)


def test_unknown_category_fails():
    df = generate(n=2000, seed=1)
    df.loc[:500, "contract"] = "lifetime"
    assert any("неизвестных категорий" in e for e in validate(df).errors)


def test_scoring_data_without_target_passes():
    df = generate(n=2000, seed=1).drop(columns=["churned"])
    assert validate(df, require_target=False).ok
