import pytest
from fastapi.testclient import TestClient

from churn.api import create_app
from churn.pipeline import run_retrain

CUSTOMER = {
    "customer_id": "C42-0000001",
    "tenure_months": 3,
    "monthly_charges": 89.5,
    "total_charges": 268.5,
    "support_calls_90d": 4,
    "num_products": 1,
    "age": 27,
    "contract": "month-to-month",
    "payment_method": "cash",
    "internet_service": "fiber",
}


@pytest.fixture
def client(settings, data_path):
    run_retrain(data_path, settings)
    with TestClient(create_app(settings)) as c:
        yield c


def test_not_ready_without_model(settings):
    with TestClient(create_app(settings)) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/ready").status_code == 503
        assert c.post("/predict", json=CUSTOMER).status_code == 503


def test_ready_and_predict(client):
    assert client.get("/ready").status_code == 200
    r = client.post("/predict", json=CUSTOMER)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["model_version"].startswith("v")
    assert "x-request-id" in r.headers


def test_risky_customer_scores_higher_than_loyal(client):
    loyal = {**CUSTOMER, "tenure_months": 72, "total_charges": 3600, "monthly_charges": 50,
             "contract": "two_year", "payment_method": "card", "support_calls_90d": 0, "num_products": 4}
    risky = client.post("/predict", json=CUSTOMER).json()["churn_probability"]
    safe = client.post("/predict", json=loyal).json()["churn_probability"]
    assert risky > safe


def test_invalid_input_is_422(client):
    assert client.post("/predict", json={**CUSTOMER, "age": 5}).status_code == 422
    assert client.post("/predict", json={**CUSTOMER, "contract": "lifetime"}).status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json={"items": [CUSTOMER] * 3})
    assert r.status_code == 200
    assert len(r.json()["items"]) == 3


def test_metrics_exposed(client):
    client.post("/predict", json=CUSTOMER)
    text = client.get("/metrics").text
    assert 'http_requests_total{method="POST",path="/predict",status="200"}' in text
    assert "churn_model_loaded 1.0" in text
    assert "churn_probability_bucket" in text


def test_hot_reload_after_promotion(client, settings, data_path):
    before = client.get("/model").json()["version"]
    run_retrain(data_path, settings.model_copy(update={"min_improvement": -1.0}))
    client.app.state.model_state.refresh()
    assert client.get("/model").json()["version"] != before
