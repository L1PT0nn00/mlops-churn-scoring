"""HTTP-сервис скоринга оттока.

    GET  /health         liveness: процесс жив
    GET  /ready          readiness: модель загружена, можно слать трафик
    GET  /model          какая версия в проде и её метрики
    POST /predict        один клиент
    POST /predict/batch  до 1000 клиентов
    GET  /metrics        метрики для Prometheus

Сервис раз в CHURN_MODEL_RELOAD_SECONDS проверяет реестр и подхватывает
новую прод-версию без перезапуска (и после отката тоже).
"""

import asyncio
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from churn.config import Settings, get_settings
from churn.drift import DriftMonitor
from churn.features import build_features
from churn.logging_utils import setup_logging
from churn.registry import LoadedModel, Registry

log = logging.getLogger("churn.api")

REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram(
    "http_request_duration_seconds", "Request latency", ["path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
PREDICTIONS = Counter("churn_predictions_total", "Predictions by label", ["label"])
PROBA = Histogram("churn_probability", "Predicted churn probability", buckets=[i / 10 for i in range(1, 11)])
PSI = Gauge("churn_feature_psi", "PSI of a feature vs training reference", ["feature"])
MODEL_LOADED = Gauge("churn_model_loaded", "1 if a model is loaded")
MODEL_INFO = Gauge("churn_model_info", "Currently loaded model version", ["version"])
MODEL_AUC = Gauge("churn_model_holdout_auc", "Holdout ROC-AUC of the loaded model")
MODEL_RELOADS = Counter("churn_model_reloads_total", "Model reload attempts", ["result"])


class CustomerIn(BaseModel):
    customer_id: str = Field(min_length=1, max_length=64)
    tenure_months: float = Field(ge=0, le=600)
    monthly_charges: float = Field(gt=0, le=10_000)
    total_charges: float = Field(ge=0)
    support_calls_90d: int = Field(ge=0, le=1_000)
    num_products: int = Field(ge=1, le=50)
    age: int = Field(ge=18, le=120)
    contract: Literal["month-to-month", "one_year", "two_year"]
    payment_method: Literal["card", "bank_transfer", "cash"]
    internet_service: Literal["fiber", "dsl", "none"]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customer_id": "C42-0000001", "tenure_months": 3, "monthly_charges": 89.5,
                "total_charges": 268.5, "support_calls_90d": 4, "num_products": 1, "age": 27,
                "contract": "month-to-month", "payment_method": "cash", "internet_service": "fiber",
            }
        }
    )


class PredictionOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    customer_id: str
    churn_probability: float
    will_churn: bool
    model_version: str


class BatchIn(BaseModel):
    items: List[CustomerIn] = Field(min_length=1, max_length=1000)


class BatchOut(BaseModel):
    items: List[PredictionOut]


class ModelState:
    def __init__(self, registry: Registry, settings: Settings):
        self.registry = registry
        self.settings = settings
        self.loaded: Optional[LoadedModel] = None
        self.monitor: Optional[DriftMonitor] = None
        self._lock = threading.Lock()

    def refresh(self) -> bool:
        try:
            version = self.registry.production_version()
        except Exception:
            log.exception("cannot read registry")
            MODEL_RELOADS.labels("error").inc()
            return False
        current = self.loaded
        if version is None or (current is not None and current.version == version):
            return False
        try:
            new = self.registry.load(version)
        except Exception:
            log.exception("cannot load model", extra={"version": version})
            MODEL_RELOADS.labels("error").inc()
            return False
        with self._lock:
            self.loaded = new
            self.monitor = DriftMonitor(new.reference, self.settings.drift_window, self.settings.drift_every)
        MODEL_INFO.clear()
        MODEL_INFO.labels(version).set(1)
        MODEL_LOADED.set(1)
        MODEL_AUC.set(float(new.meta.get("metrics", {}).get("roc_auc", 0.0)))
        PSI.clear()
        MODEL_RELOADS.labels("success").inc()
        log.info("model loaded", extra={"version": version, "previous": current.version if current else None})
        return True


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level)
    state = ModelState(Registry(settings.model_dir), settings)
    MODEL_LOADED.set(0)

    async def reload_loop() -> None:
        while True:
            await asyncio.sleep(settings.model_reload_seconds)
            await asyncio.to_thread(state.refresh)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state.refresh()
        task = asyncio.create_task(reload_loop())
        log.info("service started", extra={"model_version": state.loaded.version if state.loaded else None})
        yield
        task.cancel()
        log.info("service stopping")  # uvicorn получил SIGTERM и дорабатывает текущие запросы

    app = FastAPI(title="Churn scoring", version="0.1.0", lifespan=lifespan)
    app.state.model_state = state

    @app.middleware("http")
    async def observe(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")  # шаблон пути, а не URL: без взрыва меток
            elapsed = time.perf_counter() - start
            if path not in ("/metrics", "/health", "/ready"):
                REQUESTS.labels(request.method, path, str(status)).inc()
                LATENCY.labels(path).observe(elapsed)
                log.info("request", extra={"request_id": request_id, "method": request.method, "path": path,
                                           "status": status, "duration_ms": round(elapsed * 1000, 1)})
        response.headers["x-request-id"] = request_id
        return response

    def score(items: List[CustomerIn]) -> List[PredictionOut]:
        loaded, monitor = state.loaded, state.monitor
        if loaded is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        raw = pd.DataFrame([item.model_dump() for item in items])
        features = build_features(raw)
        proba = loaded.model.predict_proba(features)[:, 1]
        if monitor is not None:
            psi = monitor.update(features)
            if psi:
                for feature, value in psi.items():
                    PSI.labels(feature).set(value)
        out = []
        for item, p in zip(items, proba):
            label = bool(p >= loaded.threshold)
            PREDICTIONS.labels("churn" if label else "stay").inc()
            PROBA.observe(float(p))
            out.append(PredictionOut(customer_id=item.customer_id, churn_probability=round(float(p), 4),
                                     will_churn=label, model_version=loaded.version))
        return out

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready(response: Response):
        if state.loaded is None:
            response.status_code = 503
            return {"status": "not ready", "reason": "model not loaded"}
        return {"status": "ready", "model_version": state.loaded.version}

    @app.get("/model")
    def model_info():
        if state.loaded is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        meta = state.loaded.meta
        keys = ("version", "trained_at", "train_period", "test_period", "metrics", "threshold", "gate")
        return {k: meta.get(k) for k in keys}

    @app.post("/predict", response_model=PredictionOut)
    def predict(customer: CustomerIn):
        return score([customer])[0]

    @app.post("/predict/batch", response_model=BatchOut)
    def predict_batch(batch: BatchIn):
        return BatchOut(items=score(batch.items))

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
