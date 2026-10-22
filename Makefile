PY ?= .venv/bin/python
TAG ?= 0.1.0

.PHONY: install test lint train serve load drift up up-all down k8s-build k8s-up k8s-down

install:            ## окружение для локальной разработки
	python3 -m venv .venv && $(PY) -m pip install -r requirements.txt -r requirements-dev.txt

test:               ## тесты
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

train:              ## данные + обучение + ворота качества
	$(PY) -m churn.pipeline bootstrap

serve:              ## API на localhost:8000
	$(PY) -m uvicorn churn.api:app --reload --port 8000

load:               ## обычный трафик
	$(PY) scripts/load_test.py --n 600

drift:              ## трафик с дрейфом: смотри панель PSI в Grafana
	$(PY) scripts/load_test.py --n 600 --shift 1.0

up:                 ## стенд: api, mlflow, s3, postgres, prometheus, grafana
	docker compose up -d --build

up-all:             ## то же + airflow
	docker compose --profile airflow up -d --build

down:
	docker compose --profile airflow down

k8s-build:          ## образ для локального кубера Docker Desktop
	docker build -t churn-mlops:$(TAG) .

k8s-up:
	kubectl apply -k k8s/

k8s-down:
	kubectl delete -k k8s/
