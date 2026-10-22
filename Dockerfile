
# --- Стадия сборки: колёса зависимостей. Компиляторы и кеш pip в финальный образ не попадут.
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# --- Финальный образ
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHURN_DATA_DIR=/data \
    CHURN_MODEL_DIR=/models

# tini становится PID 1: пробрасывает SIGTERM в приложение и убирает зомби-процессы
RUN apt-get update \
 && apt-get install -y --no-install-recommends tini \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Зависимости до кода: правка кода не ломает кеш этого слоя.
# Колёса подключаются bind-mount'ом только на время установки: если сделать
# COPY + rm -rf, удалённые файлы всё равно останутся в слое COPY и раздуют образ.
RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    pip install --no-cache-dir /wheels/*

COPY churn/ ./churn/
COPY scripts/ ./scripts/

RUN useradd --create-home --uid 10001 app \
 && mkdir -p /data /models \
 && chown app:app /data /models
USER app

EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "churn.api:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "20"]
