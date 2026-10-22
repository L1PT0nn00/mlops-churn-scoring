FROM python:3.11-slim
# sqlalchemy фиксируем явно: mlflow 2.18 не работает с SQLAlchemy 2.1+
# (транзитивная зависимость без версии однажды сломала сборку)
RUN pip install --no-cache-dir mlflow==2.18.0 sqlalchemy==2.0.36 boto3==1.35.76 psycopg2-binary==2.9.10 \
 && useradd --create-home --uid 10001 mlflow
USER mlflow
EXPOSE 5000
