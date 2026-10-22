FROM apache/airflow:2.10.4-python3.11

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Код проекта: DAG вызывают те же функции, что и CLI
COPY --chown=airflow:root churn/ /opt/airflow/project/churn/
ENV PYTHONPATH=/opt/airflow/project
