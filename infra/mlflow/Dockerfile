# MLflow tracking server: backend Postgres (DB mlflow), artifact ở bucket s3://mlflow (MinIO).
# Phiên bản phải khớp mlflow trong nhóm [gpu] của pyproject.toml.
FROM ghcr.io/mlflow/mlflow:v2.22.1

RUN pip install --no-cache-dir psycopg2-binary==2.9.10 boto3==1.38.46
