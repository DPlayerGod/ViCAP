#!/bin/sh
# Chạy một lần khi volume Postgres còn trống (docker-entrypoint-initdb.d).
# Mỗi dịch vụ một DB; cùng owner là POSTGRES_USER.
set -eu

for db in airflow iceberg mlflow superset; do
  echo "tạo database $db"
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres \
    -c "CREATE DATABASE $db OWNER \"$POSTGRES_USER\""
done
