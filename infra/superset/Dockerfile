# Superset đọc ClickHouse (clickhouse-connect cung cấp dialect clickhousedb://).
FROM apache/superset:5.0.0

USER root
RUN uv pip install --no-cache-dir --python /app/.venv/bin/python \
      clickhouse-connect==0.8.18 psycopg2-binary==2.9.10 \
 || pip install --no-cache-dir clickhouse-connect==0.8.18 psycopg2-binary==2.9.10
COPY infra/superset/superset_config.py /app/pythonpath/superset_config.py
USER superset
