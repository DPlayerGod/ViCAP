# Airflow 3.4 + Spark local (PLAN mục 9.4). Build từ gốc repo: docker compose build airflow-scheduler
FROM apache/airflow:3.4.0-python3.12

ARG AIRFLOW_VERSION=3.4.0
ARG ICEBERG_VERSION=1.6.1
ARG ICEBERG_SPARK_SHA1=889381d67a6c3538ee35aac5745d3f36208ab33a
ARG ICEBERG_AWS_SHA1=a88b785db369e6211dd79bae06c915877d5c4805

USER root
# Java 17 cho pyspark 3.5: image Airflow 3.4 là Debian 13, apt không còn openjdk-17 → chép JRE từ Temurin.
COPY --from=eclipse-temurin:17-jre /opt/java/openjdk /opt/java/openjdk
ENV JAVA_HOME=/opt/java/openjdk \
    PATH="/opt/java/openjdk/bin:${PATH}"
# procps vì spark-submit gọi `ps`.
RUN apt-get update \
 && apt-get install -y --no-install-recommends procps \
 && apt-get clean && rm -rf /var/lib/apt/lists/* \
 && java -version

# Jar Iceberg cho Spark: runtime (SQL extensions, MERGE INTO) + aws-bundle (S3FileIO, REST catalog).
RUN set -eux; \
    mkdir -p /opt/spark-jars; cd /opt/spark-jars; \
    base=https://repo1.maven.org/maven2/org/apache/iceberg; \
    curl -fsSLO "$base/iceberg-spark-runtime-3.5_2.12/${ICEBERG_VERSION}/iceberg-spark-runtime-3.5_2.12-${ICEBERG_VERSION}.jar"; \
    curl -fsSLO "$base/iceberg-aws-bundle/${ICEBERG_VERSION}/iceberg-aws-bundle-${ICEBERG_VERSION}.jar"; \
    echo "${ICEBERG_SPARK_SHA1}  iceberg-spark-runtime-3.5_2.12-${ICEBERG_VERSION}.jar" | sha1sum -c -; \
    echo "${ICEBERG_AWS_SHA1}  iceberg-aws-bundle-${ICEBERG_VERSION}.jar" | sha1sum -c -; \
    chmod 644 /opt/spark-jars/*.jar
ENV SPARK_JARS_DIR=/opt/spark-jars

USER airflow
# Cài dependency của vicap theo constraints của Airflow để không kéo lệch bản của Airflow.
# Code thật được mount vào /opt/airflow/src (PYTHONPATH), sửa code không cần build lại.
COPY --chown=airflow:root pyproject.toml README.md /tmp/vicap/
COPY --chown=airflow:root src /tmp/vicap/src
RUN pip install --no-cache-dir \
      "apache-airflow==${AIRFLOW_VERSION}" "/tmp/vicap[airflow,spark]" \
      --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.12.txt" \
 && rm -rf /tmp/vicap
