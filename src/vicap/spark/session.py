"""SparkSession local mode trong container Airflow, dùng chung Iceberg REST catalog với PyIceberg.

``spark_conf`` là hàm thuần (test không cần pyspark / JVM). ``get_spark`` import pyspark bên trong
hàm để import module này không đòi pyspark (image gpu-worker không cài).
"""

from __future__ import annotations

import glob
import os
from typing import TYPE_CHECKING

from vicap.config import IcebergConfig

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

ICEBERG_EXTENSIONS = "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions"
DEFAULT_JARS_DIR = "/opt/spark-jars"  # infra/airflow/Dockerfile


def _jars(jars_dir: str) -> str:
    jars = sorted(glob.glob(os.path.join(jars_dir, "*.jar")))
    return ",".join(jars)


def spark_conf(
    config: IcebergConfig | None = None,
    *,
    jars_dir: str | None = None,
    driver_memory: str = "1g",
) -> dict[str, str]:
    """Cấu hình Spark: Iceberg SQL extensions (bắt buộc cho ``MERGE INTO``) + catalog REST ``lakehouse``."""
    config = config or IcebergConfig()
    jars_dir = jars_dir or os.getenv("SPARK_JARS_DIR", DEFAULT_JARS_DIR)
    cat = f"spark.sql.catalog.{config.catalog_name}"
    conf = {
        "spark.sql.extensions": ICEBERG_EXTENSIONS,
        cat: "org.apache.iceberg.spark.SparkCatalog",
        f"{cat}.type": "rest",
        f"{cat}.uri": config.uri,
        f"{cat}.warehouse": config.warehouse,
        f"{cat}.io-impl": "org.apache.iceberg.aws.s3.S3FileIO",
        f"{cat}.s3.endpoint": config.s3_endpoint,
        f"{cat}.s3.path-style-access": "true",
        f"{cat}.s3.access-key-id": config.access_key,
        f"{cat}.s3.secret-access-key": config.secret_key,
        f"{cat}.client.region": config.region,
        # PyIceberg (Bronze, Gold) và Spark cùng ghi các bảng: không cache metadata, luôn đọc snapshot mới nhất
        f"{cat}.cache-enabled": "false",
        "spark.sql.defaultCatalog": config.catalog_name,
        # Spark 3.5 chưa có DataFrame.toArrow() public → toPandas() qua Arrow rồi pa.Table.from_pandas
        "spark.sql.execution.arrow.pyspark.enabled": "true",
        "spark.sql.session.timeZone": "UTC",
        "spark.driver.memory": driver_memory,
        "spark.sql.shuffle.partitions": "4",  # mỗi lô vài trăm video
        "spark.ui.enabled": "false",
        "spark.ui.showConsoleProgress": "false",  # thanh tiến trình làm rối log Airflow
        # Bảng tạo bằng PyIceberg có cột required; cột Spark dựng từ UDF / window luôn nullable nên MERGE bị từ
        # chối dù không có null. Null thật đã bị suite kiểm định (NotNull) chặn trước khi ghi.
        "spark.sql.iceberg.check-nullability": "false",
    }
    jars = _jars(jars_dir)
    if jars:
        conf["spark.jars"] = jars
    return conf


def get_spark(
    app_name: str,
    config: IcebergConfig | None = None,
    *,
    master: str = "local[2]",
    **kwargs,
) -> SparkSession:
    from pyspark.sql import SparkSession

    builder = SparkSession.builder.appName(app_name).master(master)
    for key, value in spark_conf(config, **kwargs).items():
        builder = builder.config(key, value)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
