"""vicap.spark.spark_conf: kiểm tra cấu hình mà không cần pyspark / JVM."""

from __future__ import annotations

from pathlib import Path

from vicap.config import IcebergConfig
from vicap.spark import spark_conf


def _config() -> IcebergConfig:
    return IcebergConfig(uri="http://iceberg-rest:8181", s3_endpoint="http://minio:9000",
                         access_key="pipe", secret_key="s3cret")


def test_iceberg_extensions_and_rest_catalog(tmp_path: Path) -> None:
    conf = spark_conf(_config(), jars_dir=str(tmp_path))
    # thiếu dòng này thì MERGE INTO báo lỗi cú pháp (PLAN mục 9.4)
    assert conf["spark.sql.extensions"] == "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions"
    assert conf["spark.sql.catalog.lakehouse"] == "org.apache.iceberg.spark.SparkCatalog"
    assert conf["spark.sql.catalog.lakehouse.type"] == "rest"
    assert conf["spark.sql.catalog.lakehouse.uri"] == "http://iceberg-rest:8181"
    assert conf["spark.sql.catalog.lakehouse.s3.endpoint"] == "http://minio:9000"
    assert conf["spark.sql.catalog.lakehouse.s3.path-style-access"] == "true"
    assert conf["spark.sql.catalog.lakehouse.s3.access-key-id"] == "pipe"
    assert conf["spark.sql.defaultCatalog"] == "lakehouse"
    assert conf["spark.sql.execution.arrow.pyspark.enabled"] == "true"
    assert "spark.jars" not in conf  # thư mục rỗng


def test_jars_from_dir(tmp_path: Path) -> None:
    for name in ("iceberg-aws-bundle-1.6.1.jar", "iceberg-spark-runtime-3.5_2.12-1.6.1.jar", "README.txt"):
        (tmp_path / name).write_text("x")
    jars = spark_conf(_config(), jars_dir=str(tmp_path))["spark.jars"].split(",")
    assert [Path(j).name for j in jars] == ["iceberg-aws-bundle-1.6.1.jar", "iceberg-spark-runtime-3.5_2.12-1.6.1.jar"]


def test_jars_dir_from_env(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "a.jar").write_text("x")
    monkeypatch.setenv("SPARK_JARS_DIR", str(tmp_path))
    assert spark_conf(_config())["spark.jars"].endswith("a.jar")
