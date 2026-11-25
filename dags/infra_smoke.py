"""Kiểm tra hạ tầng (TODO mục Hạ tầng): Spark + Iceberg REST, PyIceberg + MinIO, GPU worker.

Chạy tay từ UI (không lịch). Mỗi task tự dọn bảng nó tạo trong namespace ``smoke``.
"""

from __future__ import annotations

from datetime import datetime

from airflow.sdk import dag, task
from common.gpu import gpu_task

SPARK_TABLE = "lakehouse.smoke.spark_merge"
PYICEBERG_TABLE = "smoke.pyiceberg_roundtrip"


@dag(
    dag_id="infra_smoke",
    schedule=None,
    start_date=datetime(2026, 11, 1),
    catchup=False,
    tags=["infra"],
)
def infra_smoke():
    @task
    def spark_iceberg() -> list[tuple[str, int]]:
        """SparkSession có Iceberg SQL extensions → MERGE INTO chạy được trên catalog REST."""
        from vicap.spark import get_spark

        spark = get_spark("infra_smoke")
        try:
            spark.sql("CREATE NAMESPACE IF NOT EXISTS lakehouse.smoke")
            spark.sql(f"DROP TABLE IF EXISTS {SPARK_TABLE} PURGE")
            spark.sql(f"CREATE TABLE {SPARK_TABLE} (video_id STRING, n INT) USING iceberg")
            spark.sql(f"INSERT INTO {SPARK_TABLE} VALUES ('1', 1), ('2', 2)")
            spark.createDataFrame([("2", 20), ("3", 30)], "video_id STRING, n INT").createOrReplaceTempView("src")
            spark.sql(f"""
                MERGE INTO {SPARK_TABLE} t USING src s ON t.video_id = s.video_id
                WHEN MATCHED THEN UPDATE SET t.n = s.n
                WHEN NOT MATCHED THEN INSERT *
            """)
            rows = [(r.video_id, r.n) for r in spark.sql(f"SELECT * FROM {SPARK_TABLE} ORDER BY video_id").collect()]
            assert rows == [("1", 1), ("2", 20), ("3", 30)], rows
            spark.sql(f"DROP TABLE {SPARK_TABLE} PURGE")
            return rows
        finally:
            spark.stop()

    @task
    def pyiceberg_minio() -> int:
        """PyIceberg ghi / đọc qua REST + MinIO bằng helper của vicap (key vicap-pipeline)."""
        import pyarrow as pa
        from pyiceberg.schema import Schema
        from pyiceberg.types import LongType, NestedField, StringType

        from vicap.iceberg.catalog import load_lakehouse_catalog
        from vicap.iceberg.reader import read_table
        from vicap.iceberg.writer import ensure_table, replace_by_keys

        catalog = load_lakehouse_catalog()
        schema = Schema(
            NestedField(1, "video_id", StringType(), required=True),
            NestedField(2, "n", LongType(), required=False),
        )
        table = ensure_table(catalog, PYICEBERG_TABLE, schema)
        replace_by_keys(table, pa.table({"video_id": ["1", "2"], "n": [1, 2]}))
        replace_by_keys(table, pa.table({"video_id": ["2"], "n": [20]}))
        n = read_table(table.refresh()).num_rows
        assert n == 2, n
        catalog.purge_table(PYICEBERG_TABLE)
        return n

    gpu_check = gpu_task(
        "gpu_check",
        ["python", "-c", "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"],
    )

    [spark_iceberg(), pyiceberg_minio()] >> gpu_check


infra_smoke()
