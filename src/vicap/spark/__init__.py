"""SparkSession local cho các job tổng hợp (silver_clean, profile) — PLAN mục 9.4."""

from vicap.spark.session import get_spark, spark_conf

__all__ = ["get_spark", "spark_conf"]
