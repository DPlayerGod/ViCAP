"""``silver_clean`` (Spark): ``bronze.videos_raw`` → ``silver.videos``, khoá ``video_id``.

- Mỗi video giữ bản crawl mới nhất (``crawl_ts`` lớn nhất, hoà thì ``batch_id`` lớn hơn) trong các dòng Bronze
  được chọn.
- ``caption`` / ``hashtags`` dựng lại từ ``description`` (``desc`` gốc) bằng ``vicap.silver.text`` (UDF Python).
- Ghi bằng ``MERGE INTO`` (PLAN mục 5): video mới thì thêm; video đã có thì chỉ cập nhật khi bản này crawl sau
  bản trong Silver, nên chạy lại một lô cũ không đè bản mới hơn.

Import pyspark bên trong hàm: module này chỉ chạy trong image Airflow (có Java + jar Iceberg).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from vicap.iceberg.tables import SILVER_VIDEOS_SCHEMA
from vicap.silver.text import caption_hashtags, clean_caption

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

SILVER_COLUMNS: tuple[str, ...] = tuple(f.name for f in SILVER_VIDEOS_SCHEMA.fields)
# Cột Silver lấy thẳng từ Bronze (cùng tên, cùng kiểu)
_FROM_BRONZE = tuple(c for c in SILVER_COLUMNS if c not in {"caption", "hashtags", "updated_at"})


def latest_per_video(bronze: DataFrame) -> DataFrame:
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    w = Window.partitionBy("video_id").orderBy(F.col("crawl_ts").desc(), F.col("batch_id").desc())
    return bronze.withColumn("_rn", F.row_number().over(w)).filter(F.col("_rn") == 1).drop("_rn")


def build_silver(bronze: DataFrame, updated_at: datetime) -> DataFrame:
    """Dòng Bronze → dòng ``silver.videos`` (đúng thứ tự cột của schema)."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import ArrayType, StringType

    caption = F.udf(clean_caption, StringType())
    hashtags = F.udf(caption_hashtags, ArrayType(StringType(), containsNull=False))
    latest = latest_per_video(bronze)
    cols = {c: F.col(c) for c in _FROM_BRONZE}
    cols["caption"] = caption("description")
    cols["hashtags"] = hashtags("description")
    cols["updated_at"] = F.lit(updated_at).cast("timestamp")
    return latest.select(*(cols[c].alias(c) for c in SILVER_COLUMNS))


@dataclass(frozen=True)
class MergeCounts:
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0  # Silver đã có bản crawl bằng / mới hơn


def merge_into_silver(spark: SparkSession, src: DataFrame, target: str) -> MergeCounts:
    """``MERGE INTO target`` theo ``video_id``. Không có gì đổi thì không commit snapshot rỗng."""
    from pyspark.sql import functions as F

    current = spark.table(target).select("video_id", F.col("crawl_ts").alias("_t_crawl_ts"))
    joined = src.join(current, "video_id", "left")
    is_new = F.col("_t_crawl_ts").isNull()
    changes = joined.filter(is_new | (F.col("crawl_ts") > F.col("_t_crawl_ts")))
    total = src.count()
    inserted = changes.filter(is_new).count()
    updated = changes.count() - inserted
    counts = MergeCounts(inserted=inserted, updated=updated, unchanged=total - inserted - updated)
    if inserted + updated == 0:
        return counts

    changes.select(*SILVER_COLUMNS).createOrReplaceTempView("_silver_changes")
    spark.sql(f"""
        MERGE INTO {target} t
        USING _silver_changes s
        ON t.video_id = s.video_id
        WHEN MATCHED AND s.crawl_ts > t.crawl_ts THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)
    return counts
