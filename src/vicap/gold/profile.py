"""Hồ sơ creator bằng Spark (PLAN mục 4.3): ``gold.video_predictions`` → một dòng mỗi ``username``.

- ``label_ratios`` = trung bình ``probs`` trên mọi video của creator, bỏ Others, giữ thứ tự nhãn của checkpoint
  (``ModelConfig.profile_labels``): giống ``kltn_model.Predictor.profile`` nhưng ở dạng vector 19 phần tử.
- ``main_label`` = nhãn có tỉ lệ cao nhất (hoà thì lấy nhãn đứng trước, như ``argmax``).
- ``confidence`` theo số video: ``low`` < 5, ``medium`` 5–9, ``high`` ≥ 10.
- ``model_run_id`` lấy từ dự đoán mới nhất, ``batch_id`` = lô mới nhất có video của creator.

``merge_changed`` ghi bằng ``MERGE INTO`` và chỉ đụng tới dòng thật sự đổi, nên chạy lại không tạo snapshot rỗng.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import TYPE_CHECKING

from vicap.config import ModelConfig
from vicap.iceberg.tables import GOLD_CREATOR_PROFILE_HISTORY, GOLD_CREATOR_PROFILES
from vicap.silver.videos import MergeCounts

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame, SparkSession

MEDIUM_MIN_VIDEOS = 5
HIGH_MIN_VIDEOS = 10
PROFILE_COLUMNS = tuple(f.name for f in GOLD_CREATOR_PROFILES.schema.fields)
HISTORY_COLUMNS = tuple(f.name for f in GOLD_CREATOR_PROFILE_HISTORY.schema.fields)
# cột so sánh để biết hồ sơ có đổi không (updated_at luôn mới nên không tính)
COMPARE_COLUMNS = ("n_videos", "label_ratios", "main_label", "confidence", "model_run_id", "batch_id")


def confidence_level(n_videos: int) -> str:
    if n_videos >= HIGH_MIN_VIDEOS:
        return "high"
    return "medium" if n_videos >= MEDIUM_MIN_VIDEOS else "low"


def _confidence(n: Column) -> Column:
    from pyspark.sql import functions as F

    return (F.when(n >= HIGH_MIN_VIDEOS, "high").when(n >= MEDIUM_MIN_VIDEOS, "medium").otherwise("low"))


def build_profiles(predictions: DataFrame, updated_at: datetime, model: ModelConfig | None = None) -> DataFrame:
    """``predictions`` (cột ``username``, ``probs``, ``model_run_id``, ``batch_id``, ``predicted_at``) → hồ sơ."""
    from pyspark.sql import functions as F

    model = model or ModelConfig()
    keep = [j for j, lb in enumerate(model.labels) if lb not in model.profile_exclude]
    names = F.array(*[F.lit(model.labels[j]) for j in keep])
    return (
        predictions.filter(F.col("username").isNotNull())
        .groupBy("username")
        .agg(
            F.count("*").cast("int").alias("n_videos"),
            F.array(*[F.avg(F.col("probs")[j]).cast("float") for j in keep]).alias("label_ratios"),
            F.max_by("model_run_id", "predicted_at").alias("model_run_id"),
            F.max("batch_id").alias("batch_id"),
        )
        .withColumn("main_label", F.element_at(names, F.expr("array_position(label_ratios, array_max(label_ratios))")
                                               .cast("int")))
        .withColumn("confidence", _confidence(F.col("n_videos")))
        .withColumn("updated_at", F.lit(updated_at).cast("timestamp"))
        .select(*PROFILE_COLUMNS)
    )


def as_history(profiles: DataFrame, snapshot_date: date) -> DataFrame:
    from pyspark.sql import functions as F

    return profiles.withColumn("snapshot_date", F.lit(snapshot_date).cast("date")).select(*HISTORY_COLUMNS)


def merge_changed(spark: SparkSession, src: DataFrame, target: str, keys: Sequence[str],
                  compare: Sequence[str] = COMPARE_COLUMNS) -> MergeCounts:
    """``MERGE INTO target`` theo ``keys``, chỉ với dòng mới hoặc khác bản đang có ở một cột của ``compare``."""
    from pyspark.sql import functions as F

    current = spark.table(target).select(*keys, *[F.col(c).alias(f"_t_{c}") for c in compare],
                                         F.lit(True).alias("_t_exists"))
    joined = src.join(current, list(keys), "left")
    is_new = F.col("_t_exists").isNull()
    same = F.lit(True)
    for c in compare:
        same = same & F.col(c).eqNullSafe(F.col(f"_t_{c}"))
    changes = joined.filter(is_new | ~same).cache()
    total = src.count()
    inserted = changes.filter(is_new).count()
    updated = changes.count() - inserted
    counts = MergeCounts(inserted=inserted, updated=updated, unchanged=total - inserted - updated)
    if inserted + updated:
        view = "_merge_" + target.replace(".", "_")
        changes.select(*src.columns).createOrReplaceTempView(view)
        on = " AND ".join(f"t.{k} = s.{k}" for k in keys)
        spark.sql(f"""
            MERGE INTO {target} t
            USING {view} s
            ON {on}
            WHEN MATCHED THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
        """)
    changes.unpersist()
    return counts
