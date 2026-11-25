"""Schema Iceberg bản v0 cho 8 bảng ở PLAN mục 4 và ``ops.quarantine`` (mục 9.3).

Bronze / Silver còn giữ cột ``raw_json`` (itemStruct gốc) để chỉnh schema sau khi crawler chạy thật.
Cột embedding lưu ``list<list<float>>`` (float32): ``text_emb`` 2×768, ``image_emb`` 8×768.
Field id ở đây chỉ để dựng ``Schema``; catalog gán lại id khi ``create_table``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.transforms import IdentityTransform
from pyiceberg.types import (
    BooleanType,
    DateType,
    FloatType,
    IcebergType,
    IntegerType,
    ListType,
    LongType,
    NestedField,
    StringType,
    TimestamptzType,
)

# Một cột: (tên, kiểu hoặc hàm dựng kiểu nhận bộ cấp id, required)
_TypeSpec = IcebergType | Callable[[Callable[[], int]], IcebergType]


def _list_of(element: _TypeSpec) -> Callable[[Callable[[], int]], IcebergType]:
    def build(next_id: Callable[[], int]) -> IcebergType:
        element_id = next_id()
        inner = element(next_id) if callable(element) else element
        return ListType(element_id=element_id, element_type=inner, element_required=True)

    return build


FLOAT_VECTOR = _list_of(FloatType())
FLOAT_MATRIX = _list_of(FLOAT_VECTOR)
STRING_LIST = _list_of(StringType())


def _schema(*columns: tuple[str, _TypeSpec, bool]) -> Schema:
    counter = iter(range(1, 10_000))

    def next_id() -> int:
        return next(counter)

    fields = []
    for name, spec, required in columns:
        field_id = next_id()
        field_type = spec(next_id) if callable(spec) else spec
        fields.append(NestedField(field_id=field_id, name=name, field_type=field_type, required=required))
    return Schema(*fields)


def _identity(schema: Schema, column: str) -> PartitionSpec:
    source = schema.find_field(column)
    return PartitionSpec(
        PartitionField(source_id=source.field_id, field_id=1000, transform=IdentityTransform(), name=column)
    )


@dataclass(frozen=True)
class TableDef:
    identifier: str
    schema: Schema
    partition_spec: PartitionSpec | None = None
    key: tuple[str, ...] = ()


BRONZE_VIDEOS_RAW_SCHEMA = _schema(
    ("video_id", StringType(), True),
    ("username", StringType(), True),
    ("posted_at", TimestamptzType(), False),
    ("duration", IntegerType(), False),  # giây
    ("media_type", StringType(), False),  # video / slideshow (PLAN mục 4.1)
    ("media_key", StringType(), False),  # videos/{video_id}/video.mp4 hoặc videos/{video_id}/ (slideshow)
    ("thumbnail_key", StringType(), False),  # videos/{video_id}/cover.{jpg|png}
    ("raw_json", StringType(), False),
    ("crawl_ts", TimestamptzType(), True),
    ("batch_id", StringType(), True),
    ("processing_date", DateType(), True),
)

BRONZE_CREATORS_RAW_SCHEMA = _schema(
    ("username", StringType(), True),
    ("follower_count", LongType(), False),
    ("discovered_via", StringType(), False),  # hashtag phát hiện
    ("discovered_at", TimestamptzType(), True),
    ("raw_json", StringType(), False),
    ("batch_id", StringType(), True),
)

SILVER_VIDEOS_SCHEMA = _schema(
    ("video_id", StringType(), True),
    ("username", StringType(), True),
    ("posted_at", TimestamptzType(), False),
    ("duration", IntegerType(), False),
    ("media_type", StringType(), True),
    ("caption", StringType(), False),  # đã bỏ #hashtag
    ("hashtags", STRING_LIST, False),
    ("play_count", LongType(), False),
    ("like_count", LongType(), False),
    ("comment_count", LongType(), False),
    ("share_count", LongType(), False),
    ("media_key", StringType(), False),
    ("thumbnail_key", StringType(), False),
    ("crawl_ts", TimestamptzType(), True),
    ("batch_id", StringType(), True),
)

SILVER_TRANSCRIPTS_SCHEMA = _schema(
    ("video_id", StringType(), True),
    ("transcript_raw", StringType(), False),
    ("has_transcript", BooleanType(), True),
    ("asr_model", StringType(), False),
    ("batch_id", StringType(), True),
    ("created_at", TimestamptzType(), True),
)

GOLD_FEATURES_SCHEMA = _schema(
    ("video_id", StringType(), True),
    ("text_emb", FLOAT_MATRIX, True),  # 2 × 768
    ("image_emb", FLOAT_MATRIX, True),  # 8 × 768
    ("n_source_images", IntegerType(), False),  # số ảnh trước khi linspace về 8
    ("text_encoder", StringType(), True),
    ("image_encoder", StringType(), True),
    ("batch_id", StringType(), True),
    ("created_at", TimestamptzType(), True),
)

GOLD_VIDEO_PREDICTIONS_SCHEMA = _schema(
    ("video_id", StringType(), True),
    ("username", StringType(), True),
    ("probs", FLOAT_VECTOR, True),  # 20 xác suất
    ("labels", STRING_LIST, True),  # decide(P, th, at_least_one=True), PLAN mục 4.3
    ("threshold", FloatType(), True),
    ("model_name", StringType(), True),
    ("model_version", StringType(), True),
    ("model_run_id", StringType(), True),
    ("batch_id", StringType(), True),
    ("predicted_at", TimestamptzType(), True),
)

_PROFILE_COLUMNS: tuple[tuple[str, _TypeSpec, bool], ...] = (
    ("username", StringType(), True),
    ("n_videos", IntegerType(), True),
    ("label_ratios", FLOAT_VECTOR, True),  # 19 nhãn, bỏ Others
    ("main_label", StringType(), False),
    ("confidence", StringType(), True),  # low / medium / high
    ("model_run_id", StringType(), True),
    ("batch_id", StringType(), True),
    ("updated_at", TimestamptzType(), True),
)

GOLD_CREATOR_PROFILES_SCHEMA = _schema(*_PROFILE_COLUMNS)
GOLD_CREATOR_PROFILE_HISTORY_SCHEMA = _schema(("snapshot_date", DateType(), True), *_PROFILE_COLUMNS)

# Dòng bị cách ly ở mọi bước (PLAN mục 9.3); vicap.quality.quarantine ghi vào đây.
OPS_QUARANTINE_SCHEMA = _schema(
    ("video_id", StringType(), False),  # null khi dòng không gắn với video (hồ sơ creator)
    ("username", StringType(), False),
    ("stage", StringType(), True),  # bronze_videos, silver_videos, gold_features, …
    ("batch_id", StringType(), True),
    ("processing_date", DateType(), True),
    ("errors", STRING_LIST, True),  # nhãn check, vd non_zero_vector:text_emb[0]
    ("severity", StringType(), True),  # error / warn
    ("raw_ref", StringType(), False),  # s3://raw/videos/{video_id}/
    ("resolved", BooleanType(), True),
    ("created_at", TimestamptzType(), True),
)


BRONZE_VIDEOS_RAW = TableDef(
    "bronze.videos_raw", BRONZE_VIDEOS_RAW_SCHEMA, _identity(BRONZE_VIDEOS_RAW_SCHEMA, "processing_date")
)
BRONZE_CREATORS_RAW = TableDef("bronze.creators_raw", BRONZE_CREATORS_RAW_SCHEMA)
SILVER_VIDEOS = TableDef("silver.videos", SILVER_VIDEOS_SCHEMA, key=("video_id",))
SILVER_TRANSCRIPTS = TableDef("silver.transcripts", SILVER_TRANSCRIPTS_SCHEMA, key=("video_id",))
GOLD_FEATURES = TableDef("gold.features", GOLD_FEATURES_SCHEMA, key=("video_id",))
GOLD_VIDEO_PREDICTIONS = TableDef("gold.video_predictions", GOLD_VIDEO_PREDICTIONS_SCHEMA, key=("video_id",))
GOLD_CREATOR_PROFILES = TableDef("gold.creator_profiles", GOLD_CREATOR_PROFILES_SCHEMA, key=("username",))
GOLD_CREATOR_PROFILE_HISTORY = TableDef(
    "gold.creator_profile_history",
    GOLD_CREATOR_PROFILE_HISTORY_SCHEMA,
    _identity(GOLD_CREATOR_PROFILE_HISTORY_SCHEMA, "snapshot_date"),
    key=("snapshot_date", "username"),
)

OPS_QUARANTINE = TableDef(
    "ops.quarantine", OPS_QUARANTINE_SCHEMA, _identity(OPS_QUARANTINE_SCHEMA, "processing_date")
)

ALL_TABLES: tuple[TableDef, ...] = (
    BRONZE_VIDEOS_RAW,
    BRONZE_CREATORS_RAW,
    SILVER_VIDEOS,
    SILVER_TRANSCRIPTS,
    GOLD_FEATURES,
    GOLD_VIDEO_PREDICTIONS,
    GOLD_CREATOR_PROFILES,
    GOLD_CREATOR_PROFILE_HISTORY,
    OPS_QUARANTINE,
)
