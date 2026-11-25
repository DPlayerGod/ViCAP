"""Cấu hình đọc từ biến môi trường (`.env`, xem `.env.example`).

Khác HOSE:
- Không có mật khẩu mặc định. Biến bắt buộc được đọc trong ``default_factory``, nên thiếu biến
  thì báo lỗi khi *khởi tạo* config (``MinioConfig()``), không phải lúc import module.
- Mọi hàm nhận ``config: X | None = None`` rồi tự tạo ``X()`` bên trong (PLAN mục 9.2).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


class MissingEnvError(RuntimeError):
    """Thiếu biến môi trường bắt buộc."""


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise MissingEnvError(f"thiếu biến môi trường {name} (xem .env.example)")
    return value


def _env(name: str, default: str) -> str:
    return os.getenv(name) or default


def _required(name: str):
    return field(default_factory=lambda: _require_env(name))


def _optional(name: str, default: str):
    return field(default_factory=lambda: _env(name, default))


@dataclass(frozen=True)
class MinioConfig:
    endpoint: str = _optional("MINIO_ENDPOINT", "http://localhost:9000")
    access_key: str = _required("AWS_ACCESS_KEY_ID")
    secret_key: str = _required("AWS_SECRET_ACCESS_KEY")
    region: str = _optional("AWS_REGION", "us-east-1")
    bucket: str = "lakehouse"  # derived/ (frame, audio), quarantine/, warehouse/
    raw_bucket: str = "raw"  # lớp 0: videos/{video_id}/…, chỉ crawler ghi (PLAN mục 4.2)


def crawler_minio_config() -> MinioConfig:
    """Key của user MinIO ``vicap-crawler``: chỉ user này ghi được bucket ``raw`` (PLAN mục 4.2).

    ``MinioConfig()`` mặc định dùng ``AWS_*`` = user ``vicap-pipeline`` (đọc ``raw``, ghi ``lakehouse``).
    """
    return MinioConfig(
        access_key=_require_env("MINIO_CRAWLER_ACCESS_KEY"),
        secret_key=_require_env("MINIO_CRAWLER_SECRET_KEY"),
    )


@dataclass(frozen=True)
class IcebergConfig:
    catalog_name: str = "lakehouse"
    uri: str = _optional("CATALOG_URI_CLIENT", "http://localhost:8181")
    warehouse: str = _optional("CATALOG_WAREHOUSE", "s3://lakehouse/warehouse/")
    s3_endpoint: str = _optional("MINIO_ENDPOINT", "http://localhost:9000")
    access_key: str = _required("AWS_ACCESS_KEY_ID")
    secret_key: str = _required("AWS_SECRET_ACCESS_KEY")
    region: str = _optional("AWS_REGION", "us-east-1")


@dataclass(frozen=True)
class ClickHouseConfig:
    host: str = _optional("CLICKHOUSE_HOST", "localhost")
    port: int = field(default_factory=lambda: int(_env("CLICKHOUSE_PORT", "8123")))
    username: str = _required("CLICKHOUSE_USER")
    password: str = _required("CLICKHOUSE_PASSWORD")
    database: str = _optional("CLICKHOUSE_DB", "vicap")
    secure: bool = field(default_factory=lambda: _env("CLICKHOUSE_SECURE", "false").lower() == "true")


@dataclass(frozen=True)
class KafkaConfig:
    bootstrap_servers: str = _optional("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
    topic: str = _optional("KAFKA_TOPIC_VIDEOS", "tiktok.videos")
    group_id: str = "bronze-writer"  # PLAN mục 5: tắt auto commit, commit sau khi ghi Iceberg


@dataclass(frozen=True)
class MlflowConfig:
    tracking_uri: str = _optional("MLFLOW_TRACKING_URI", "http://localhost:5000")
    model_name: str = "kltn-p1"
    model_alias: str = "champion"

    @property
    def model_uri(self) -> str:
        return f"models:/{self.model_name}@{self.model_alias}"


@dataclass(frozen=True)
class ModelConfig:
    """Luật gán nhãn và encoder (PLAN mục 4.1, 4.3). Đổi encoder thì phải tính lại `gold.features`."""

    threshold: float = 0.75
    at_least_one: bool = True  # decide(P, th, at_least_one=True); không dùng decide_rule
    n_labels: int = 20
    text_dim: int = 768
    image_dim: int = 768
    n_text: int = 2  # [0] caption + hashtag, [1] transcript_raw
    n_images: int = 8  # _0 ảnh bìa, _1…_7 frame
    text_max_tokens: int = 256
    transcript_chunk_tokens: int = 254
    # TODO: đối chiếu đúng checkpoint PhoBERT với notebook v12 trước khi tính embedding mới.
    text_encoder: str = _optional("TEXT_ENCODER", "vinai/phobert-base")
    image_encoder: str = _optional("IMAGE_ENCODER", "google/siglip2-base-patch16-224")


@dataclass(frozen=True)
class PipelineConfig:
    minio: MinioConfig = field(default_factory=MinioConfig)
    iceberg: IcebergConfig = field(default_factory=IcebergConfig)
    clickhouse: ClickHouseConfig = field(default_factory=ClickHouseConfig)
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
