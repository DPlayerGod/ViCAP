"""Crawler: TikTok → bucket ``raw`` + topic ``tiktok.videos`` (PLAN mục 2, 4.2; TODO Pipeline).

Mỗi bài:

1. Lấy ``itemStruct`` (``TikTokClient.fetch_item``), đọc trường chính (``parse_item``).
2. Ghi file lên ``raw/videos/{video_id}/`` bằng user MinIO ``vicap-crawler``:
   ``video.mp4`` hoặc ``img_{NNN}.{jpg|png}`` (slideshow), ``cover.{jpg|png}`` (nguyên bytes), rồi ``info.json``
   (``itemStruct`` gốc) sau cùng.
3. Chỉ khi ghi xong mới gửi message (key ``video_id``) vào Kafka. Bronze đọc message thấy key nào thì file
   đó đã có trên MinIO.

Bài lỗi được thử lại; vẫn lỗi thì gửi message ``status="error"`` để Bronze ghi vào ``ops.quarantine``.
``mp4`` đã có trên MinIO thì không tải lại (crawl lại chủ yếu để cập nhật lượt xem / thích), trừ khi
``refresh_media=True``. Ảnh bìa, ảnh slideshow và ``info.json`` luôn ghi lại.

Task crawl chạy lại (retry của Airflow) sẽ gửi lại message của cùng ``batch_id``: Bronze phải khử trùng
theo ``(batch_id, video_id)``, giữ ``crawl_ts`` mới nhất.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from loguru import logger

from vicap.config import KafkaConfig, MinioConfig, crawler_minio_config
from vicap.ingestion.message import VideoMessage
from vicap.ingestion.parse import SLIDESHOW, PostMeta, parse_item
from vicap.storage import paths
from vicap.storage.objects import s3_filesystem
from vicap.utils.dates import now_utc

MIN_VIDEO_BYTES = 10_000  # nhỏ hơn thì coi như tải hỏng (Giai đoạn 0)
_CONTENT_TYPES = {"jpg": "image/jpeg", "png": "image/png"}


class Client(Protocol):
    def list_post_ids(self, username: str, n: int) -> list[str]: ...
    def fetch_item(self, username: str, video_id: str) -> dict[str, Any]: ...
    def video_source(self, item: dict[str, Any], username: str) -> tuple[str, dict[str, str]]: ...
    def download(self, url: str, headers: dict[str, str] | None = None) -> bytes: ...


class Store(Protocol):
    def put(self, key: str, data: bytes, content_type: str) -> None: ...
    def exists(self, key: str) -> bool: ...


Sink = Callable[[VideoMessage], None]


class RawStore:
    """Bucket ``raw`` bằng key của ``vicap-crawler`` (chỉ user này ghi được ``raw``)."""

    def __init__(self, config: MinioConfig | None = None, fs: Any = None):
        self.config = config or crawler_minio_config()
        self.fs = fs or s3_filesystem(self.config)
        self.bucket = self.config.raw_bucket

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self.fs.pipe_file(f"{self.bucket}/{key}", data, ContentType=content_type)

    def exists(self, key: str) -> bool:
        return self.fs.exists(f"{self.bucket}/{key}")


class KafkaSink:
    """Gửi message vào ``tiktok.videos``. ``close()`` flush và báo lỗi nếu có message gửi hỏng."""

    def __init__(self, config: KafkaConfig | None = None, producer: Any = None):
        self.config = config or KafkaConfig()
        if producer is None:
            from kafka import KafkaProducer

            producer = KafkaProducer(
                bootstrap_servers=self.config.bootstrap_servers.split(","),
                acks="all",
                retries=5,
                max_in_flight_requests_per_connection=1,  # thử lại không đảo thứ tự
                linger_ms=50,
            )
        self.producer = producer
        self._pending: list[Any] = []

    def __call__(self, message: VideoMessage) -> None:
        self._pending.append(self.producer.send(self.config.topic, key=message.key(), value=message.to_bytes()))

    def close(self, timeout: float = 30) -> None:
        try:
            self.producer.flush(timeout)
            for future in self._pending:
                future.get(timeout=timeout)  # ném lỗi Kafka nếu message không gửi được
        finally:
            self._pending.clear()
            self.producer.close()

    def __enter__(self) -> KafkaSink:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass
class CrawlStats:
    batch_id: str
    creators: int = 0
    posts: int = 0
    ok: int = 0
    failed: int = 0
    media_skipped: int = 0  # mp4 đã có trên MinIO, không tải lại
    creator_errors: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"batch {self.batch_id}: {self.creators} creator, {self.posts} bài, ok {self.ok}, lỗi {self.failed}, "
            f"không tải lại mp4 {self.media_skipped}, creator lỗi {len(self.creator_errors)}"
        )


def _put_image(store: Store, key: str, data: bytes) -> str:
    store.put(key, data, _CONTENT_TYPES[key.rsplit(".", 1)[-1]])
    return key


def _download_any(client: Client, urls: Iterable[str]) -> bytes:
    """Thử lần lượt các CDN, trả bytes của URL đầu tiên tải được."""
    last: Exception | None = None
    for url in urls:
        try:
            return client.download(url)
        except Exception as e:  # noqa: BLE001 — thử CDN tiếp theo
            last = e
    raise RuntimeError(f"không tải được từ CDN nào: {last}")


def crawl_post(
    client: Client,
    store: Store,
    username: str,
    video_id: str,
    *,
    batch_id: str,
    crawl_ts: datetime,
    refresh_media: bool = False,
) -> tuple[VideoMessage, bool]:
    """Tải một bài lên ``raw/``. Trả ``(message, đã bỏ qua mp4 vì có sẵn)``."""
    item = client.fetch_item(username, video_id)
    meta: PostMeta = parse_item(item)
    if meta.video_id != video_id:
        raise ValueError(f"TikTok trả bài {meta.video_id} khi hỏi {video_id}")

    skipped = False
    image_keys: list[str] = []
    if meta.media_type == SLIDESHOW:
        for i, urls in enumerate(meta.image_urls, 1):
            data = _download_any(client, urls)
            key = paths.raw_video(video_id, paths.slide_name(i, paths.image_ext(data)))
            image_keys.append(_put_image(store, key, data))
        media_key = paths.raw_prefix(video_id)
    else:
        media_key = paths.raw_video(video_id, paths.VIDEO_FILE)
        if not refresh_media and store.exists(media_key):
            skipped = True
        else:
            url, headers = client.video_source(item, meta.username)
            data = client.download(url, headers)
            if len(data) < MIN_VIDEO_BYTES:
                raise RuntimeError(f"{video_id}: mp4 chỉ có {len(data)} byte")
            store.put(media_key, data, "video/mp4")

    thumbnail_key = None
    if meta.cover_url:
        data = client.download(meta.cover_url)
        thumbnail_key = _put_image(store, paths.raw_video(video_id, paths.cover_name(paths.image_ext(data))), data)

    info_key = paths.raw_video(video_id, paths.INFO_FILE)
    store.put(info_key, json.dumps(item, ensure_ascii=False).encode(), "application/json")

    message = VideoMessage(
        video_id=video_id,
        username=meta.username,
        batch_id=batch_id,
        crawl_ts=crawl_ts,
        media_type=meta.media_type,
        posted_at=meta.posted_at,
        description=meta.description,
        hashtags=list(meta.hashtags),
        duration=meta.duration,
        width=meta.width,
        height=meta.height,
        play_count=meta.play_count,
        like_count=meta.like_count,
        comment_count=meta.comment_count,
        share_count=meta.share_count,
        save_count=meta.save_count,
        follower_count=meta.follower_count,
        info_key=info_key,
        media_key=media_key,
        thumbnail_key=thumbnail_key,
        image_keys=image_keys,
    )
    return message, skipped


def crawl_creators(
    usernames: Iterable[str],
    *,
    n: int,
    batch_id: str,
    client: Client,
    store: Store,
    send: Sink,
    retries: int = 2,
    sleep: float = 1.0,
    refresh_media: bool = False,
    now: Callable[[], datetime] = now_utc,
    pause: Callable[[float], None] = time.sleep,
) -> CrawlStats:
    """Crawl ``n`` bài gần nhất của từng creator. Một bài / creator lỗi không dừng cả lô."""
    stats = CrawlStats(batch_id=batch_id)
    for username in usernames:
        stats.creators += 1
        try:
            ids = client.list_post_ids(username, n)
        except Exception as e:  # noqa: BLE001 — ghi lại, sang creator tiếp
            stats.creator_errors[username] = f"{type(e).__name__}: {e}"[:300]
            logger.warning("[{}] không lấy được danh sách bài: {}", username, e)
            continue
        if not ids:
            stats.creator_errors[username] = "danh sách bài rỗng"
            logger.warning("[{}] danh sách bài rỗng", username)
            continue

        for k, video_id in enumerate(ids, 1):
            stats.posts += 1
            crawl_ts = now()
            error = ""
            for attempt in range(retries + 1):
                try:
                    message, skipped = crawl_post(
                        client, store, username, video_id,
                        batch_id=batch_id, crawl_ts=crawl_ts, refresh_media=refresh_media,
                    )
                    break
                except Exception as e:  # noqa: BLE001 — lỗi tạm thời (vd. status=10231) thường qua khi thử lại
                    error = f"{type(e).__name__}: {e}"
                    logger.info("[{} {}] lần {} lỗi: {}", username, video_id, attempt + 1, error)
                    if attempt < retries:
                        pause(3 * (attempt + 1))
            else:
                message, skipped = (
                    VideoMessage.failed(video_id, username, batch_id=batch_id, crawl_ts=crawl_ts, error=error),
                    False,
                )

            send(message)
            stats.ok += message.ok
            stats.failed += not message.ok
            stats.media_skipped += skipped
            logger.info(
                "[{} {}/{}] {} {}{}", username, k, len(ids), video_id,
                message.media_type if message.ok else "LỖI " + (message.error or "")[:120],
                " (mp4 có sẵn)" if skipped else "",
            )
            pause(sleep)
    logger.info(stats.summary())
    return stats
