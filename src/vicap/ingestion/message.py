"""Message của topic ``tiktok.videos`` (PLAN mục 4.2 "Kafka — topic"). Crawler ghi, Bronze writer đọc.

- Key = ``video_id`` (crawl lại cùng video vào cùng partition).
- Value = JSON UTF-8, vài KB: metadata chính + key MinIO trong bucket ``raw``. Không chứa mp4 hay cả
  ``info.json``; crawler đẩy file lên ``raw/`` trước rồi mới gửi message.
- ``status="error"``: bài không tải được sau khi thử lại. Vẫn gửi message để Bronze ghi vào
  ``ops.quarantine`` (stage ``crawl``), lỗi chỉ nằm ở một chỗ.
- ``v`` là phiên bản schema; đổi trường thì tăng ``v`` và cho Bronze đọc được cả bản cũ.

``from_bytes`` kiểm định đủ để Bronze tách message hỏng ra (``MessageError``) mà không dừng cả lô.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

SCHEMA_VERSION = 1
STATUS_OK = "ok"
STATUS_ERROR = "error"
MEDIA_TYPES = ("video", "slideshow")

_VIDEO_ID = re.compile(r"^\d+$")
_DATETIME_FIELDS = ("crawl_ts", "posted_at")
_INT_FIELDS = (
    "duration",
    "width",
    "height",
    "play_count",
    "like_count",
    "comment_count",
    "share_count",
    "save_count",
    "follower_count",
)


class MessageError(ValueError):
    """Message không đọc được hoặc sai schema."""


@dataclass(frozen=True)
class VideoMessage:
    video_id: str
    username: str
    batch_id: str
    crawl_ts: datetime  # UTC
    status: str = STATUS_OK
    error: str | None = None
    media_type: str | None = None
    posted_at: datetime | None = None  # UTC
    description: str | None = None  # caption gốc, gồm cả #hashtag
    hashtags: list[str] = field(default_factory=list)
    duration: int | None = None
    width: int | None = None
    height: int | None = None
    play_count: int | None = None
    like_count: int | None = None
    comment_count: int | None = None
    share_count: int | None = None
    save_count: int | None = None
    follower_count: int | None = None
    # Key trong bucket raw (không có s3:// và bucket)
    info_key: str | None = None  # videos/{video_id}/info.json
    media_key: str | None = None  # videos/{video_id}/video.mp4, hoặc videos/{video_id}/ với slideshow
    thumbnail_key: str | None = None  # videos/{video_id}/cover.{jpg|png}
    image_keys: list[str] = field(default_factory=list)  # slideshow: img_001…
    v: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate(self)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    def key(self) -> bytes:
        return self.video_id.encode()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for name in _DATETIME_FIELDS:
            if data[name] is not None:
                data[name] = data[name].isoformat()
        return data

    def to_bytes(self) -> bytes:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":")).encode()

    @classmethod
    def from_bytes(cls, raw: bytes | str) -> VideoMessage:
        try:
            data = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise MessageError(f"không phải JSON: {e}") from e
        if not isinstance(data, dict):
            raise MessageError("value phải là object JSON")
        if data.get("v") != SCHEMA_VERSION:
            raise MessageError(f"phiên bản schema không hỗ trợ: {data.get('v')!r}")
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise MessageError(f"trường lạ: {sorted(unknown)}")
        for name in _DATETIME_FIELDS:
            if data.get(name) is not None:
                try:
                    data[name] = datetime.fromisoformat(data[name])
                except (TypeError, ValueError) as e:
                    raise MessageError(f"{name} không phải ISO datetime: {data[name]!r}") from e
        try:
            return cls(**data)
        except TypeError as e:  # thiếu trường bắt buộc
            raise MessageError(str(e)) from e

    @classmethod
    def failed(cls, video_id: str, username: str, *, batch_id: str, crawl_ts: datetime, error: str) -> VideoMessage:
        return cls(
            video_id=video_id, username=username, batch_id=batch_id, crawl_ts=crawl_ts,
            status=STATUS_ERROR, error=error[:1000] or "lỗi không rõ",
        )


def _validate(m: VideoMessage) -> None:
    if not isinstance(m.video_id, str) or not _VIDEO_ID.match(m.video_id):
        raise MessageError(f"video_id không hợp lệ: {m.video_id!r}")
    for name in ("username", "batch_id"):
        if not isinstance(getattr(m, name), str) or not getattr(m, name):
            raise MessageError(f"{name} rỗng")
    for name in _DATETIME_FIELDS:
        value = getattr(m, name)
        if value is not None and (not isinstance(value, datetime) or value.tzinfo is None):
            raise MessageError(f"{name} phải là datetime có múi giờ")
    for name in _INT_FIELDS:
        value = getattr(m, name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise MessageError(f"{name} phải là số nguyên: {value!r}")
    if not isinstance(m.hashtags, list) or not all(isinstance(t, str) for t in m.hashtags):
        raise MessageError("hashtags phải là list[str]")
    if not isinstance(m.image_keys, list) or not all(isinstance(k, str) for k in m.image_keys):
        raise MessageError("image_keys phải là list[str]")

    if m.status == STATUS_ERROR:
        if not m.error:
            raise MessageError("message lỗi phải có error")
        return
    if m.status != STATUS_OK:
        raise MessageError(f"status không hợp lệ: {m.status!r}")
    if m.media_type not in MEDIA_TYPES:
        raise MessageError(f"media_type không hợp lệ: {m.media_type!r}")
    if m.posted_at is None:
        raise MessageError("thiếu posted_at")
    prefix = f"videos/{m.video_id}/"
    for name in ("info_key", "media_key"):
        value = getattr(m, name)
        if not value or not value.startswith(prefix):
            raise MessageError(f"{name} phải nằm dưới {prefix}: {value!r}")
    if m.thumbnail_key is not None and not m.thumbnail_key.startswith(prefix):
        raise MessageError(f"thumbnail_key phải nằm dưới {prefix}: {m.thumbnail_key!r}")
    if m.media_type == "slideshow" and not m.image_keys:
        raise MessageError("slideshow phải có image_keys")
