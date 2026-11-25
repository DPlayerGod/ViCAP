"""Key trên MinIO và quy ước tên file (PLAN mục 4.2). Thay cho ``StagingPathBuilder`` của HOSE.

Hàm ở đây chỉ dựng key (không có ``s3://`` và bucket); bucket lấy từ ``MinioConfig``:

- Bucket ``raw`` (``MinioConfig.raw_bucket``) — lớp 0, nằm ngoài Iceberg, **chỉ crawler ghi**:
  ``raw_video(video_id, name)`` → ``videos/{video_id}/{name}`` với ``name`` là ``video.mp4``,
  ``info.json``, ``cover.{jpg|png}`` hoặc ``img_{NNN}.{jpg|png}`` (slideshow).
- Bucket ``lakehouse`` (``MinioConfig.bucket``): ``frame`` / ``audio`` do gpu-worker sinh ra từ ``raw``,
  ``quarantine`` cho lô / dòng bị cách ly, ``warehouse/`` của Iceberg.

Key chỉ theo ``video_id`` (không phụ thuộc username), crawl lại thì ghi đè đúng file cũ.
``legacy_stem`` = ``{username}__{YYYYMMDD_HHMMSS}__{video_id}`` (giờ đăng, UTC) là kiểu tên của
dataset v5, chỉ dùng để ghép file / dòng v5 với ``video_id``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime

from vicap.utils.dates import format_date

SEP = "__"
POSTED_AT_FORMAT = "%Y%m%d_%H%M%S"
N_FRAMES = 8  # _0 ảnh bìa, _1…_7 frame (PLAN mục 4.1)
IMAGE_EXTS = ("jpg", "png")

VIDEO_FILE = "video.mp4"
INFO_FILE = "info.json"  # metadata yt-dlp đầy đủ; message Kafka chỉ mang key tới file này

_VIDEO_ID = re.compile(r"^\d+$")
_POSTED_AT = re.compile(r"^\d{8}_\d{6}$")
_IMG_SUFFIX = re.compile(r"__img_\d+$")
_EXT = re.compile(r"\.[A-Za-z][A-Za-z0-9]{1,4}$")
_RAW_NAME = re.compile(r"^(video\.mp4|info\.json|cover\.(jpg|png)|img_\d{3}\.(jpg|png))$")
_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]+$")


def _check_video_id(video_id: str) -> str:
    video_id = str(video_id)
    if not _VIDEO_ID.match(video_id):
        raise ValueError(f"video_id phải là chuỗi số: {video_id!r}")
    return video_id


def _check_ext(ext: str) -> str:
    ext = ext.lower().lstrip(".")
    ext = "jpg" if ext == "jpeg" else ext
    if ext not in IMAGE_EXTS:
        raise ValueError(f"đuôi ảnh không hỗ trợ: {ext!r}")
    return ext


def image_ext(data: bytes) -> str:
    """Đuôi theo magic bytes; ảnh bìa lưu nguyên bytes nên đuôi là thuộc tính của từng ảnh."""
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    raise ValueError("không nhận ra định dạng ảnh (chỉ hỗ trợ JPEG / PNG)")


# ─────────────────────────── Bucket raw (chỉ crawler ghi) ───────────────────


def cover_name(ext: str) -> str:
    return f"cover.{_check_ext(ext)}"


def slide_name(index: int, ext: str = "jpg") -> str:
    """Ảnh slideshow, đánh số từ 1 như dataset v5: ``img_001.jpg``."""
    if index < 1:
        raise ValueError("ảnh slideshow đánh số từ 1")
    return f"img_{index:03d}.{_check_ext(ext)}"


def raw_prefix(video_id: str) -> str:
    return f"videos/{_check_video_id(video_id)}/"


def raw_video(video_id: str, name: str) -> str:
    """``videos/{video_id}/{name}``.

    ``name`` ∈ ``video.mp4``, ``info.json``, ``cover.{jpg|png}``, ``img_{NNN}.{jpg|png}``.
    """
    if not _RAW_NAME.match(name):
        raise ValueError(f"tên file raw không hợp lệ: {name!r}")
    return raw_prefix(video_id) + name


# ─────────────────────────── Bucket lakehouse ───────────────────────────────


def frame_name(video_id: str, index: int) -> str:
    """``{video_id}_{k}.jpg``; ``_0`` là ảnh bìa, ``_1…_7`` là frame.

    Cả 8 ảnh đều JPEG (chuyển RGB, cạnh dài ≤ 640, quality 90), kể cả khi ảnh bìa gốc là PNG, như
    ``extract_frame.py`` đã dựng dataset ``tiktok-frames-v1`` (PLAN mục 4.1).
    """
    if not 0 <= index < N_FRAMES:
        raise ValueError(f"index frame phải trong [0, {N_FRAMES - 1}]: {index}")
    return f"{_check_video_id(video_id)}_{index}.jpg"


def audio_name(video_id: str) -> str:
    """mp3 16 kHz mono 64 kbps, như dataset ``tiktok-audio-v1`` (PLAN mục 4.1)."""
    return f"{_check_video_id(video_id)}.mp3"


def frame(video_id: str, k: int) -> str:
    return f"derived/frames/{frame_name(video_id, k)}"


def audio(video_id: str) -> str:
    return f"derived/audio/{audio_name(video_id)}"


def quarantine(domain: str, processing_date: str | date | datetime, batch_id: str) -> str:
    """``quarantine/{domain}/processing_date=…/batch_id=…`` (không có ``/`` cuối)."""
    for part in (domain, batch_id):
        if not _SEGMENT.match(part):
            raise ValueError(f"không dùng được làm phần của key: {part!r}")
    return f"quarantine/{domain}/processing_date={format_date(processing_date)}/batch_id={batch_id}"


def s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


# ─────────────────────────── Tên kiểu dataset v5 ────────────────────────────


@dataclass(frozen=True)
class LegacyStem:
    username: str
    posted_at: datetime  # UTC
    video_id: str

    def __str__(self) -> str:
        return legacy_stem(self.username, self.posted_at, self.video_id)


def _to_utc(posted_at: datetime | int | float) -> datetime:
    if isinstance(posted_at, (int, float)):  # createTime (epoch giây) của TikTok
        return datetime.fromtimestamp(posted_at, tz=UTC)
    if posted_at.tzinfo is None:
        raise ValueError("posted_at phải có timezone (UTC)")
    return posted_at.astimezone(UTC)


def legacy_username(username: str) -> str:
    """Username như dataset v5 ghi trong tên file (bỏ dấu ``.``)."""
    return username.replace(".", "")


def legacy_stem(username: str, posted_at: datetime | int | float, video_id: str) -> str:
    """Dựng ``{username}__{YYYYMMDD_HHMMSS}__{video_id}`` theo kiểu dataset v5."""
    stamp = _to_utc(posted_at).strftime(POSTED_AT_FORMAT)
    return SEP.join((legacy_username(username), stamp, _check_video_id(video_id)))


def parse_legacy_stem(name: str) -> LegacyStem:
    """Tách tên file / thư mục kiểu v5 → (username, posted_at, video_id).

    Nhận ``….mp4``, thư mục slideshow, ``…__img_001.jpg`` và đường dẫn đầy đủ. Username có thể chứa
    ``_`` hoặc ``__`` nên tách từ bên phải: bỏ đuôi file và hậu tố ``__img_NNN`` rồi ``rsplit("__", 2)``.
    """
    base = re.split(r"[\\/]", name.rstrip("/\\"))[-1]
    base = _EXT.sub("", base)
    base = _IMG_SUFFIX.sub("", base)
    parts = base.rsplit(SEP, 2)
    if len(parts) != 3 or not parts[0]:
        raise ValueError(f"không phải tên kiểu {{username}}__{{YYYYMMDD_HHMMSS}}__{{video_id}}: {name!r}")
    username, stamp, video_id = parts
    if not _POSTED_AT.match(stamp):
        raise ValueError(f"phần thời gian sai định dạng YYYYMMDD_HHMMSS: {name!r}")
    posted_at = datetime.strptime(stamp, POSTED_AT_FORMAT).replace(tzinfo=UTC)
    return LegacyStem(username, posted_at, _check_video_id(video_id))


def video_id_from_legacy(name: str) -> str:
    return parse_legacy_stem(name).video_id


def legacy_slide_image_name(stem: str, index: int, ext: str = "jpg") -> str:
    """Ảnh trong slideshow của dataset v5: ``{stem}__img_001.jpg`` (đánh số từ 1)."""
    return f"{stem}__img_{index:03d}.{_check_ext(ext)}"
