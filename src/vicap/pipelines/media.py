"""Bước ``media`` (trước ``asr`` / ``embed``): ``silver.videos`` → 8 ảnh + audio ở ``lakehouse/derived/`` +
``silver.media``.

Mỗi video: tải file gốc từ ``raw/`` về thư mục tạm, dựng 8 ảnh và mp3 bằng ``vicap.features.media`` (port của
``extract_frame.py`` / ``extract_audio.py``), ghi ``derived/frames/{video_id}_{k}.jpg`` và
``derived/audio/{video_id}.mp3``, rồi ghi một dòng ``silver.media`` (``replace_by_keys``). Video đã có trong
``silver.media`` thì bỏ qua (``--refresh`` để làm lại). Lỗi → ``ops.quarantine`` stage ``media``.

Chỉ cần CPU + ffmpeg của ``imageio-ffmpeg`` (extra ``media``), chạy được trên máy host lẫn gpu-worker::

    uv run python -m vicap.pipelines.media
    uv run python -m vicap.pipelines.media --batch-id 20261125-143434 --refresh
"""

from __future__ import annotations

import argparse
import tempfile
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow as pa
from loguru import logger
from pyiceberg.expressions import In
from pyiceberg.io.pyarrow import schema_to_pyarrow

from vicap.config import MinioConfig
from vicap.features.media import Frames, extract_audio, slideshow_frames, video_frames
from vicap.iceberg.reader import read_table
from vicap.iceberg.tables import OPS_QUARANTINE, SILVER_MEDIA, SILVER_VIDEOS
from vicap.iceberg.writer import ensure_table, replace_by_keys
from vicap.quality.quarantine import quarantine_by_batch
from vicap.storage import paths
from vicap.storage.objects import s3_filesystem
from vicap.utils.dates import now_utc, parse_date
from vicap.utils.log import setup_logging

STAGE = "media"


@dataclass
class MediaStats:
    videos: int = 0
    done: int = 0
    skipped: int = 0  # đã có trong silver.media
    errors: int = 0
    batches: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"media: {self.videos} video (lô {', '.join(self.batches) or '-'}); cắt {self.done}, "
                f"bỏ qua {self.skipped}, lỗi {self.errors}")


def _ext(key: str) -> str:
    return PurePosixPath(key).suffix or ".bin"


def build_media(row: dict, *, read_raw: Callable[[str], bytes], write: Callable[[str, bytes, str], None],
                created_at: datetime) -> dict:
    """Một dòng ``silver.videos`` → ghi 8 ảnh (+ mp3) và trả về dòng ``silver.media``."""
    vid = row["video_id"]
    with tempfile.TemporaryDirectory(prefix=f"vicap_{vid}_") as tmp:
        d = Path(tmp)
        thumb = None
        if row.get("thumbnail_key"):
            try:
                thumb = d / f"cover{_ext(row['thumbnail_key'])}"
                thumb.write_bytes(read_raw(row["thumbnail_key"]))
            except FileNotFoundError:  # như script gốc: không có ảnh bìa thì lấy frame đầu / slide đầu
                logger.warning("media {}: không có ảnh bìa {}, dùng frame đầu", vid, row["thumbnail_key"])
                thumb = None

        duration = None
        if row["media_type"] == "slideshow":
            slides = []
            for i, key in enumerate(row.get("image_keys") or []):
                p = d / f"slide_{i:03d}{_ext(key)}"
                p.write_bytes(read_raw(key))
                slides.append(p)
            frames: Frames = slideshow_frames(slides, thumb)
        else:
            video = d / "video.mp4"
            video.write_bytes(read_raw(row["media_key"]))
            frames = video_frames(video, thumb)
            mp3 = d / "audio.mp3"
            duration = extract_audio(video, mp3)
            write(paths.audio(vid), mp3.read_bytes(), "audio/mpeg")

        for k, img in enumerate(frames.images):
            write(paths.frame(vid, k), img, "image/jpeg")
    return {
        "video_id": vid, "media_type": row["media_type"], "n_source_images": frames.n_source,
        "thumb_source": frames.thumb_source, "has_audio": duration is not None, "audio_duration": duration,
        "batch_id": row["batch_id"], "created_at": created_at,
    }


def run_media(
    batch_ids: Sequence[str] | None = None,
    *,
    refresh: bool = False,
    workers: int = 4,
    processing_date: str | date | datetime | None = None,
    catalog: Any = None,
    fs: Any = None,
    now: Callable[[], datetime] = now_utc,
    silver_table: str = SILVER_VIDEOS.identifier,
    media_table: str = SILVER_MEDIA.identifier,
    quarantine_table: str = OPS_QUARANTINE.identifier,
) -> MediaStats:
    created_at = now()
    pdate = parse_date(processing_date or created_at)
    if catalog is None:
        from vicap.iceberg.catalog import load_lakehouse_catalog

        catalog = load_lakehouse_catalog()
    config = MinioConfig()
    fs = fs or s3_filesystem(config)

    silver = catalog.load_table(silver_table)
    videos = read_table(silver, In("batch_id", list(batch_ids)) if batch_ids else None).to_pylist()
    media = ensure_table(catalog, media_table, SILVER_MEDIA.schema)
    have = set() if refresh else set(read_table(media).column("video_id").to_pylist())
    todo = [r for r in videos if r["video_id"] not in have]
    stats = MediaStats(videos=len(videos), skipped=len(videos) - len(todo),
                       batches=sorted({r["batch_id"] for r in videos}))
    if not todo:
        logger.info(stats.summary())
        return stats

    def read_raw(key: str) -> bytes:
        return fs.cat_file(f"{config.raw_bucket}/{key}")

    def write(key: str, data: bytes, content_type: str) -> None:
        fs.pipe_file(f"{config.bucket}/{key}", data, ContentType=content_type)

    def one(row: dict) -> tuple[dict, dict | None, str | None]:
        try:
            return row, build_media(row, read_raw=read_raw, write=write, created_at=created_at), None
        except Exception as e:  # noqa: BLE001 — lỗi một video không dừng cả lô
            return row, None, f"{type(e).__name__}: {e}"[:500]

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        results = list(ex.map(one, todo))

    rows = [m for _, m, _ in results if m is not None]
    problems = [(r["batch_id"], r["video_id"], r["username"], err) for r, m, err in results if err]
    if rows:
        replace_by_keys(media, pa.Table.from_pylist(rows, schema=schema_to_pyarrow(media.schema())))
    resolved: dict[str, set[str]] = defaultdict(set)
    for m in rows:
        resolved[m["batch_id"]].add(m["video_id"])
    if problems or resolved:
        quarantine = ensure_table(catalog, quarantine_table, OPS_QUARANTINE.schema, OPS_QUARANTINE.partition_spec)
        quarantine_by_batch(quarantine, problems, stage=STAGE, processing_date=pdate, resolved=resolved)
    for b, v, _, err in problems:
        logger.warning("cách ly [{} {}] {}: {}", STAGE, b, v, err)
    stats.done, stats.errors = len(rows), len(problems)
    logger.info(stats.summary())
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-id", action="append", default=None, help="lô Silver (lặp được); mặc định cả bảng")
    ap.add_argument("--refresh", action="store_true", help="cắt lại cả video đã có trong silver.media")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    setup_logging()
    print(run_media(args.batch_id, refresh=args.refresh, workers=args.workers).summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
