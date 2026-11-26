"""Bước ``asr`` (PLAN mục 5): ``silver.media`` + ``derived/audio/`` → ``silver.transcripts`` (khoá ``video_id``).

Trạng thái như notebook (``vicap.features.asr``): slideshow → ``missing``; audio < 1 s → ``empty_or_music``
(không chạy model); còn lại chạy PhoWhisper. Video đã có transcript ``ok`` / ``empty_or_music`` / ``missing``
cùng model thì bỏ qua (``--refresh`` để chạy lại); ``error`` thì chạy lại. Lỗi → ``ops.quarantine`` stage ``asr``.

Cần GPU: chạy trong gpu-worker (Airflow gọi bằng DockerOperator). Chạy tay::

    docker compose --profile "*" run --rm --no-deps gpu-worker python -m vicap.pipelines.asr
"""

from __future__ import annotations

import argparse
import tempfile
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
from loguru import logger
from pyiceberg.expressions import In
from pyiceberg.io.pyarrow import schema_to_pyarrow

from vicap.config import MinioConfig
from vicap.features.asr import ASR_MODEL, FINAL_STATUS, status_of, transcribe
from vicap.features.media import MIN_AUDIO_SEC
from vicap.iceberg.reader import read_table
from vicap.iceberg.tables import OPS_QUARANTINE, SILVER_MEDIA, SILVER_TRANSCRIPTS, SILVER_VIDEOS
from vicap.iceberg.writer import ensure_table, replace_by_keys
from vicap.quality.quarantine import quarantine_by_batch
from vicap.storage import paths
from vicap.storage.objects import s3_filesystem
from vicap.utils.dates import now_utc, parse_date
from vicap.utils.log import setup_logging

STAGE = "asr"


@dataclass
class AsrStats:
    videos: int = 0
    no_media: int = 0  # chưa qua bước media
    skipped: int = 0
    counts: dict[str, int] = field(default_factory=dict)  # status → số video ghi lần này

    def summary(self) -> str:
        done = ", ".join(f"{k} {v}" for k, v in sorted(self.counts.items())) or "0"
        return (f"asr: {self.videos} video; ghi {done}; bỏ qua {self.skipped}, "
                f"chưa có media {self.no_media}")


def run_asr(
    batch_ids: Sequence[str] | None = None,
    *,
    refresh: bool = False,
    model: str = ASR_MODEL,
    asr: Callable[..., Any] | None = None,
    processing_date: str | date | datetime | None = None,
    catalog: Any = None,
    fs: Any = None,
    now: Callable[[], datetime] = now_utc,
    silver_table: str = SILVER_VIDEOS.identifier,
    media_table: str = SILVER_MEDIA.identifier,
    transcripts_table: str = SILVER_TRANSCRIPTS.identifier,
    quarantine_table: str = OPS_QUARANTINE.identifier,
) -> AsrStats:
    created_at = now()
    pdate = parse_date(processing_date or created_at)
    if catalog is None:
        from vicap.iceberg.catalog import load_lakehouse_catalog

        catalog = load_lakehouse_catalog()
    config = MinioConfig()
    fs = fs or s3_filesystem(config)

    flt = In("batch_id", list(batch_ids)) if batch_ids else None
    videos = {r["video_id"]: r for r in read_table(catalog.load_table(silver_table), flt).to_pylist()}
    media_tbl = ensure_table(catalog, media_table, SILVER_MEDIA.schema)
    ids = list(videos)
    media = {r["video_id"]: r for r in read_table(media_tbl, In("video_id", ids)).to_pylist()} if ids else {}
    out_tbl = ensure_table(catalog, transcripts_table, SILVER_TRANSCRIPTS.schema)
    done: set[str] = set()
    if ids and not refresh:
        old = read_table(out_tbl, In("video_id", ids)).to_pylist()
        done = {r["video_id"] for r in old if r["transcript_status"] in FINAL_STATUS and r["asr_model"] == model}

    stats = AsrStats(videos=len(videos))
    rows: dict[str, tuple[str, str, str | None]] = {}  # video_id → (text, status, lỗi)
    runnable: list[str] = []
    for vid in ids:
        if vid in done:
            stats.skipped += 1
        elif vid not in media:
            stats.no_media += 1
        elif not media[vid]["has_audio"]:
            rows[vid] = ("", "missing", None)
        elif (media[vid]["audio_duration"] or 0) < MIN_AUDIO_SEC:
            rows[vid] = ("", "empty_or_music", None)
        else:
            runnable.append(vid)
    if stats.no_media:
        logger.warning("asr: {} video chưa có trong {} (chạy bước media trước)", stats.no_media, media_table)

    if runnable:
        if asr is None:
            from vicap.features.asr import load_asr

            asr = load_asr(model)
        with tempfile.TemporaryDirectory(prefix="vicap_asr_") as tmp:
            files = []
            for vid in runnable:
                p = Path(tmp) / paths.audio_name(vid)
                p.write_bytes(fs.cat_file(f"{config.bucket}/{paths.audio(vid)}"))
                files.append(str(p))
            for vid, (text, err) in zip(runnable, transcribe(asr, files), strict=True):
                rows[vid] = (text, status_of(text, err), err)

    if rows:
        table = pa.Table.from_pylist([
            {"video_id": vid, "transcript_raw": text, "transcript_status": status,
             "has_transcript": status == "ok", "asr_model": model, "batch_id": videos[vid]["batch_id"],
             "created_at": created_at}
            for vid, (text, status, _) in rows.items()
        ], schema=schema_to_pyarrow(out_tbl.schema()))
        replace_by_keys(out_tbl, table)
    for _, status, _ in rows.values():
        stats.counts[status] = stats.counts.get(status, 0) + 1

    problems = [(videos[v]["batch_id"], v, videos[v]["username"], err) for v, (_, _, err) in rows.items() if err]
    resolved: dict[str, set[str]] = defaultdict(set)
    for v, (_, _, err) in rows.items():
        if not err:
            resolved[videos[v]["batch_id"]].add(v)
    if problems or resolved:
        quarantine = ensure_table(catalog, quarantine_table, OPS_QUARANTINE.schema, OPS_QUARANTINE.partition_spec)
        quarantine_by_batch(quarantine, problems, stage=STAGE, processing_date=pdate, resolved=resolved)
    for b, v, _, err in problems:
        logger.warning("cách ly [{} {}] {}: {}", STAGE, b, v, err)
    logger.info(stats.summary())
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-id", action="append", default=None, help="lô Silver (lặp được); mặc định cả bảng")
    ap.add_argument("--refresh", action="store_true", help="chạy lại cả video đã có transcript")
    ap.add_argument("--model", default=ASR_MODEL)
    args = ap.parse_args(argv)

    setup_logging()
    print(run_asr(args.batch_id, refresh=args.refresh, model=args.model).summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
