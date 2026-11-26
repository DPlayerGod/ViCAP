"""Bước ``embed`` (PLAN mục 4.1, 5): Silver + ``derived/frames/`` → ``gold.features`` (khoá ``video_id``).

- ``text_emb[0]``: PhoBERT(``caption + " " + " ".join(hashtags)``) (``vicap.silver.text.caption_hashtag``).
- ``text_emb[1]``: PhoBERT chunked(``transcript_raw``); không có transcript → vector 0.
- ``image_emb``: SigLIP2 của 8 ảnh ``derived/frames/{video_id}_{k}.jpg``, giữ từng ảnh (8 × 768).

Chỉ nhận video đã có ``silver.media`` và transcript ở trạng thái cuối (ok / empty_or_music / missing); video
đã có embedding cùng encoder thì bỏ qua (``--refresh`` để tính lại). PhoBERT rồi SigLIP2 chạy lần lượt, giải
phóng VRAM sau mỗi model. Kiểm định ``gold_features_suite``: dòng lỗi → ``ops.quarantine`` stage
``gold_features``, dòng hợp lệ → ``replace_by_keys``.

Cần GPU: chạy trong gpu-worker (``make embed``).
"""

from __future__ import annotations

import argparse
import gc
import io
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pyarrow as pa
from loguru import logger
from pyiceberg.expressions import In
from pyiceberg.io.pyarrow import schema_to_pyarrow

from vicap.config import MinioConfig, ModelConfig
from vicap.features.asr import FINAL_STATUS
from vicap.iceberg.reader import read_table
from vicap.iceberg.tables import GOLD_FEATURES, OPS_QUARANTINE, SILVER_MEDIA, SILVER_TRANSCRIPTS, SILVER_VIDEOS
from vicap.iceberg.writer import ensure_table, replace_by_keys
from vicap.quality import gold_features_suite, validate
from vicap.quality.quarantine import quarantine_by_batch
from vicap.silver.text import caption_hashtag
from vicap.storage import paths
from vicap.storage.objects import s3_filesystem
from vicap.utils.dates import now_utc, parse_date
from vicap.utils.log import setup_logging

STAGE = "gold_features"


@dataclass
class EmbedStats:
    videos: int = 0
    not_ready: int = 0  # chưa có media / transcript cuối
    skipped: int = 0  # đã có embedding cùng encoder
    written: int = 0
    quarantined: int = 0

    def summary(self) -> str:
        return (f"embed: {self.videos} video; ghi {self.written}, cách ly {self.quarantined}, "
                f"bỏ qua {self.skipped}, chưa đủ dữ liệu {self.not_ready}")


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def select_ready(videos: dict[str, dict], media: dict[str, dict], transcripts: dict[str, dict],
                 done: set[str]) -> tuple[list[str], int, int]:
    """``(video cần tính, số chưa đủ dữ liệu, số bỏ qua)``."""
    ready, not_ready, skipped = [], 0, 0
    for vid in videos:
        if vid in done:
            skipped += 1
        elif vid in media and transcripts.get(vid, {}).get("transcript_status") in FINAL_STATUS:
            ready.append(vid)
        else:
            not_ready += 1
    return ready, not_ready, skipped


def to_matrix_column(x: np.ndarray) -> pa.Array:
    """``(n, k, d)`` float32 → ``list<list<float>>`` (``FLOAT_MATRIX`` của ``gold.features``)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    n, k, d = x.shape
    inner = pa.ListArray.from_arrays(pa.array(np.arange(0, n * k * d + 1, d, dtype=np.int32)),
                                     pa.array(x.reshape(-1)))
    return pa.ListArray.from_arrays(pa.array(np.arange(0, n * k + 1, k, dtype=np.int32)), inner)


def run_embed(
    batch_ids: Sequence[str] | None = None,
    *,
    refresh: bool = False,
    model: ModelConfig | None = None,
    text_encoder: Any = None,
    image_encoder: Any = None,
    processing_date: str | date | datetime | None = None,
    catalog: Any = None,
    fs: Any = None,
    now: Callable[[], datetime] = now_utc,
    silver_table: str = SILVER_VIDEOS.identifier,
    media_table: str = SILVER_MEDIA.identifier,
    transcripts_table: str = SILVER_TRANSCRIPTS.identifier,
    features_table: str = GOLD_FEATURES.identifier,
    quarantine_table: str = OPS_QUARANTINE.identifier,
) -> EmbedStats:
    m = model or ModelConfig()
    created_at = now()
    pdate = parse_date(processing_date or created_at)
    if catalog is None:
        from vicap.iceberg.catalog import load_lakehouse_catalog

        catalog = load_lakehouse_catalog()
    config = MinioConfig()
    fs = fs or s3_filesystem(config)

    flt = In("batch_id", list(batch_ids)) if batch_ids else None
    videos = {r["video_id"]: r for r in read_table(catalog.load_table(silver_table), flt).to_pylist()}
    ids = list(videos)
    stats = EmbedStats(videos=len(ids))
    if not ids:
        logger.info(stats.summary())
        return stats

    def by_id(identifier: str, table_def) -> dict[str, dict]:
        t = ensure_table(catalog, identifier, table_def.schema, table_def.partition_spec)
        return {r["video_id"]: r for r in read_table(t, In("video_id", ids)).to_pylist()}

    media = by_id(media_table, SILVER_MEDIA)
    transcripts = by_id(transcripts_table, SILVER_TRANSCRIPTS)
    features = ensure_table(catalog, features_table, GOLD_FEATURES.schema)
    done: set[str] = set()
    if not refresh:
        old = read_table(features, In("video_id", ids)).select(["video_id", "text_encoder", "image_encoder"])
        done = {r["video_id"] for r in old.to_pylist()
                if r["text_encoder"] == m.text_encoder and r["image_encoder"] == m.image_encoder}
    todo, stats.not_ready, stats.skipped = select_ready(videos, media, transcripts, done)
    if stats.not_ready:
        logger.warning("embed: {} video chưa có media / transcript (chạy media, asr trước)", stats.not_ready)
    if not todo:
        logger.info(stats.summary())
        return stats

    # ── text: PhoBERT ──
    caps = [caption_hashtag(videos[v].get("caption"), videos[v].get("hashtags")) for v in todo]
    asr = [transcripts[v].get("transcript_raw") or "" for v in todo]
    if text_encoder is None:
        from vicap.features.text import PhoBert

        text_encoder = PhoBert(m.text_encoder)
    text_emb = np.stack([text_encoder.encode_texts(caps), text_encoder.encode_chunked(asr)], axis=1)
    text_encoder = None  # bỏ tham chiếu để giải phóng VRAM trước khi nạp SigLIP2
    _free_gpu()

    # ── ảnh: SigLIP2 ──
    def frames_of(vid: str) -> list[Any]:
        out = []
        for k in range(m.n_images):
            try:
                out.append(io.BytesIO(fs.cat_file(f"{config.bucket}/{paths.frame(vid, k)}")))
            except FileNotFoundError:
                out.append(None)  # vector 0 → NonZeroVector cách ly
        return out

    if image_encoder is None:
        from vicap.features.image import SigLip

        image_encoder = SigLip(m.image_encoder)
    image_emb = image_encoder.encode([frames_of(v) for v in todo])
    image_encoder = None
    _free_gpu()

    schema = schema_to_pyarrow(features.schema())
    table = pa.table({
        "video_id": pa.array(todo, pa.string()),
        "text_emb": to_matrix_column(text_emb),
        "image_emb": to_matrix_column(image_emb),
        "n_source_images": pa.array([media[v]["n_source_images"] for v in todo], pa.int32()),
        "text_encoder": pa.array([m.text_encoder] * len(todo), pa.string()),
        "image_encoder": pa.array([m.image_encoder] * len(todo), pa.string()),
        "batch_id": pa.array([videos[v]["batch_id"] for v in todo], pa.string()),
        "created_at": pa.array([created_at] * len(todo), pa.timestamp("us", tz="UTC")),
    }).cast(schema)

    res = validate(table, gold_features_suite(silver=pa.table({"video_id": pa.array(todo, pa.string())}), model=m))
    res.batch.quarantine_and_raise(table, domain=STAGE, processing_date=pdate.isoformat(),
                                   batch_id=",".join(sorted({videos[v]["batch_id"] for v in todo})))
    ok = res.ok.drop_columns([c for c in res.ok.column_names if c not in schema.names])
    if ok.num_rows:
        replace_by_keys(features, ok.cast(schema))
    bad = res.bad.select(["video_id", "batch_id", "_errors"]).to_pylist()
    problems = [(r["batch_id"], r["video_id"], videos[r["video_id"]]["username"], "; ".join(r["_errors"]))
                for r in bad]
    resolved: dict[str, set[str]] = defaultdict(set)
    for v, b in zip(ok.column("video_id").to_pylist(), ok.column("batch_id").to_pylist(), strict=True):
        resolved[b].add(v)
    quarantine = ensure_table(catalog, quarantine_table, OPS_QUARANTINE.schema, OPS_QUARANTINE.partition_spec)
    quarantine_by_batch(quarantine, problems, stage=STAGE, processing_date=pdate, resolved=resolved)
    for b, v, _, err in problems:
        logger.warning("cách ly [{} {}] {}: {}", STAGE, b, v, err)
    stats.written, stats.quarantined = ok.num_rows, len(problems)
    logger.info(stats.summary())
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-id", action="append", default=None, help="lô Silver (lặp được); mặc định cả bảng")
    ap.add_argument("--refresh", action="store_true", help="tính lại cả video đã có embedding")
    args = ap.parse_args(argv)

    setup_logging()
    print(run_embed(args.batch_id, refresh=args.refresh).summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
