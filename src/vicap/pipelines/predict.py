"""Bước ``predict`` (PLAN mục 4.3, 5): ``gold.features`` → ``gold.video_predictions`` (khoá ``video_id``).

Nạp ``models:/kltn-p1@champion`` (``vicap.model.predict.load_p1``), dựng input bằng ``build_p1_inputs``, gán nhãn
bằng ``decide`` của checkpoint (``P >= 0,75``, luôn có ít nhất một nhãn). Mỗi dòng ghi kèm tên, phiên bản,
``model_run_id`` của MLflow và ngưỡng. Video đã có dự đoán của cùng phiên bản thì bỏ qua (``--refresh`` để chạy
lại); đổi alias sang phiên bản khác thì mọi video được dự đoán lại. Kiểm định ``gold_predictions_suite``: dòng lỗi
→ ``ops.quarantine`` stage ``gold_predictions``.

Chạy trong gpu-worker (``make predict``); model nhỏ nên CPU cũng được.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from loguru import logger
from pyiceberg.expressions import In
from pyiceberg.io.pyarrow import schema_to_pyarrow

from vicap.iceberg.reader import read_table
from vicap.iceberg.tables import GOLD_FEATURES, GOLD_VIDEO_PREDICTIONS, OPS_QUARANTINE, SILVER_VIDEOS
from vicap.iceberg.writer import ensure_table, replace_by_keys
from vicap.model.predict import P1, load_p1, predict_batch
from vicap.quality import gold_predictions_suite, validate
from vicap.quality.quarantine import quarantine_by_batch
from vicap.utils.dates import now_utc, parse_date
from vicap.utils.log import setup_logging

STAGE = "gold_predictions"


@dataclass
class PredictStats:
    videos: int = 0
    skipped: int = 0
    written: int = 0
    quarantined: int = 0
    model: str = "-"

    def summary(self) -> str:
        return (f"predict ({self.model}): {self.videos} video; ghi {self.written}, cách ly {self.quarantined}, "
                f"bỏ qua {self.skipped}")


def run_predict(
    batch_ids: Sequence[str] | None = None,
    *,
    refresh: bool = False,
    p1: P1 | None = None,
    processing_date: str | date | datetime | None = None,
    catalog: Any = None,
    now: Callable[[], datetime] = now_utc,
    silver_table: str = SILVER_VIDEOS.identifier,
    features_table: str = GOLD_FEATURES.identifier,
    predictions_table: str = GOLD_VIDEO_PREDICTIONS.identifier,
    quarantine_table: str = OPS_QUARANTINE.identifier,
) -> PredictStats:
    predicted_at = now()
    pdate = parse_date(processing_date or predicted_at)
    if catalog is None:
        from vicap.iceberg.catalog import load_lakehouse_catalog

        catalog = load_lakehouse_catalog()

    flt = In("batch_id", list(batch_ids)) if batch_ids else None
    features = read_table(catalog.load_table(features_table), flt)
    stats = PredictStats(videos=features.num_rows)
    if not features.num_rows:
        logger.info(stats.summary())
        return stats
    ids = features.column("video_id").to_pylist()

    p1 = p1 or load_p1()
    stats.model = f"{p1.name} v{p1.version}"
    out_tbl = ensure_table(catalog, predictions_table, GOLD_VIDEO_PREDICTIONS.schema)
    if not refresh:
        old = read_table(out_tbl, In("video_id", ids)).select(["video_id", "model_name", "model_version"])
        done = {r["video_id"] for r in old.to_pylist()
                if (r["model_name"], r["model_version"]) == (p1.name, p1.version)}
        if done:
            features = features.filter(pc.invert(pc.is_in(features.column("video_id"),
                                                                           pa.array(sorted(done)))))
        stats.skipped = len(done)
    if not features.num_rows:
        logger.info(stats.summary())
        return stats

    ids = features.column("video_id").to_pylist()
    users = {r["video_id"]: r["username"] for r in
             read_table(catalog.load_table(silver_table), In("video_id", ids)).select(["video_id", "username"])
             .to_pylist()}
    text_emb = np.array(features.column("text_emb").to_pylist(), dtype=np.float32)
    image_emb = np.array(features.column("image_emb").to_pylist(), dtype=np.float32)
    probs, labels = predict_batch(p1, text_emb, image_emb)

    n = len(ids)
    table = pa.table({
        "video_id": pa.array(ids, pa.string()),
        "username": pa.array([users.get(v) for v in ids], pa.string()),
        "probs": pa.array([p.tolist() for p in probs], pa.list_(pa.float32())),
        "labels": pa.array(labels, pa.list_(pa.string())),
        "threshold": pa.array([p1.threshold] * n, pa.float32()),
        "model_name": pa.array([p1.name] * n, pa.string()),
        "model_version": pa.array([p1.version] * n, pa.string()),
        "model_run_id": pa.array([p1.run_id] * n, pa.string()),
        "batch_id": features.column("batch_id").combine_chunks().cast(pa.string()),
        "predicted_at": pa.array([predicted_at] * n, pa.timestamp("us", tz="UTC")),
    })
    res = validate(table, gold_predictions_suite(features=features.select(["video_id"])))
    res.batch.quarantine_and_raise(table, domain=STAGE, processing_date=pdate.isoformat(),
                                   batch_id=",".join(sorted(set(table.column("batch_id").to_pylist()))))
    schema = schema_to_pyarrow(out_tbl.schema())
    ok = res.ok.select(schema.names).cast(schema)
    if ok.num_rows:
        replace_by_keys(out_tbl, ok)

    problems = [(r["batch_id"], r["video_id"], r["username"], "; ".join(r["_errors"]))
                for r in res.bad.select(["batch_id", "video_id", "username", "_errors"]).to_pylist()]
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
    ap.add_argument("--batch-id", action="append", default=None, help="lô (lặp được); mặc định cả gold.features")
    ap.add_argument("--refresh", action="store_true", help="dự đoán lại cả video đã có dự đoán của phiên bản này")
    args = ap.parse_args(argv)

    setup_logging()
    print(run_predict(args.batch_id, refresh=args.refresh).summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
