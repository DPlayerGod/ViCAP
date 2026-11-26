"""Bước ``quality_check`` (cuối DAG, PLAN mục 5 và 9.3 chiều "Hợp lý"): báo cáo một lô → ``ops.quality_checks``.

Mỗi check một dòng ``(batch_id, check_name, value, severity, detail)``; ``severity`` là ``ok`` hoặc ``warn``.
Chỉ cảnh báo, không dừng DAG (lỗi dừng lô / cách ly dòng đã xử lý ở từng bước). Chạy lại thì thay cả lô.

| check                         | value                                             | warn khi                        |
|-------------------------------|---------------------------------------------------|---------------------------------|
| ``n_videos``                  | số video của lô ở ``silver.videos``               | = 0                             |
| ``missing_transcript_ratio``  | tỉ lệ video không có transcript ``ok``            | lệch > 0,2 so với lô trước      |
| ``missing_embedding``         | video ở Silver chưa có ``gold.features``          | > 0                             |
| ``missing_prediction``        | video có features chưa có dự đoán                 | > 0                             |
| ``quarantined``               | dòng ``ops.quarantine`` của lô (detail theo stage)| > 0                             |
| ``label_shift``               | khoảng cách total variation giữa phân bố nhãn dự đoán của lô và lô trước | > 0,3 |

"Lô trước" = ``batch_id`` lớn nhất nhỏ hơn lô này trong ``gold.video_predictions``.

    docker compose exec airflow-scheduler python -m vicap.pipelines.quality_check --batch-id 20261125-143434
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
from loguru import logger
from pyiceberg.expressions import EqualTo, In

from vicap.iceberg.reader import read_table
from vicap.iceberg.tables import (
    GOLD_FEATURES,
    GOLD_VIDEO_PREDICTIONS,
    OPS_QUALITY_CHECKS,
    OPS_QUARANTINE,
    SILVER_TRANSCRIPTS,
    SILVER_VIDEOS,
)
from vicap.iceberg.writer import ensure_table, write_table
from vicap.utils.dates import now_utc
from vicap.utils.log import setup_logging

TRANSCRIPT_SHIFT = 0.2
LABEL_SHIFT = 0.3


@dataclass(frozen=True)
class CheckRow:
    check_name: str
    value: float | None
    severity: str
    detail: str | None = None


def _ok(warn: bool) -> str:
    return "warn" if warn else "ok"


def _read(catalog: Any, identifier: str, row_filter=None) -> pa.Table | None:
    if not catalog.table_exists(identifier):
        return None
    return read_table(catalog.load_table(identifier), row_filter)


def _ids(table: pa.Table | None) -> set[str]:
    return set() if table is None else set(table.column("video_id").to_pylist())


def label_distribution(labels: Sequence[Sequence[str]]) -> dict[str, float]:
    """Tỉ lệ mỗi nhãn trên tổng số lần gán nhãn (đa nhãn: một video có thể góp nhiều nhãn)."""
    c = Counter(lb for row in labels for lb in row)
    total = sum(c.values())
    return {k: v / total for k, v in c.items()} if total else {}


def total_variation(p: dict[str, float], q: dict[str, float]) -> float:
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


def _missing_transcript_ratio(catalog: Any, ids: set[str], table: str) -> float | None:
    if not ids:
        return None
    t = _read(catalog, table, In("video_id", sorted(ids)))
    ok = 0 if t is None else sum(s == "ok" for s in t.column("transcript_status").to_pylist())
    return 1 - ok / len(ids)


def build_checks(
    batch_id: str,
    *,
    catalog: Any,
    silver_table: str = SILVER_VIDEOS.identifier,
    transcripts_table: str = SILVER_TRANSCRIPTS.identifier,
    features_table: str = GOLD_FEATURES.identifier,
    predictions_table: str = GOLD_VIDEO_PREDICTIONS.identifier,
    quarantine_table: str = OPS_QUARANTINE.identifier,
) -> list[CheckRow]:
    silver = _read(catalog, silver_table, EqualTo("batch_id", batch_id))
    ids = _ids(silver)
    rows = [CheckRow("n_videos", float(len(ids)), _ok(not ids))]

    preds = _read(catalog, predictions_table)
    prev = None
    if preds is not None:
        earlier = [b for b in set(preds.column("batch_id").to_pylist()) if b < batch_id]
        prev = max(earlier) if earlier else None

    ratio = _missing_transcript_ratio(catalog, ids, transcripts_table)
    prev_ratio = None
    if prev is not None:
        prev_silver = _read(catalog, silver_table, EqualTo("batch_id", prev))
        prev_ratio = _missing_transcript_ratio(catalog, _ids(prev_silver), transcripts_table)
    shifted = ratio is not None and prev_ratio is not None and abs(ratio - prev_ratio) > TRANSCRIPT_SHIFT
    rows.append(CheckRow("missing_transcript_ratio", ratio, _ok(shifted),
                         None if prev_ratio is None else f"lô trước {prev}: {prev_ratio:.3f}"))

    feats = _ids(_read(catalog, features_table, In("video_id", sorted(ids)))) if ids else set()
    no_emb = sorted(ids - feats)
    rows.append(CheckRow("missing_embedding", float(len(no_emb)), _ok(bool(no_emb)),
                         ",".join(no_emb[:20]) or None))

    batch_preds = None if preds is None else preds.filter(
        pc.is_in(preds.column("video_id"), pa.array(sorted(feats), pa.string())))
    no_pred = sorted(feats - _ids(batch_preds))
    rows.append(CheckRow("missing_prediction", float(len(no_pred)), _ok(bool(no_pred)),
                         ",".join(no_pred[:20]) or None))

    q = _read(catalog, quarantine_table, EqualTo("batch_id", batch_id))
    by_stage = Counter([] if q is None else q.column("stage").to_pylist())
    rows.append(CheckRow("quarantined", float(sum(by_stage.values())), _ok(bool(by_stage)),
                         json.dumps(dict(sorted(by_stage.items()))) if by_stage else None))

    if batch_preds is not None and batch_preds.num_rows and prev is not None:
        cur = label_distribution(batch_preds.column("labels").to_pylist())
        old = preds.filter(pc.equal(preds.column("batch_id"), prev))
        tv = total_variation(cur, label_distribution(old.column("labels").to_pylist()))
        rows.append(CheckRow("label_shift", tv, _ok(tv > LABEL_SHIFT), f"so với lô {prev}"))
    else:
        rows.append(CheckRow("label_shift", None, "ok", "chưa có lô trước"))
    return rows


def run_quality_check(
    batch_id: str,
    *,
    catalog: Any = None,
    now: Callable[[], datetime] = now_utc,
    checks_table: str = OPS_QUALITY_CHECKS.identifier,
    **tables: str,
) -> list[CheckRow]:
    if catalog is None:
        from vicap.iceberg.catalog import load_lakehouse_catalog

        catalog = load_lakehouse_catalog()
    rows = build_checks(batch_id, catalog=catalog, **tables)
    created_at = now()
    out = ensure_table(catalog, checks_table, OPS_QUALITY_CHECKS.schema)
    data = pa.Table.from_pylist(
        [{"batch_id": batch_id, **r.__dict__, "created_at": created_at} for r in rows],
        schema=pa.schema([("batch_id", pa.string()), ("check_name", pa.string()), ("value", pa.float64()),
                          ("severity", pa.string()), ("detail", pa.string()),
                          ("created_at", pa.timestamp("us", tz="UTC"))]),
    )
    write_table(out, data, mode="overwrite", overwrite_filter=EqualTo("batch_id", batch_id))
    for r in rows:
        log = logger.warning if r.severity == "warn" else logger.info
        log("quality_check [{}] {} = {} ({}){}", batch_id, r.check_name, r.value, r.severity,
            f" — {r.detail}" if r.detail else "")
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-id", required=True)
    args = ap.parse_args(argv)

    setup_logging()
    rows = run_quality_check(args.batch_id)
    print(f"quality_check {args.batch_id}: {sum(r.severity == 'warn' for r in rows)} cảnh báo / {len(rows)} check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
