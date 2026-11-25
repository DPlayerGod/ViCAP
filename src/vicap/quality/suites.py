"""Suite kiểm định cho từng bảng video (PLAN mục 9.3), khai báo bằng config của ``checks.py``.

Mỗi bảng có hai danh sách:

- ``batch_checks`` — cấp lô: ERROR thì dừng lô (``quarantine_and_raise``), WARN thì chỉ ghi nhận.
- ``row_checks``   — cách ly dòng: ``split_valid`` tách dòng lỗi, phần còn lại chạy tiếp.

Cách dùng trong pipeline::

    res = validate(table, gold_features_suite(silver=silver_ids))
    res.batch.quarantine_and_raise(table, domain="gold_features", processing_date=ds, batch_id=batch_id)
    write_quarantine_rows(ops_table, res.bad, stage="gold_features", batch_id=batch_id, processing_date=ds)
    replace_by_keys(features_table, res.ok)
"""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from vicap.config import ModelConfig
from vicap.quality.checks import (
    Check,
    ColumnRelation,
    Covers,
    ForeignKey,
    InSet,
    ListLength,
    MeanShift,
    NonEmpty,
    NonZeroVector,
    NotNull,
    Positive,
    RequiredColumns,
    Unique,
    VectorDtype,
    VectorRange,
    VectorShape,
    run_check,
    run_suite,
    split_valid,
)
from vicap.quality.result import Severity, ValidationResult

MEDIA_TYPES = ("video", "slideshow")
CONFIDENCE_LEVELS = ("low", "medium", "high")
N_PROFILE_LABELS = 19  # 20 nhãn bỏ Others


@dataclass(frozen=True)
class TableSuite:
    name: str
    required: tuple[str, ...]
    batch_checks: tuple[Check, ...] = ()
    row_checks: tuple[Check, ...] = ()


@dataclass(frozen=True)
class SuiteResult:
    batch: ValidationResult
    ok: pa.Table
    bad: pa.Table  # có cột ``_errors``


def validate(table: pa.Table, suite: TableSuite) -> SuiteResult:
    """Chạy cấp lô trước; nếu lô hợp lệ mới tách dòng.

    Lô không hợp lệ: ``ok`` rỗng, ``bad`` là cả lô (``_errors`` = lỗi cấp lô), để caller gọi
    ``batch.quarantine_and_raise``.
    """
    gate = run_check(table, RequiredColumns(suite.required))
    if not gate.passed:
        batch = ValidationResult(False, (f"{suite.name}: {gate.message}",), (), (gate,))
    else:
        batch = run_suite(table, suite.batch_checks, suite.name)
    if not batch.is_valid:
        errors = pa.array([list(batch.errors)] * table.num_rows, type=pa.list_(pa.string()))
        return SuiteResult(batch, table.slice(0, 0), table.append_column("_errors", errors))
    ok, bad = split_valid(table, suite.row_checks)
    return SuiteResult(batch, ok, bad)


def _optional(check: Check, reference: pa.Table | None) -> tuple[Check, ...]:
    return (check,) if reference is not None else ()


# ────────────────────────────────── BRONZE ─────────────────────────────────


def bronze_videos_suite() -> TableSuite:
    return TableSuite(
        name="bronze_videos_raw",
        required=("video_id", "username", "media_type", "media_key", "thumbnail_key",
                  "posted_at", "crawl_ts", "batch_id"),
        batch_checks=(
            NonEmpty(),
            ColumnRelation("posted_at", "<=", "crawl_ts", severity=Severity.WARN),
        ),
        row_checks=(
            NotNull(("video_id", "username", "media_key", "thumbnail_key")),
            InSet("media_type", MEDIA_TYPES),
        ),
    )


# ────────────────────────────────── SILVER ─────────────────────────────────


def silver_videos_suite() -> TableSuite:
    return TableSuite(
        name="silver_videos",
        required=("video_id", "username", "media_type", "duration", "caption", "hashtags", "batch_id"),
        batch_checks=(Unique(("video_id",)),),
        row_checks=(
            NotNull(("video_id", "username", "media_type")),
            InSet("media_type", MEDIA_TYPES),
            Positive(("duration",)),
        ),
    )


def silver_transcripts_suite(previous: pa.Table | None = None, max_shift: float = 0.2) -> TableSuite:
    """``previous``: lô trước, để cảnh báo khi tỉ lệ có transcript đổi nhiều."""
    return TableSuite(
        name="silver_transcripts",
        required=("video_id", "transcript_raw", "has_transcript", "batch_id"),
        batch_checks=(Unique(("video_id",)), *_optional(MeanShift("has_transcript", previous, max_shift), previous)),
        row_checks=(NotNull(("video_id", "has_transcript")),),
    )


# ─────────────────────────────────── GOLD ──────────────────────────────────


def gold_features_suite(silver: pa.Table | None = None, model: ModelConfig | None = None) -> TableSuite:
    """Luật embedding ở PLAN mục 4.1.

    ``silver``: các video Silver của lô (cột ``video_id``). Video Silver không có embedding không
    nằm trong bảng này nên không cách ly được ở đây; check ``Covers`` chỉ cảnh báo, còn bước
    ``embed`` tự ghi video tính embedding lỗi vào quarantine.
    """
    m = model or ModelConfig()
    return TableSuite(
        name="gold_features",
        required=("video_id", "text_emb", "image_emb", "text_encoder", "image_encoder", "batch_id"),
        batch_checks=(
            Unique(("video_id",)),
            VectorDtype("text_emb", depth=2),
            VectorDtype("image_emb", depth=2),
            *_optional(Covers("video_id", silver, "video_id"), silver),
        ),
        row_checks=(
            NotNull(("video_id",)),
            VectorShape("text_emb", (m.n_text, m.text_dim)),
            VectorShape("image_emb", (m.n_images, m.image_dim)),
            NonZeroVector("text_emb", rows=(0,)),  # text_emb[1] (transcript) được phép bằng 0
            NonZeroVector("image_emb"),
        ),
    )


def gold_predictions_suite(
    features: pa.Table | None = None,
    previous: pa.Table | None = None,
    model: ModelConfig | None = None,
    max_shift: float = 0.1,
) -> TableSuite:
    """``features``: để kiểm mỗi dự đoán có video ở ``gold.features``; ``previous``: lô trước."""
    m = model or ModelConfig()
    return TableSuite(
        name="gold_video_predictions",
        required=("video_id", "username", "probs", "labels", "threshold", "model_run_id", "batch_id"),
        batch_checks=(
            Unique(("video_id",)),
            VectorDtype("probs", depth=1),
            *_optional(MeanShift("probs", previous, max_shift), previous),
        ),
        row_checks=(
            NotNull(("video_id", "username", "model_run_id")),
            VectorShape("probs", (m.n_labels,)),
            VectorRange("probs", 0.0, 1.0),
            ListLength("labels", min_len=1 if m.at_least_one else 0),
            *_optional(ForeignKey("video_id", features, "video_id"), features),
        ),
    )


def gold_profiles_suite(creators: pa.Table | None = None) -> TableSuite:
    """``creators``: bảng creator (cột ``username``) để kiểm khoá ngoại."""
    return TableSuite(
        name="gold_creator_profiles",
        required=("username", "n_videos", "label_ratios", "confidence", "batch_id"),
        batch_checks=(Unique(("username",)), VectorDtype("label_ratios", depth=1)),
        row_checks=(
            NotNull(("username", "confidence")),
            Positive(("n_videos",)),
            VectorShape("label_ratios", (N_PROFILE_LABELS,)),
            VectorRange("label_ratios", 0.0, 1.0),
            InSet("confidence", CONFIDENCE_LEVELS),
            *_optional(ForeignKey("username", creators, "username"), creators),
        ),
    )
