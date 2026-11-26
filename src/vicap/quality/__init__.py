"""Tầng data-quality (PyArrow + NumPy) theo 6 chiều chất lượng dữ liệu và hai cấp kiểm tra.

- ``result``  — ``Dimension``, ``Severity``, ``CheckOutcome``, ``ValidationResult``.
- ``checks``  — config + hàm ``check(table, config)``, ``run_suite`` (cấp lô), ``row_mask`` / ``split_valid``
  (cách ly dòng).
- ``suites``  — suite cho từng bảng video, ``validate(table, suite) -> SuiteResult``.
- ``quarantine`` — ghi dòng bị cách ly vào bảng ``ops.quarantine``.
"""

from __future__ import annotations

from vicap.quality.checks import (
    Check,
    ColumnRelation,
    Covers,
    ForeignKey,
    InRange,
    InSet,
    ListLength,
    MatchesDate,
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
    row_mask,
    run_check,
    run_suite,
    split_valid,
)
from vicap.quality.quarantine import (
    merge_quarantine_rows,
    quarantine_by_batch,
    quarantine_rows,
    write_quarantine_rows,
)
from vicap.quality.result import CheckOutcome, Dimension, Severity, ValidationResult
from vicap.quality.suites import (
    SuiteResult,
    TableSuite,
    bronze_videos_suite,
    gold_features_suite,
    gold_predictions_suite,
    gold_profiles_suite,
    silver_transcripts_suite,
    silver_videos_suite,
    validate,
)

__all__ = [
    "Dimension", "Severity", "CheckOutcome", "ValidationResult",
    "Check", "RequiredColumns", "NotNull", "Covers", "Unique", "InRange", "Positive", "InSet",
    "ListLength", "VectorDtype", "VectorShape", "NonZeroVector", "VectorRange", "ColumnRelation",
    "ForeignKey", "MeanShift", "NonEmpty", "MatchesDate",
    "run_check", "run_suite", "row_mask", "split_valid",
    "TableSuite", "SuiteResult", "validate", "quarantine_rows", "write_quarantine_rows", "merge_quarantine_rows",
    "quarantine_by_batch",
    "bronze_videos_suite", "silver_videos_suite", "silver_transcripts_suite",
    "gold_features_suite", "gold_predictions_suite", "gold_profiles_suite",
]
