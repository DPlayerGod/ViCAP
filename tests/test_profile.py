"""Phần thuần Python của bước profile (phần Spark: test_integration_profile.py)."""

from __future__ import annotations

import pytest

from vicap.config import ModelConfig
from vicap.gold.profile import COMPARE_COLUMNS, HISTORY_COLUMNS, PROFILE_COLUMNS, confidence_level
from vicap.quality.suites import N_PROFILE_LABELS


@pytest.mark.parametrize(("n", "level"), [(1, "low"), (4, "low"), (5, "medium"), (9, "medium"), (10, "high")])
def test_confidence_level(n: int, level: str) -> None:
    assert confidence_level(n) == level


def test_profile_labels_follow_checkpoint_order() -> None:
    m = ModelConfig()
    assert len(m.labels) == m.n_labels and m.labels == tuple(sorted(m.labels))
    assert len(m.profile_labels) == N_PROFILE_LABELS and "Others" not in m.profile_labels
    assert m.profile_labels == m.labels[:17] + m.labels[18:]


def test_columns() -> None:
    assert HISTORY_COLUMNS == ("snapshot_date", *PROFILE_COLUMNS)
    assert set(COMPARE_COLUMNS) <= set(PROFILE_COLUMNS) and "updated_at" not in COMPARE_COLUMNS
