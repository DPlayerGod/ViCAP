from __future__ import annotations

from datetime import datetime, timedelta, timezone

from vicap.utils.dates import format_batch_id, format_date


def test_batch_id_is_utc() -> None:
    vn = timezone(timedelta(hours=7))
    assert format_batch_id(datetime(2026, 11, 25, 7, 0, 5, tzinfo=vn)) == "20261125-000005"
    assert format_batch_id(datetime(2026, 11, 25, 0, 0, 5)) == "20261125-000005"


def test_format_date() -> None:
    assert format_date("2026-11-25") == "2026-11-25"
