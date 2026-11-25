from __future__ import annotations

from datetime import UTC, date, datetime


def now_utc() -> datetime:
    """Thời điểm hiện tại dạng UTC instant (timezone-aware).

    Chuẩn lưu trữ: mọi cột timestamp lưu UTC instant. Việc quy đổi sang
    giờ địa phương (nếu cần) làm ở tầng hiển thị/serving, không phải khi ghi.
    """
    return datetime.now(UTC)


def parse_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def format_date(value: str | date | datetime) -> str:
    return parse_date(value).isoformat()


def format_batch_id(logical_date: datetime) -> str:
    """`batch_id` = `YYYYMMDD-HHMMSS` (UTC) từ logical date của DAG run (PLAN mục 5)."""
    if logical_date.tzinfo is not None:
        logical_date = logical_date.astimezone(UTC)
    return logical_date.strftime("%Y%m%d-%H%M%S")
