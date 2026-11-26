from __future__ import annotations

from datetime import date

import pytest

from vicap.pipelines.daily import GPU_STEPS, STEPS, run_daily, select_steps


def test_steps_order_matches_plan() -> None:
    assert list(STEPS) == ["crawl", "bronze", "silver_clean", "media", "asr", "embed", "predict", "profile",
                           "load_clickhouse", "quality_check"]
    assert set(GPU_STEPS) <= set(STEPS)


def test_select_steps() -> None:
    assert select_steps(only=["profile", "crawl"]) == ["crawl", "profile"]  # luôn theo thứ tự DAG
    assert select_steps(skip=["crawl"])[0] == "bronze"
    with pytest.raises(ValueError, match="nope"):
        select_steps(skip=["nope"])


def test_run_daily_calls_in_order_and_stops_on_error() -> None:
    calls: list[tuple[str, str, date]] = []

    def step(name):
        def run(batch_id, pdate):
            calls.append((name, batch_id, pdate))
            if name == "embed":
                raise RuntimeError("lỗi cấp lô")
            return name
        return run

    fakes = {n: step(n) for n in STEPS}
    out = run_daily("b1", processing_date="2026-11-26", only=["silver_clean", "profile"], steps=fakes)
    assert out == {"silver_clean": "silver_clean", "profile": "profile"}
    assert calls == [("silver_clean", "b1", date(2026, 11, 26)), ("profile", "b1", date(2026, 11, 26))]

    calls.clear()
    with pytest.raises(RuntimeError):
        run_daily("b1", processing_date="2026-11-26", skip=["crawl"], steps=fakes)
    assert [c[0] for c in calls] == ["bronze", "silver_clean", "media", "asr", "embed"]  # dừng ở embed
