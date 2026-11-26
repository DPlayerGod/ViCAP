"""``run_daily(batch_id)``: chạy lần lượt mọi bước của DAG ``daily_pipeline`` trong một tiến trình (PLAN mục 5).

Thứ tự ``STEPS`` giống DAG (``dags/daily_pipeline.py``). Mỗi bước nhận ``batch_id`` và ngày xử lý; các bước
sau ``bronze`` chỉ đụng tới video của lô đó (profile tính lại creator có video trong lô). Dùng để chạy đầu–cuối
không cần Airflow và cho test; môi trường phải có đủ phụ thuộc của các bước được chọn (Spark cho
``silver_clean`` / ``profile``, GPU + torch cho ``asr`` / ``embed`` / ``predict``), nên thường chạy một đoạn::

    python -m vicap.pipelines.daily --batch-id 20261126-000000 --only silver_clean,profile,load_clickhouse
    python -m vicap.pipelines.daily --batch-id 20261126-000000 --skip crawl
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import date
from typing import Any

from loguru import logger

from vicap.utils.dates import now_utc, parse_date
from vicap.utils.log import setup_logging


def _crawl(batch_id: str, pdate: date) -> Any:
    from vicap.pipelines.crawl import run_crawl

    return run_crawl(batch_id)


def _bronze(batch_id: str, pdate: date) -> Any:
    from vicap.pipelines.bronze import run_bronze

    return run_bronze(pdate)


def _silver(batch_id: str, pdate: date) -> Any:
    from vicap.pipelines.silver import run_silver_clean

    return run_silver_clean([batch_id], processing_date=pdate)


def _media(batch_id: str, pdate: date) -> Any:
    from vicap.pipelines.media import run_media

    return run_media([batch_id], processing_date=pdate)


def _asr(batch_id: str, pdate: date) -> Any:
    from vicap.pipelines.asr import run_asr

    return run_asr([batch_id], processing_date=pdate)


def _embed(batch_id: str, pdate: date) -> Any:
    from vicap.pipelines.embed import run_embed

    return run_embed([batch_id], processing_date=pdate)


def _predict(batch_id: str, pdate: date) -> Any:
    from vicap.pipelines.predict import run_predict

    return run_predict([batch_id], processing_date=pdate)


def _profile(batch_id: str, pdate: date) -> Any:
    from vicap.pipelines.profile import run_profile

    return run_profile([batch_id], processing_date=pdate)


def _load_clickhouse(batch_id: str, pdate: date) -> Any:
    from vicap.pipelines.load_clickhouse import run_load_clickhouse

    return run_load_clickhouse()


def _quality_check(batch_id: str, pdate: date) -> Any:
    from vicap.pipelines.quality_check import run_quality_check

    return run_quality_check(batch_id)


Step = Callable[[str, date], Any]
STEPS: dict[str, Step] = {
    "crawl": _crawl,
    "bronze": _bronze,
    "silver_clean": _silver,
    "media": _media,
    "asr": _asr,
    "embed": _embed,
    "predict": _predict,
    "profile": _profile,
    "load_clickhouse": _load_clickhouse,
    "quality_check": _quality_check,
}
GPU_STEPS = ("media", "asr", "embed", "predict")  # chạy trong image gpu-worker (DockerOperator)


def select_steps(only: Sequence[str] | None = None, skip: Sequence[str] = ()) -> list[str]:
    unknown = (set(only or ()) | set(skip)) - set(STEPS)
    if unknown:
        raise ValueError(f"bước không có: {sorted(unknown)}; các bước: {list(STEPS)}")
    return [s for s in STEPS if (not only or s in only) and s not in skip]


def run_daily(
    batch_id: str,
    *,
    processing_date: str | date | None = None,
    only: Sequence[str] | None = None,
    skip: Sequence[str] = (),
    steps: dict[str, Step] | None = None,
) -> dict[str, Any]:
    """Chạy các bước được chọn theo thứ tự; bước nào lỗi thì dừng (như DAG). Trả kết quả từng bước."""
    steps = steps or STEPS
    pdate = parse_date(processing_date or now_utc())
    results: dict[str, Any] = {}
    for name in select_steps(only, skip):
        logger.info("daily [{}] → {}", batch_id, name)
        results[name] = steps[name](batch_id, pdate)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-id", required=True)
    ap.add_argument("--processing-date", default=None)
    ap.add_argument("--only", default=None, help="các bước, cách nhau bằng dấu phẩy")
    ap.add_argument("--skip", default="", help="các bước bỏ qua, cách nhau bằng dấu phẩy")
    args = ap.parse_args(argv)

    def split(s: str | None) -> list[str]:
        return [x.strip() for x in (s or "").split(",") if x.strip()]

    setup_logging()
    res = run_daily(args.batch_id, processing_date=args.processing_date, only=split(args.only) or None,
                    skip=split(args.skip))
    for name, r in res.items():
        print(f"{name}: {r.summary() if hasattr(r, 'summary') else r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
