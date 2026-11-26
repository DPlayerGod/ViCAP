"""DAG ``daily_pipeline`` (PLAN mục 5)::

    batch_id → crawl → bronze → silver_clean → media → asr → embed → predict → profile → load_clickhouse
             → quality_check

- ``batch_id`` = ``YYYYMMDD-HHMMSS`` (UTC) từ logical date (chạy tay không có logical date thì lấy ``run_after``);
  retry dùng lại cùng id nên mọi bước idempotent. Ngày xử lý = ngày của ``batch_id``.
- crawl, bronze, silver_clean, profile, load_clickhouse, quality_check chạy trong Airflow (Spark local cho
  silver_clean / profile). media, asr, embed, predict chạy trong image gpu-worker bằng DockerOperator
  (``dags/common/gpu.py``); media chỉ cần CPU + ffmpeg nhưng image Airflow không có.
- Các bước sau bronze chỉ xử lý video của ``batch_id``; dòng lỗi bị cách ly, phần còn lại chạy tiếp; lỗi cấp
  lô thì task lỗi và DAG dừng.
- Tham số khi trigger: ``creators`` (danh sách username cách nhau bằng dấu phẩy, mặc định ``configs/creators.txt``),
  ``n`` (số bài gần nhất mỗi creator).

Dịch vụ cần: ``make up-daily``. Trigger: ``make dag-trigger`` hoặc từ UI.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.sdk import Param, dag, get_current_context, task
from common.gpu import gpu_task


def _pdate(batch_id: str) -> str:
    return datetime.strptime(batch_id[:8], "%Y%m%d").date().isoformat()


@dag(
    dag_id="daily_pipeline",
    schedule="@daily",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,  # Spark local + GPU 4 GB: không chạy hai lô cùng lúc
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    params={
        "creators": Param("", type="string", description="username, cách nhau bằng dấu phẩy; trống = creators.txt"),
        "n": Param(10, type="integer", minimum=1, description="số bài gần nhất mỗi creator"),
    },
    tags=["vicap"],
)
def daily_pipeline():
    @task
    def batch_id() -> str:
        from vicap.utils.dates import format_batch_id

        ctx = get_current_context()
        when = ctx.get("logical_date") or ctx["dag_run"].run_after
        return format_batch_id(when)

    @task
    def crawl(bid: str) -> str:
        from vicap.pipelines.crawl import run_crawl

        params = get_current_context()["params"]
        creators = [c.strip().lstrip("@") for c in params["creators"].split(",") if c.strip()] or None
        stats = run_crawl(bid, creators=creators, n=int(params["n"]))
        for username, error in stats.creator_errors.items():
            print(f"creator lỗi {username}: {error}")
        return stats.summary()

    @task
    def bronze(bid: str) -> str:
        from vicap.pipelines.bronze import run_bronze

        return run_bronze(_pdate(bid)).summary()

    @task
    def silver_clean(bid: str) -> str:
        from vicap.pipelines.silver import run_silver_clean

        return run_silver_clean([bid], processing_date=_pdate(bid)).summary()

    @task
    def profile(bid: str) -> str:
        from vicap.pipelines.profile import run_profile

        return run_profile([bid], processing_date=_pdate(bid)).summary()

    @task
    def load_clickhouse() -> str:
        from vicap.pipelines.load_clickhouse import run_load_clickhouse

        return run_load_clickhouse().summary()

    @task
    def quality_check(bid: str) -> list[dict]:
        from vicap.pipelines.quality_check import run_quality_check

        return [r.__dict__ for r in run_quality_check(bid)]

    def worker(step: str, bid, *, gpu: bool = True):
        # bid là XComArg: Airflow render lúc chạy và tự thêm phụ thuộc batch_id → step, như một @task
        return gpu_task(step, ["python", "-m", f"vicap.pipelines.{step}", "--batch-id", bid], gpu=gpu)

    bid = batch_id()
    crawled = crawl(bid)
    loaded = bronze(bid)
    silver = silver_clean(bid)
    media, asr, embed, predict = (worker("media", bid, gpu=False), worker("asr", bid), worker("embed", bid),
                                  worker("predict", bid))
    prof = profile(bid)
    ch = load_clickhouse()
    qc = quality_check(bid)

    crawled >> loaded >> silver >> media >> asr >> embed >> predict >> prof >> ch >> qc


daily_pipeline()
