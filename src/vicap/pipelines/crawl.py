"""Bước ``crawl`` (PLAN mục 5): creator trong ``configs/creators.txt`` → ``raw/`` + topic ``tiktok.videos``.

Chạy tay (máy host, cần ``make up-crawl``)::

    uv run python -m vicap.pipelines.crawl --creators anhdaubep_vn --n 2
    uv run python -m vicap.pipelines.crawl --file configs/creators.txt --n 10 --batch-id 20261125-000000
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from vicap.ingestion.crawler import CrawlStats, KafkaSink, RawStore, crawl_creators
from vicap.ingestion.tiktok import TikTokClient
from vicap.utils.dates import format_batch_id, now_utc

DEFAULT_CREATORS_FILE = Path("configs/creators.txt")
DEFAULT_N = 10  # Đợt B: 10 video gần nhất mỗi creator (PLAN mục 3)


class CrawlFailedError(RuntimeError):
    """Cả lô không có bài nào tải được (bị chặn, mất mạng, …): dừng task để Airflow thử lại."""


def read_creators(path: Path) -> list[str]:
    """Mỗi dòng một username, bỏ ``@`` đầu, bỏ comment ``#`` và dòng trống, bỏ trùng."""
    seen: dict[str, None] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.split("#", 1)[0].strip().lstrip("@")
        if name:
            seen.setdefault(name, None)
    return list(seen)


def run_crawl(
    batch_id: str,
    *,
    creators: Sequence[str] | None = None,
    creators_file: Path = DEFAULT_CREATORS_FILE,
    n: int = DEFAULT_N,
    refresh_media: bool = False,
    cookies_from_browser: str | None = None,
) -> CrawlStats:
    usernames = list(creators) if creators else read_creators(creators_file)
    if not usernames:
        raise ValueError(f"không có creator nào (truyền creators hoặc điền {creators_file})")

    store = RawStore()
    with TikTokClient(cookies_from_browser=cookies_from_browser) as client, KafkaSink() as sink:
        stats = crawl_creators(
            usernames, n=n, batch_id=batch_id, client=client, store=store, send=sink, refresh_media=refresh_media
        )
    if stats.ok == 0:
        raise CrawlFailedError(f"không tải được bài nào: {stats.summary()}; creator lỗi: {stats.creator_errors}")
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--creators", help="danh sách username, cách nhau bằng dấu phẩy (thay cho --file)")
    ap.add_argument("--file", type=Path, default=DEFAULT_CREATORS_FILE)
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="số bài gần nhất mỗi creator")
    ap.add_argument("--batch-id", default=None, help="mặc định: giờ hiện tại, YYYYMMDD-HHMMSS (UTC)")
    ap.add_argument("--refresh-media", action="store_true", help="tải lại mp4 kể cả khi đã có trên MinIO")
    ap.add_argument("--cookies-from-browser", default=None, help="vd. firefox, khi bị TikTok chặn")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("kafka").setLevel(logging.WARNING)  # kafka-python log mỗi lần kết nối ở mức INFO
    creators = [c.strip().lstrip("@") for c in args.creators.split(",") if c.strip()] if args.creators else None
    stats = run_crawl(
        args.batch_id or format_batch_id(now_utc()),
        creators=creators,
        creators_file=args.file,
        n=args.n,
        refresh_media=args.refresh_media,
        cookies_from_browser=args.cookies_from_browser,
    )
    print(stats.summary())
    for username, error in stats.creator_errors.items():
        print(f"  creator lỗi {username}: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
