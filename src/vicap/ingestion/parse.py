"""Đọc ``itemStruct`` của TikTok thành các trường chính (PLAN mục 4, 4.2). Hàm thuần, không gọi mạng.

``itemStruct`` là JSON gốc TikTok trả cho một bài (lấy qua yt-dlp, ``vicap.ingestion.tiktok``). Crawler lưu
nguyên nó thành ``raw/videos/{video_id}/info.json`` và Bronze chép vào cột ``raw_json``; các trường ở đây
chỉ là phần đi kèm message Kafka.

Quy tắc đã kiểm ở Giai đoạn 0 (186 video, ``scripts/phase0_compare.py``):

- Hashtag lấy từ các ``#…`` trong ``desc`` (khớp dataset 186/186), bỏ ký tự zero-width space ``\\u200b``.
  ``challenges`` của TikTok đã bị viết thường nên chỉ dùng khi ``desc`` không có hashtag.
- ``createTime`` là giờ đăng theo UTC; không suy từ ``video_id >> 32`` (lệch vài giây).
- Slideshow: có ``imagePost.images``; ``video.duration`` bằng 0 nên thời lượng lấy từ nhạc.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

VIDEO = "video"
SLIDESHOW = "slideshow"

_HASHTAG = re.compile(r"#([^\s#]+)")
_VIDEO_ID = re.compile(r"^\d+$")


class ItemParseError(ValueError):
    """``itemStruct`` thiếu trường bắt buộc hoặc sai kiểu (TikTok đổi cấu trúc, bài bị gỡ, …)."""


@dataclass(frozen=True)
class PostMeta:
    video_id: str
    username: str
    posted_at: datetime  # UTC
    media_type: str  # video | slideshow
    description: str  # desc gốc, gồm cả #hashtag
    hashtags: tuple[str, ...]
    duration: int | None  # giây
    width: int | None
    height: int | None
    play_count: int | None
    like_count: int | None
    comment_count: int | None
    share_count: int | None
    save_count: int | None
    follower_count: int | None
    cover_url: str | None
    # Slideshow: mỗi ảnh một danh sách URL (thử lần lượt các CDN); video thường thì rỗng.
    image_urls: tuple[tuple[str, ...], ...] = ()


def to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _positive(value: Any) -> int | None:
    n = to_int(value)
    return n if n and n > 0 else None


def extract_hashtags(description: str) -> list[str]:
    """``#tag`` trong caption theo thứ tự xuất hiện, bỏ trùng, giữ nguyên chữ hoa / thường."""
    seen: dict[str, None] = {}
    for tag in _HASHTAG.findall(description.replace("​", " ")):
        tag = tag.strip()
        if tag:
            seen.setdefault(tag, None)
    return list(seen)


def strip_hashtags(description: str) -> str:
    """Caption đã bỏ ``#hashtag`` và gộp khoảng trắng (cách so caption ở Giai đoạn 0)."""
    text = _HASHTAG.sub(" ", description.replace("​", " "))
    return " ".join(text.split())


def _first_url(node: Any) -> str | None:
    """URL đầu tiên của một nút ``{"imageURL": {"urlList": [...]}}``."""
    urls = ((node or {}).get("imageURL") or {}).get("urlList") or [] if isinstance(node, dict) else []
    return next((u for u in urls if u), None)


def parse_item(item: dict[str, Any]) -> PostMeta:
    if not isinstance(item, dict):
        raise ItemParseError(f"itemStruct phải là dict, nhận {type(item).__name__}")

    video_id = str(item.get("id") or "")
    if not _VIDEO_ID.match(video_id):
        raise ItemParseError(f"id không hợp lệ: {video_id!r}")
    username = str((item.get("author") or {}).get("uniqueId") or "")
    if not username:
        raise ItemParseError(f"{video_id}: thiếu author.uniqueId")
    create_time = to_int(item.get("createTime"))
    if not create_time:
        raise ItemParseError(f"{video_id}: thiếu createTime")

    description = str(item.get("desc") or "")
    hashtags = extract_hashtags(description)
    if not hashtags:
        hashtags = [c["title"] for c in item.get("challenges") or [] if isinstance(c, dict) and c.get("title")]

    video = item.get("video") or {}
    music = item.get("music") or {}
    images = (item.get("imagePost") or {}).get("images") or []
    stats = item.get("statsV2") or item.get("stats") or {}
    author_stats = item.get("authorStatsV2") or item.get("authorStats") or {}

    if images:
        media_type = SLIDESHOW
        duration = _positive(music.get("duration"))
        width, height = _positive(images[0].get("imageWidth")), _positive(images[0].get("imageHeight"))
        image_urls = tuple(tuple(u for u in (im.get("imageURL") or {}).get("urlList") or [] if u) for im in images)
        if not all(image_urls):
            raise ItemParseError(f"{video_id}: có ảnh slideshow không có URL")
        cover_url = video.get("cover") or _first_url((item.get("imagePost") or {}).get("cover")) or image_urls[0][0]
    else:
        media_type = VIDEO
        duration = _positive(video.get("duration")) or _positive(music.get("duration"))
        width, height = _positive(video.get("width")), _positive(video.get("height"))
        image_urls = ()
        cover_url = video.get("cover") or video.get("originCover")

    return PostMeta(
        video_id=video_id,
        username=username,
        posted_at=datetime.fromtimestamp(create_time, UTC),
        media_type=media_type,
        description=description,
        hashtags=tuple(hashtags),
        duration=duration,
        width=width,
        height=height,
        play_count=to_int(stats.get("playCount")),
        like_count=to_int(stats.get("diggCount")),
        comment_count=to_int(stats.get("commentCount")),
        share_count=to_int(stats.get("shareCount")),
        save_count=to_int(stats.get("collectCount")),
        follower_count=to_int(author_stats.get("followerCount")),
        cover_url=cover_url or None,
        image_urls=image_urls,
    )
