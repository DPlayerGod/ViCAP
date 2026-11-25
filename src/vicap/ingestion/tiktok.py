"""Gọi TikTok qua yt-dlp theo cách đã kiểm ở Giai đoạn 0 (``scripts/test_collect.py``, PLAN mục 3).

- Danh sách bài: trang ``@username`` với ``extract_flat`` (chỉ lấy id, nhanh).
- Một bài: ``TikTokIE._extract_web_data_and_status`` trả ``itemStruct`` gốc của TikTok. Đây là **hàm nội bộ**
  của yt-dlp (có thể đổi giữa các phiên bản), nên ``yt-dlp`` được ghim trong ``pyproject.toml``. Dùng hàm này
  vì ``itemStruct`` có ``imagePost`` (ảnh slideshow) và không chứa cookie / header như ``info.json`` của yt-dlp.
- URL mp4: dựng lại info của yt-dlp từ ``itemStruct`` rồi chọn format như Giai đoạn 0 (bỏ bản watermark
  ``download``, ưu tiên h264).

Không cần cookie (Giai đoạn 0: 246 bài tải được không cookie). Bị chặn thì truyền
``cookies_from_browser="firefox"`` (Chrome trên Windows không đọc được cookie).
"""

from __future__ import annotations

from typing import Any

import yt_dlp
from yt_dlp.networking import Request

VIDEO_FORMAT = "b[vcodec^=h264][format_id!*=download]/b[format_id!*=download]/b"


class TikTokError(RuntimeError):
    """TikTok trả lỗi cho một bài (``status`` khác 0, vd. 10231 tạm thời, 10204 bài bị gỡ)."""

    def __init__(self, video_id: str, status: Any):
        super().__init__(f"{video_id}: TikTok status={status}")
        self.video_id = video_id
        self.status = status


def post_url(username: str, video_id: str) -> str:
    return f"https://www.tiktok.com/@{username}/video/{video_id}"


class TikTokClient:
    def __init__(self, *, cookies_from_browser: str | None = None):
        base: dict[str, Any] = {"quiet": True, "no_warnings": True}
        if cookies_from_browser:
            base["cookiesfrombrowser"] = (cookies_from_browser,)
        self._base = base
        self._ydl = yt_dlp.YoutubeDL({**base, "format": VIDEO_FORMAT})
        self._ie = self._ydl.get_info_extractor("TikTok")

    def list_post_ids(self, username: str, n: int) -> list[str]:
        """``n`` bài gần nhất trên trang creator (bài ghim có thể đứng đầu)."""
        with yt_dlp.YoutubeDL({**self._base, "extract_flat": "in_playlist", "playlistend": n}) as ydl:
            info = ydl.extract_info(f"https://www.tiktok.com/@{username}", download=False)
        ids = [str(e["id"]) for e in (info or {}).get("entries") or [] if e and e.get("id")]
        return ids[:n]

    def fetch_item(self, username: str, video_id: str) -> dict[str, Any]:
        item, status = self._ie._extract_web_data_and_status(post_url(username, video_id), video_id)
        if status != 0 or not item:
            raise TikTokError(video_id, status)
        return item

    def video_source(self, item: dict[str, Any], username: str) -> tuple[str, dict[str, str]]:
        """URL mp4 đã chọn format + header cần gửi kèm khi tải."""
        url = post_url(username, str(item["id"]))
        info = self._ie._parse_aweme_video_web(item, url, str(item["id"]))
        self._ydl.add_default_extra_info(info, self._ie, url)
        selected = self._ydl.process_ie_result(info, download=False)
        if not selected.get("url"):
            raise RuntimeError(f"{item['id']}: yt-dlp không chọn được format mp4")
        headers = selected.get("http_headers") or info.get("http_headers") or {}
        return selected["url"], headers

    def download(self, url: str, headers: dict[str, str] | None = None) -> bytes:
        with self._ydl.urlopen(Request(url, headers=headers or {})) as resp:
            return resp.read()

    def close(self) -> None:
        self._ydl.close()

    def __enter__(self) -> TikTokClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
