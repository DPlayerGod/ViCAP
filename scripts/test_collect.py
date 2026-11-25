"""Giai đoạn 0: thử thu thập TikTok bằng yt-dlp (xem PLAN mục 3, TODO Giai đoạn 0).

Tải N bài gần nhất của mỗi creator, đặt tên theo PLAN mục 4.2, ghi metadata.csv + report.txt.

    python scripts/test_collect.py phase0/test_creators.txt --n 10 --out test_out/
    python scripts/test_collect.py phase0/test_creators.txt --n 10 --out test_out/ --cookies-from-browser firefox

Thư mục ra:
    videos/{stem}.mp4                         video thường
    videos/{stem}/{stem}__img_NNN.{jpg|png}   ảnh slideshow
    videos/{stem}/{stem}.mp3                  âm thanh slideshow
    thumbnails/{stem}.{jpg|png}               `video.cover` (khớp thumbnail Kaggle)

Ảnh được lưu nguyên bytes TikTok trả về, không chuyển định dạng; đuôi theo định dạng thật.
PIL / OpenCV nhận định dạng theo nội dung nên đuôi không ảnh hưởng embedding.
    info/{video_id}.json                      itemStruct gốc của TikTok (để đối chiếu)
    metadata.csv, report.txt

Ghi chú:
- yt-dlp chỉ trả mp3 cho slideshow. Ảnh slideshow lấy từ `imagePost` trong itemStruct, đọc qua
  `TikTokIE._extract_web_data_and_status` (hàm nội bộ của yt-dlp, có thể đổi giữa các phiên bản).
- Phần thời gian trong tên file là `createTime` theo UTC (khớp ví dụ trong PLAN 4.2). Không suy từ
  `video_id >> 32` vì lệch vài giây.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import yt_dlp
from yt_dlp.networking import Request

# Bỏ bản có watermark ("download"); ưu tiên h264 cho dễ đọc bằng OpenCV.
VIDEO_FORMAT = "b[vcodec^=h264][format_id!*=download]/b[format_id!*=download]/b"
JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@dataclass
class Row:
    username: str
    video_id: str
    stem: str = ""
    media_type: str = ""  # video | slideshow
    posted_at_utc: str = ""
    create_time: int | None = None
    desc: str = ""  # caption gốc (gồm cả hashtag)
    caption: str = ""  # desc đã bỏ #hashtag
    hashtags: str = ""  # nối bằng "|"
    duration: int | None = None  # giây; slideshow dùng thời lượng nhạc
    width: int | None = None
    height: int | None = None
    vcodec: str = ""
    format_id: str = ""
    n_images: int | None = None
    view_count: int | None = None
    like_count: int | None = None
    comment_count: int | None = None
    share_count: int | None = None
    save_count: int | None = None
    music_title: str = ""
    media_ok: bool = False
    media_bytes: int = 0
    images_ok: int = 0
    audio_ok: bool = False
    thumbnail_ok: bool = False
    thumb_format: str = ""  # jpg | png | webp — bytes gốc, không chuyển định dạng
    thumb_md5: str = ""  # để so từng byte với thumbnail trên Kaggle
    seconds: float = 0.0
    attempts: int = 1
    error: str = ""


def raw_stem(username: str, create_time: int, video_id: str) -> str:
    ts = datetime.fromtimestamp(create_time, timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{username}__{ts}__{video_id}"


def read_creators(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip().lstrip("@")
        if line:
            out.append(line)
    return out


def to_int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def fetch(ydl: yt_dlp.YoutubeDL, url: str, headers: dict | None = None) -> bytes:
    with ydl.urlopen(Request(url, headers=headers or {})) as resp:
        return resp.read()


def image_ext(data: bytes) -> str:
    """Đuôi file theo nội dung (TikTok trả JPEG hoặc PNG; dataset cũ có ~19% PNG)."""
    if data[:3] == JPEG_MAGIC:
        return ".jpg"
    if data[:8] == PNG_MAGIC:
        return ".png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    raise ValueError(f"định dạng ảnh lạ ({data[:12]!r})")


def save_image(data: bytes, path_no_ext: Path) -> Path:
    """Ghi nguyên bytes TikTok trả về (không chuyển định dạng), đuôi theo định dạng thật."""
    path = path_no_ext.with_name(path_no_ext.name + image_ext(data))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def list_posts(ydl_flat: yt_dlp.YoutubeDL, username: str) -> list[str]:
    info = ydl_flat.extract_info(f"https://www.tiktok.com/@{username}", download=False)
    return [e["id"] for e in (info or {}).get("entries") or [] if e and e.get("id")]


def collect_post(ydl: yt_dlp.YoutubeDL, ie, username: str, video_id: str, out: Path) -> Row:
    row = Row(username=username, video_id=video_id)
    t0 = time.perf_counter()
    try:
        url = f"https://www.tiktok.com/@{username}/video/{video_id}"
        item, status = ie._extract_web_data_and_status(url, video_id)
        if status != 0 or not item:
            raise RuntimeError(f"status={status}")
        (out / "info").mkdir(parents=True, exist_ok=True)
        (out / "info" / f"{video_id}.json").write_text(
            json.dumps(item, ensure_ascii=False, indent=1), encoding="utf-8"
        )

        row.create_time = to_int(item.get("createTime"))
        row.posted_at_utc = datetime.fromtimestamp(row.create_time, timezone.utc).isoformat()
        row.stem = raw_stem(username, row.create_time, video_id)
        row.desc = item.get("desc") or ""
        tags = [c.get("title") for c in item.get("challenges") or [] if c.get("title")]
        if not tags:
            tags = [t.get("hashtagName") for t in item.get("textExtra") or [] if t.get("hashtagName")]
        row.hashtags = "|".join(tags)
        row.caption = re.sub(r"\s+", " ", re.sub(r"#\S+", " ", row.desc)).strip()
        stats = item.get("statsV2") or item.get("stats") or {}
        row.view_count = to_int(stats.get("playCount"))
        row.like_count = to_int(stats.get("diggCount"))
        row.comment_count = to_int(stats.get("commentCount"))
        row.share_count = to_int(stats.get("shareCount"))
        row.save_count = to_int(stats.get("collectCount"))
        music = item.get("music") or {}
        row.music_title = music.get("title") or ""
        video = item.get("video") or {}
        images = (item.get("imagePost") or {}).get("images") or []

        info = ie._parse_aweme_video_web(item, url, video_id)
        ydl.add_default_extra_info(info, ie, url)
        try:
            sel = ydl.process_ie_result(info, download=False)
        except yt_dlp.utils.YoutubeDLError:
            if not images:  # video thường mà không có format là lỗi thật
                raise
            sel = {}  # slideshow không có nhạc: yt-dlp không có format nào
        headers = sel.get("http_headers") or info.get("http_headers") or {}

        if images:
            row.media_type = "slideshow"
            row.n_images = len(images)
            row.duration = to_int(music.get("duration")) or to_int(sel.get("duration"))
            folder = out / "videos" / row.stem
            for i, im in enumerate(images, 1):
                urls = (im.get("imageURL") or {}).get("urlList") or []
                for u in urls:  # thử lần lượt các CDN
                    try:
                        save_image(fetch(ydl, u, headers), folder / f"{row.stem}__img_{i:03d}")
                        row.images_ok += 1
                        break
                    except Exception:
                        continue
            if sel.get("url") and sel.get("vcodec") == "none":
                data = fetch(ydl, sel["url"], headers)
                (folder / f"{row.stem}.{sel.get('ext') or 'mp3'}").write_bytes(data)
                row.audio_ok = len(data) > 0
                row.media_bytes = len(data)
            row.media_ok = row.images_ok == len(images)
            row.width = to_int(images[0].get("imageWidth"))
            row.height = to_int(images[0].get("imageHeight"))
            cover = video.get("cover") or (images[0].get("imageURL") or {}).get("urlList", [None])[0]
        else:
            row.media_type = "video"
            row.duration = to_int(video.get("duration")) or to_int(sel.get("duration"))
            row.format_id = sel.get("format_id") or ""
            row.vcodec = sel.get("vcodec") or ""
            row.width, row.height = to_int(sel.get("width")), to_int(sel.get("height"))
            path = out / "videos" / f"{row.stem}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            data = fetch(ydl, sel["url"], headers)
            path.write_bytes(data)
            row.media_bytes = len(data)
            row.media_ok = len(data) > 10_000
            cover = video.get("cover") or video.get("originCover") or sel.get("thumbnail")

        if cover:
            data = fetch(ydl, cover, headers)
            path = save_image(data, out / "thumbnails" / row.stem)
            row.thumb_format = path.suffix.lstrip(".")
            row.thumb_md5 = hashlib.md5(data).hexdigest()
            row.thumbnail_ok = True
    except Exception as e:  # ghi lỗi vào CSV, chạy tiếp bài sau
        row.error = f"{type(e).__name__}: {e}".replace("\n", " ")[:300]
        with (out / "errors.log").open("a", encoding="utf-8") as f:
            f.write(f"=== {username}/{video_id}\n{traceback.format_exc()}\n")
    row.seconds = round(time.perf_counter() - t0, 2)
    return row


def pct(a: int, b: int) -> str:
    return f"{a}/{b} ({100 * a / b:.1f}%)" if b else "0/0"


def write_report(rows: list[Row], creators: dict[str, str], out: Path, elapsed: float) -> str:
    vids = [r for r in rows if r.media_type == "video"]
    slides = [r for r in rows if r.media_type == "slideshow"]
    failed = [r for r in rows if r.error and not r.media_type]
    L = [f"yt-dlp {yt_dlp.version.__version__} — {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC", ""]

    L.append("## Creator")
    for u, st in creators.items():
        L.append(f"- {u}: {st}")

    L += ["", "## Tổng", f"- Bài: {len(rows)} (video {len(vids)}, slideshow {len(slides)}, lỗi trước khi biết loại {len(failed)})"]

    ok_v = [r for r in vids if r.media_ok and r.thumbnail_ok and not r.error]
    L += ["", "## Video thường"]
    L.append(f"- Tải được mp4 + thumbnail: {pct(len(ok_v), len(vids) + len(failed))}")
    L.append(f"- Codec: {dict(Counter(r.vcodec for r in vids))}; format: {dict(Counter(r.format_id.split('_')[0] for r in vids))}")
    L.append(f"- Định dạng thumbnail (bytes gốc): {dict(Counter(r.thumb_format for r in rows if r.thumb_format))}")

    def complete(rs: list[Row], name: str, fn) -> str:
        return f"  - {name}: {pct(sum(1 for r in rs if fn(r)), len(rs))}"

    for title, rs in (("Video thường", vids), ("Slideshow", slides)):
        L.append(f"- Đủ trường ({title}):")
        L.append(complete(rs, "caption (desc khác rỗng)", lambda r: bool(r.desc.strip())))
        L.append(complete(rs, "caption không tính hashtag", lambda r: bool(r.caption)))
        L.append(complete(rs, "hashtag (≥1)", lambda r: bool(r.hashtags)))
        L.append(complete(rs, "thời lượng > 0", lambda r: (r.duration or 0) > 0))
        L.append(complete(rs, "thumbnail", lambda r: r.thumbnail_ok))
        L.append(complete(rs, "lượt xem", lambda r: r.view_count is not None))
        L.append(complete(rs, "lượt thích", lambda r: r.like_count is not None))
        L.append(complete(rs, "đủ cả caption+hashtag+thời lượng+thumbnail",
                          lambda r: bool(r.desc.strip()) and bool(r.hashtags) and (r.duration or 0) > 0 and r.thumbnail_ok))

    L += ["", "## Slideshow"]
    if slides:
        L.append(f"- Tải đủ mọi ảnh: {pct(sum(r.media_ok for r in slides), len(slides))}")
        L.append(f"- Ảnh: {sum(r.images_ok for r in slides)}/{sum(r.n_images or 0 for r in slides)}")
        L.append(f"- Có âm thanh mp3: {pct(sum(r.audio_ok for r in slides), len(slides))}")
        L.append("- yt-dlp chuẩn chỉ trả mp3; ảnh lấy từ imagePost (hàm nội bộ yt-dlp).")
    else:
        L.append("- Không có slideshow trong mẫu.")

    L += ["", "## Lỗi"]
    errs = Counter(r.error.split(":")[0] + ": " + r.error.split(":", 1)[-1][:80] for r in rows if r.error)
    L += [f"- {n}× {e}" for e, n in errs.most_common()] or ["- Không có"]
    blocked = [r for r in rows if re.search(r"403|captcha|10204|login|blocked", r.error, re.I)]
    L.append(f"- Có dấu hiệu bị chặn (403/captcha/10204/login): {len(blocked)}")
    retried = [r for r in rows if r.attempts > 1]
    L.append(f"- Phải thử lại: {len(retried)} bài, qua sau khi thử lại: {sum(not r.error for r in retried)}")

    L += ["", "## Thời gian"]
    for title, rs in (("video", vids), ("slideshow", slides)):
        if rs:
            s = [r.seconds for r in rs]
            L.append(f"- {title}: trung bình {statistics.mean(s):.2f}s, trung vị {statistics.median(s):.2f}s, max {max(s):.2f}s")
    L.append(f"- Tổng: {elapsed:.1f}s cho {len(rows)} bài")

    text = "\n".join(L) + "\n"
    (out / "report.txt").write_text(text, encoding="utf-8")
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("creators", type=Path)
    ap.add_argument("--n", type=int, default=10, help="số bài gần nhất mỗi creator")
    ap.add_argument("--out", type=Path, default=Path("test_out"))
    ap.add_argument("--sleep", type=float, default=1.0, help="nghỉ giữa các bài (giây)")
    ap.add_argument("--retries", type=int, default=2, help="số lần thử lại khi một bài lỗi")
    ap.add_argument("--cookies-from-browser", default=None, help="vd. firefox, chrome")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    base = {"quiet": True, "no_warnings": True}
    if args.cookies_from_browser:
        base["cookiesfrombrowser"] = (args.cookies_from_browser,)
    ydl = yt_dlp.YoutubeDL({**base, "format": VIDEO_FORMAT})
    ydl_flat = yt_dlp.YoutubeDL({**base, "extract_flat": "in_playlist", "playlistend": args.n})
    ie = ydl.get_info_extractor("TikTok")

    rows: list[Row] = []
    creators: dict[str, str] = {}
    t0 = time.perf_counter()
    for username in read_creators(args.creators):
        try:
            ids = list_posts(ydl_flat, username)[: args.n]
        except Exception as e:
            creators[username] = f"LỖI lấy danh sách: {type(e).__name__}: {str(e)[:150]}"
            print(f"[{username}] {creators[username]}", flush=True)
            continue
        creators[username] = f"{len(ids)} bài" if ids else "danh sách rỗng"
        for k, vid in enumerate(ids, 1):
            for attempt in range(1, args.retries + 2):  # lỗi tạm thời (vd. status=10231) thường qua khi thử lại
                row = collect_post(ydl, ie, username, vid, args.out)
                row.attempts = attempt
                if not row.error:
                    break
                time.sleep(3 * attempt)
            rows.append(row)
            status = "OK" if not row.error and row.media_ok else f"LỖI {row.error[:100]}"
            print(f"[{username} {k}/{len(ids)}] {vid} {row.media_type or '?'} {row.seconds}s {status}", flush=True)
            time.sleep(args.sleep)

    with (args.out / "metadata.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=[x.name for x in fields(Row)])
        w.writeheader()
        w.writerows(asdict(r) for r in rows)
    print(write_report(rows, creators, args.out, time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
