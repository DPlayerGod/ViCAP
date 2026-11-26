"""Trích 8 ảnh/bài đăng ở máy local (thumbnail + 7 frame) để upload lên Kaggle.

Output: frames/{video_id}_{k}.jpg, k = 0..7, cạnh dài <= 640px (đủ nét cho OCR)
  k = 0      : thumbnail (ảnh bìa của bài đăng)
  k = 1..7   : video     -> 7 frame rải đều ở 1/8, 2/8, ..., 7/8 thời lượng (bỏ frame đầu/cuối)
               slideshow -> 7 ảnh rải đều trong các slide; ít hơn 7 slide thì lặp lại theo thứ tự
Kèm frames/frames_manifest.csv: video_id, media_type (video/slideshow), n_source, thumb_source, status, error

Cách nhận diện (video_id = dãy >= 15 chữ số trong tên file/thư mục):
  video     : mọi file .mp4/.mov/.webm trong --videos (đệ quy)
  slideshow : trong --slideshows, hoặc thư mục con tên chứa video_id (vd. 7531.../1.jpg, 2.jpg ...),
              hoặc file phẳng dạng {video_id}_1.jpg, {video_id}-2.webp ...
  thumbnail : --thumbs/{video_id}.(jpg|jpeg|png|webp); không có thì lấy frame đầu (video) / slide đầu

    pip install imageio-ffmpeg pillow tqdm pandas numpy
    python 03_src/scripts/extract_frame.py --videos 02_data/raw/videos --thumbs 02_data/raw/thumbnails \
        --ids 02_data/kaggle/tiktok-multilabel-content/dataset.csv --out 02_data/kaggle/tiktok-frames-v1/frames --dry-run
    python 03_src/scripts/extract_frame.py --videos 02_data/raw/videos --thumbs 02_data/raw/thumbnails \
        --ids 02_data/kaggle/tiktok-multilabel-content/dataset.csv --out 02_data/kaggle/tiktok-frames-v1/frames
"""
import argparse
import csv
import glob
import io
import os
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import imageio_ffmpeg
from PIL import Image
from tqdm import tqdm

N_FRAMES, LONG_SIDE = 7, 640                     # + 1 thumbnail = 8 ảnh / bài đăng
VIDEO_EXT = (".mp4", ".mov", ".webm", ".mkv")
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")
ID_RE = re.compile(r"(\d{15,})")


def vid_of(s):
    m = ID_RE.search(s)
    return m.group(1) if m else None


def natural_key(p):
    """Sắp slide theo số cuối tên file: 2.jpg trước 10.jpg."""
    name = os.path.splitext(os.path.basename(p))[0]
    vid = vid_of(name)
    rest = name.replace(vid, "") if vid else name
    nums = re.findall(r"\d+", rest)
    return (int(nums[-1]) if nums else -1, name)


def find_media(video_dir, slide_dir):
    videos, slides = {}, defaultdict(list)
    for d in filter(None, [video_dir]):
        for p in glob.glob(os.path.join(d, "**", "*"), recursive=True):
            if p.lower().endswith(VIDEO_EXT) and vid_of(os.path.basename(p)):
                videos.setdefault(vid_of(os.path.basename(p)), p)
    for d in filter(None, [slide_dir]):
        for p in glob.glob(os.path.join(d, "**", "*"), recursive=True):
            if not p.lower().endswith(IMG_EXT):
                continue
            rel = os.path.relpath(p, d)
            vid = vid_of(os.path.basename(p)) or vid_of(os.path.dirname(rel))
            if vid:
                slides[vid].append(p)
    for vid in slides:
        slides[vid].sort(key=natural_key)
    both = set(videos) & set(slides)
    if both:
        print("[chú ý] %d id có cả video lẫn slideshow -> dùng video (vd. %s)" % (len(both), sorted(both)[0]))
        for vid in both:
            del slides[vid]
    return videos, dict(slides)


def find_thumb(thumb_dir, vid):
    if not thumb_dir:
        return None
    for ext in IMG_EXT:
        p = os.path.join(thumb_dir, vid + ext)
        if os.path.exists(p):
            return p
    hits = [p for p in glob.glob(os.path.join(thumb_dir, "*%s*" % vid)) if p.lower().endswith(IMG_EXT)]
    return hits[0] if hits else None


FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def probe(path):
    """(duration_s, fps) đọc từ stderr của `ffmpeg -i` (không cần ffprobe)."""
    err = subprocess.run([FFMPEG, "-hide_banner", "-i", path], capture_output=True).stderr.decode("utf-8", "ignore")
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", err)
    if not m:
        raise ValueError("không đọc được duration")
    dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    f = re.search(r"(\d+(?:\.\d+)?)\s*fps", err)
    return dur, float(f.group(1)) if f else 30.0


def grab(path, t):
    """1 frame tại giây t (seek nhanh trước -i). Duration trong header đôi khi dài hơn
    video thật -> nếu không ra frame thì lùi dần t."""
    for scale in (1.0, 0.9, 0.75, 0.5, 0.25, 0.0):
        out = subprocess.run([FFMPEG, "-v", "error", "-ss", "%.3f" % (t * scale), "-i", path,
                              "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
                             capture_output=True).stdout
        if out:
            return Image.open(io.BytesIO(out)).convert("RGB")
    raise ValueError("ffmpeg không trả frame tại %.2fs" % t)


def save(im, out_dir, vid, k):
    im = im.convert("RGB")
    im.thumbnail((LONG_SIDE, LONG_SIDE))
    im.save(os.path.join(out_dir, "%s_%d.jpg" % (vid, k)), quality=90)


def process(job):
    vid, kind, src, thumb, out_dir = job
    if all(os.path.exists(os.path.join(out_dir, "%s_%d.jpg" % (vid, k))) for k in range(N_FRAMES + 1)):
        return [vid, kind, "", "", "skip", ""]
    try:
        if kind == "video":
            dur, fps = probe(src)
            if dur <= 0:
                raise ValueError("video 0 giây")
            n = int(round(dur * fps))
            last = max(dur - 0.1, 0)                       # tránh seek vượt cuối file
            ts = [min(dur * k / (N_FRAMES + 1), last) for k in range(1, N_FRAMES + 1)]
            frames = [grab(src, t) for t in ts]
            first = grab(src, 0) if not thumb else None
            n_source = n
        else:
            paths = src
            n = len(paths)
            pick = (np.linspace(0, n - 1, N_FRAMES).round().astype(int) if n >= N_FRAMES
                    else np.arange(N_FRAMES) * n // N_FRAMES)          # < 7 slide: lặp đều từng slide
            frames = [Image.open(paths[i]) for i in pick]
            first = Image.open(paths[0]) if not thumb else None
            n_source = len(paths)
        save(Image.open(thumb) if thumb else first, out_dir, vid, 0)
        for k, im in enumerate(frames, start=1):
            save(im, out_dir, vid, k)
        return [vid, kind, n_source, "thumbnail" if thumb else "first_frame", "ok", ""]
    except Exception as e:
        return [vid, kind, "", "", "error", str(e)[:200]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="videos")
    ap.add_argument("--slideshows", default=None, help="mặc định: tìm slideshow trong cả --videos")
    ap.add_argument("--thumbs", default="thumbnails")
    ap.add_argument("--out", default="frames")
    ap.add_argument("--ids", default=None, help="dataset.csv: chỉ trích video_id có trong file này")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--dry-run", action="store_true", help="chỉ in ra đã tìm thấy gì, không ghi ảnh")
    a = ap.parse_args()

    videos, slides = find_media(a.videos, a.slideshows or a.videos)
    thumb_dir = a.thumbs if a.thumbs and os.path.isdir(a.thumbs) else None
    jobs = [(v, "video", p) for v, p in videos.items()] + [(v, "slideshow", ps) for v, ps in slides.items()]

    if a.ids:
        import pandas as pd
        want = set(pd.read_csv(a.ids, dtype={"video_id": str})["video_id"])
        missing = sorted(want - {j[0] for j in jobs})
        jobs = [j for j in jobs if j[0] in want]
        print("dataset: %d id | tìm thấy media: %d | THIẾU: %d" % (len(want), len(jobs), len(missing)))
        if missing:
            print("  vd. thiếu:", missing[:10])

    n_thumb = sum(find_thumb(thumb_dir, j[0]) is not None for j in jobs)
    print("video: %d | slideshow: %d | có thumbnail: %d/%d" % (len(videos), len(slides), n_thumb, len(jobs)))
    if slides:
        cnt = np.array([len(ps) for ps in slides.values()])
        print("slide/bài: min %d, median %d, max %d | bài < %d slide (sẽ lặp ảnh): %d"
              % (cnt.min(), np.median(cnt), cnt.max(), N_FRAMES, (cnt < N_FRAMES).sum()))
        v0 = next(iter(slides))
        print("vd. slideshow %s:" % v0, [os.path.basename(p) for p in slides[v0]][:10])
    if a.dry_run:
        raise SystemExit

    os.makedirs(a.out, exist_ok=True)
    jobs = [(v, kind, src, find_thumb(thumb_dir, v), a.out) for v, kind, src in jobs]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(tqdm(ex.map(process, jobs, chunksize=4), total=len(jobs)))
    mpath = os.path.join(a.out, "frames_manifest.csv")
    old = {}
    if os.path.exists(mpath):                      # chạy lại: giữ thông tin dòng đã skip
        with open(mpath, encoding="utf-8") as f:
            old = {r[0]: r for r in csv.reader(f) if r and r[0] != "video_id"}
    res = [old[r[0]] if r[4] == "skip" and old.get(r[0], [""] * 5)[4] == "ok" else r for r in res]
    merged = dict(old)                             # id không xử lý lần này (vd. chạy --ids 1 phần) vẫn giữ nguyên
    merged.update({r[0]: r for r in res})
    with open(mpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "media_type", "n_source", "thumb_source", "status", "error"])
        w.writerows(merged.values())
    for s in ("ok", "skip", "error"):
        print("%-5s: %d" % (s, sum(r[4] == s for r in res)))
    if any(r[4] == "error" for r in res):
        print("Xem lỗi trong %s/frames_manifest.csv" % a.out)
