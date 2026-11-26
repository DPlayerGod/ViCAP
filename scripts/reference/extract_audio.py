# -*- coding: utf-8 -*-
"""Tach audio tu video TikTok da tai ve, chuan bi cho ASR (PhoWhisper) tren Kaggle.

Thuc hien Phan A (A1 + A2) trong 01_docs/agents/PLAN_TRANSCRIPT.md:
  - Doc danh sach video_id tu 02_data/kaggle/tiktok-multilabel-content/dataset.csv (1094 video da gan nhan
    - dung dung tap se dua vao train, khong tach audio thua cho video khong dung).
  - Voi moi video_id, tim file .mp4 tuong ung trong 02_data/raw/videos/,
    dung ffmpeg (qua imageio-ffmpeg, khong can cai dat he thong) tach audio,
    resample 16kHz mono, nen AAC 64kbps -> {video_id}.mp3 trong 02_data/kaggle/tiktok-audio-v1/audio/.
  - Ghi 02_data/kaggle/tiktok-audio-v1/audio_index.csv (video_id, audio_filename, duration_seconds).
  - Ghi 02_data/kaggle/tiktok-audio-v1/audio_extract_errors.csv cho video thieu file / loi decode /
    audio qua ngan (<1s) - khong lam dung ca script.
  - 02_data/kaggle/tiktok-audio-v1/ sau khi chay xong la thu muc san sang upload len Kaggle lam
    Dataset moi (vd. "tiktok-audio-v1"), tach rieng khoi tiktok-multilabel-content.

Chay lai duoc nhieu lan: video nao da co file .mp3 dung dung luong thi bo qua,
tru khi truyen --overwrite.

Cach dung:
    python 03_src/scripts/extract_audio.py
    python 03_src/scripts/extract_audio.py --limit 20          # chay thu 20 video dau
    python 03_src/scripts/extract_audio.py --overwrite
"""

import argparse
import csv
import json
import re
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg
from mutagen.mp3 import MP3

ROOT = Path(__file__).resolve().parents[2]
DATASET_CSV = ROOT / "02_data/kaggle/tiktok-multilabel-content" / "dataset.csv"
VIDEO_DIR = ROOT / "02_data/raw/videos"
OUT_DIR = ROOT / "02_data/kaggle/tiktok-audio-v1"
AUDIO_DIR = OUT_DIR / "audio"

MIN_DURATION_SEC = 1.0

# Ten file co dang: <username>__<YYYYMMDD>_<HHMMSS>__<video_id>.mp4 (giong transcribe_all.py)
NAME_RE = re.compile(r"^(?P<username>.+)__(?P<stamp>\d{8}_\d{6})__(?P<video_id>\d+)\.mp4$")


def human(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return "%dh%02dm%02ds" % (h, m, s) if h else "%dm%02ds" % (m, s)


def load_target_video_ids(csv_path=DATASET_CSV):
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [row["video_id"] for row in reader]


def build_video_index():
    """video_id -> Path(.mp4). Slideshow (thu muc, khong phai .mp4) khong co trong index."""
    index = {}
    for path in VIDEO_DIR.glob("*.mp4"):
        m = NAME_RE.match(path.name)
        if m:
            index[m.group("video_id")] = path
    return index


def extract_one(ffmpeg_exe, src_path, dst_path):
    cmd = [
        ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src_path),
        "-vn", "-ar", "16000", "-ac", "1", "-b:a", "64k",
        str(dst_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip()[-500:] or "ffmpeg loi khong ro nguyen nhan")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="chi xu ly N video dau tien")
    parser.add_argument("--overwrite", action="store_true", help="tach lai ca video da co audio")
    parser.add_argument("--ids", default=str(DATASET_CSV), help="CSV co cot video_id (mac dinh 02_data/kaggle/tiktok-multilabel-content/dataset.csv)")
    parser.add_argument("--out", default=str(OUT_DIR), help="Thu muc dau ra (mac dinh 02_data/kaggle/tiktok-audio-v1/)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    print("ffmpeg:", ffmpeg_exe)

    target_ids = load_target_video_ids(args.ids)
    if args.limit:
        target_ids = target_ids[: args.limit]
    print("So video can tach audio (theo %s): %d" % (args.ids, len(target_ids)))

    video_index = build_video_index()
    print("So file .mp4 tim thay trong %s: %d" % (VIDEO_DIR, len(video_index)))

    index_rows = []
    error_rows = []
    started = time.time()
    ok = skip = fail = 0

    for i, video_id in enumerate(target_ids, 1):
        dst_path = audio_dir / ("%s.mp3" % video_id)

        src_path = video_index.get(video_id)
        if src_path is None:
            error_rows.append({"video_id": video_id, "status": "missing_video_file",
                                "detail": "khong tim thay .mp4 (co the la slideshow, khong co audio)"})
            fail += 1
            continue

        if dst_path.exists() and not args.overwrite:
            skip += 1
        else:
            try:
                extract_one(ffmpeg_exe, src_path, dst_path)
            except Exception as e:
                error_rows.append({"video_id": video_id, "status": "ffmpeg_error", "detail": str(e)})
                fail += 1
                continue

        try:
            duration = MP3(dst_path).info.length
        except Exception as e:
            error_rows.append({"video_id": video_id, "status": "probe_error", "detail": str(e)})
            fail += 1
            continue

        if duration < MIN_DURATION_SEC:
            error_rows.append({"video_id": video_id, "status": "too_short",
                                "detail": "duration=%.2fs" % duration})
            # van giu file + dua vao index - Kaggle notebook se tu danh dau
            # transcript_status=empty_or_music cho truong hop nay, khong loai o day.

        index_rows.append({"video_id": video_id, "audio_filename": dst_path.name,
                            "duration_seconds": round(duration, 2)})
        ok += 1

        if i % 50 == 0 or i == len(target_ids):
            elapsed = time.time() - started
            eta = elapsed / i * (len(target_ids) - i)
            print("[%d/%d] ok=%d skip=%d fail=%d | da chay %s | con lai ~%s" % (
                i, len(target_ids), ok, skip, fail, human(elapsed), human(eta)), flush=True)

    with open(out_dir / "audio_index.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "audio_filename", "duration_seconds"])
        writer.writeheader()
        writer.writerows(index_rows)
    print("\n>> Da ghi %s (%d dong)" % (out_dir / "audio_index.csv", len(index_rows)))

    if error_rows:
        with open(out_dir / "audio_extract_errors.csv", "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "status", "detail"])
            writer.writeheader()
            writer.writerows(error_rows)
        print(">> Da ghi %s (%d dong)" % (out_dir / "audio_extract_errors.csv", len(error_rows)))

    total_bytes = sum(p.stat().st_size for p in audio_dir.glob("*.mp3"))
    print("\n>> Tong dung luong audio/: %.1f MB (%d file)" % (
        total_bytes / 1024 / 1024, len(list(audio_dir.glob("*.mp3")))))
    if out_dir.resolve() != OUT_DIR.resolve():
        return  # thu muc phu (vd. 02_data/interim/audio_others/) - khong ghi metadata Kaggle

    meta = {
        "title": "TikTok Multilabel Content - Audio (16kHz mono)",
        "id": "cngngquc/tiktok-audio-v1",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (OUT_DIR / "dataset-metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(">> Da ghi %s (sua truong 'id' voi username Kaggle that truoc khi upload)"
          % (OUT_DIR / "dataset-metadata.json"))
    print(">> XONG. Thu muc san sang upload Kaggle: %s" % OUT_DIR)


if __name__ == "__main__":
    main()
