"""Giai đoạn 0: trích 8 ảnh từ mp4 yt-dlp theo PLAN mục 4.1 và kiểm tra đọc được.

    python scripts/check_frames.py test_out/

Với mỗi `videos/{stem}.mp4`:
    frames/{video_id}_0.{jpg|png} = bản sao nguyên bytes của thumbnails/{stem}.*
    frames/{video_id}_1..7.jpg    = frame tại i/8 thời lượng (12,5% … 87,5%)

Ghi `frames_report.csv` + in tóm tắt. Đây là cài đặt theo spec, chưa phải `extract_frame.py` cũ:
khi có `extract_frame.py`, chạy nó trên cùng thư mục `videos/` rồi so ảnh với `frames/`.
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import cv2

POSITIONS = [i / 8 for i in range(1, 8)]  # 0.125 … 0.875


def parse_stem(name: str) -> tuple[str, str, str]:
    """`{username}__{YYYYMMDD_HHMMSS}__{video_id}` → (username, posted_at, video_id)."""
    stem = name.rsplit(".", 1)[0] if "." in name else name
    if "__img_" in stem:
        stem = stem.rsplit("__img_", 1)[0]
    username, posted_at, video_id = stem.rsplit("__", 2)
    return username, posted_at, video_id


def extract(mp4: Path, out_dir: Path, video_id: str) -> dict:
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        return {"ok": False, "error": "không mở được"}
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    saved = 0
    for i, p in enumerate(POSITIONS, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(int(p * n), max(n - 1, 0)))
        ok, frame = cap.read()
        if ok and frame is not None:
            cv2.imwrite(str(out_dir / f"{video_id}_{i}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
    cap.release()
    return {"ok": saved == 7, "frames": n, "fps": round(fps, 2), "cv_duration": round(n / fps, 2) if fps else None,
            "width": w, "height": h, "saved": saved, "error": "" if saved == 7 else f"chỉ đọc được {saved}/7 frame"}


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "test_out")
    out_dir = root / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {}
    if (root / "metadata.csv").exists():
        with (root / "metadata.csv").open(encoding="utf-8-sig") as f:
            meta = {r["video_id"]: r for r in csv.DictReader(f)}

    rows = []
    for mp4 in sorted((root / "videos").glob("*.mp4")):
        _, _, vid = parse_stem(mp4.name)
        r = {"video_id": vid, **extract(mp4, out_dir, vid)}
        thumbs = sorted((root / "thumbnails").glob(f"{mp4.stem}.*"))  # .jpg hoặc .png (bytes gốc)
        thumb = thumbs[0] if thumbs else None
        r["thumb_ok"] = thumb is not None and cv2.imread(str(thumb)) is not None
        if r["thumb_ok"]:
            shutil.copyfile(thumb, out_dir / f"{vid}_0{thumb.suffix}")  # giữ nguyên bytes, không đổi định dạng
        r["n_images"] = (len(list(out_dir.glob(f"{vid}_0.*")))
                         + sum((out_dir / f"{vid}_{i}.jpg").exists() for i in range(1, 8)))
        md = meta.get(vid, {}).get("duration")
        r["meta_duration"] = int(md) if md else None
        r["duration_diff"] = (round(abs(r["cv_duration"] - r["meta_duration"]), 2)
                              if r.get("cv_duration") and r["meta_duration"] else None)
        rows.append(r)

    keys = ["video_id", "ok", "thumb_ok", "n_images", "frames", "fps", "cv_duration", "meta_duration",
            "duration_diff", "width", "height", "saved", "error"]
    with (root / "frames_report.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    full = sum(r["n_images"] == 8 for r in rows)
    diffs = [r["duration_diff"] for r in rows if r["duration_diff"] is not None]
    print(f"mp4: {len(rows)}; đủ 8 ảnh (_0…_7): {full}/{len(rows)}")
    if diffs:
        print(f"lệch thời lượng OpenCV vs metadata: max {max(diffs)}s, trung bình {sum(diffs) / len(diffs):.2f}s")
    for r in rows:
        if r["n_images"] != 8:
            print(f"  LỖI {r['video_id']}: {r.get('error')} thumb_ok={r['thumb_ok']}")
    return 0 if full == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
