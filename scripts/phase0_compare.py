"""Giai đoạn 0: so yt-dlp với dataset v5 (Apify) trên cùng video_id — dùng để chốt công cụ thu thập.

    python scripts/phase0_compare.py                       # tải (bỏ qua bài đã có) rồi so
    python scripts/phase0_compare.py --skip-download       # chỉ so trên phase0/out/
    python scripts/phase0_compare.py --browser ""          # không dùng cookie

Input:  phase0_urls.txt, phase0_ref.csv (video_id, caption, hashtag, thumb_md5, thumb_wh, f1..f7_dhash,
        tuỳ chọn duration_manifest). Thư mục input lấy theo --dir (mặc định phase0/phase0_inputs/).
Output: phase0/out/ (theo --out) — {id}.mp4, {id}.info.json, {id}.{thumb_id}.* (writeallthumbnails),
                      covers/{id}.{jpg|png} (nguyên bytes ảnh bìa id="cover"), archive.txt, download_log.csv
        phase0/out/phase0_compare.csv (từng video) + phase0_report.txt (tóm tắt)
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yt_dlp
from PIL import Image

FORMAT = "b[vcodec^=h264][format_id!*=download]/b[format_id!*=download]/b"  # tránh bản "download" có watermark
MAGIC = {b"\xff\xd8\xff": "jpg", b"\x89PNG\r\n\x1a\n": "png"}
ID_RE = re.compile(r"/(?:video|photo)/(\d+)")


# ---------- tiện ích ----------
def image_ext(data: bytes) -> str:
    for m, ext in MAGIC.items():
        if data.startswith(m):
            return ext
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    raise ValueError(f"không nhận ra định dạng ảnh: {data[:12]!r}")


def dhash(img: Image.Image, n: int = 8) -> str:
    g = np.asarray(img.convert("L").resize((n + 1, n), Image.LANCZOS), dtype=np.int16)
    return "".join("1" if b else "0" for b in (g[:, 1:] > g[:, :-1]).flatten())


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def norm_ws(s) -> str:
    return " ".join(str(s or "").replace("\u200b", " ").split())  # ký tự ẩn zero-width space


def strip_tags(s) -> str:
    return norm_ws(re.sub(r"#\S+", " ", str(s or "")))


def parse_tags(v) -> set[str]:
    """Nhận cả "['a', 'b']" lẫn "a;b" (định dạng thực tế của phase0_ref.csv)."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
        return set()
    s = str(v).strip()
    items = ast.literal_eval(s) if s.startswith("[") else re.split(r"[;,|]", s)
    items = [str(t).replace("\u200b", "") for t in items]
    return {str(t).strip().lstrip("#").lower() for t in items if str(t).strip().lstrip("#")}


def wh_of(s) -> tuple[int, int] | None:
    m = re.findall(r"\d+", str(s or ""))
    return (int(m[0]), int(m[1])) if len(m) >= 2 else None


class ErrLogger:
    """Giữ lỗi cuối cùng để ghi lý do khi ignoreerrors nuốt exception."""

    def __init__(self):
        self.last = ""

    def debug(self, msg):
        pass

    def info(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        self.last = re.sub(r"\s+", " ", str(msg)).replace("ERROR: ", "").strip()


# ---------- 1. tải ----------
def make_ydl(out: Path, browser: str, smin: float, smax: float, log: ErrLogger) -> yt_dlp.YoutubeDL:
    opts = {
        "format": FORMAT,
        "outtmpl": {"default": str(out / "%(id)s.%(ext)s")},
        "writeinfojson": True,
        "write_all_thumbnails": True,  # = --write-all-thumbnails
        "download_archive": str(out / "archive.txt"),
        "ignoreerrors": True,
        "sleep_interval": smin,
        "max_sleep_interval": smax,
        "retries": 3,
        "logger": log,
        "noprogress": True,
    }
    if browser:
        try:
            ydl = yt_dlp.YoutubeDL({**opts, "cookiesfrombrowser": (browser,)})
            _ = len(ydl.cookiejar)  # cookie nạp lười: ép nạp để biết có lỗi không
            print(f"cookie: dùng {browser}")
            return ydl
        except Exception as e:  # noqa: BLE001 — mọi lỗi đọc cookie đều chuyển sang chạy không cookie
            print(f"cookie: không đọc được {browser} ({str(e)[:90]}…) → chạy không cookie")
    return yt_dlp.YoutubeDL(opts)


def save_cover(info: dict, vid: str, out: Path, ydl: yt_dlp.YoutubeDL) -> str:
    th = next((t for t in info.get("thumbnails") or [] if t.get("id") == "cover"), None)
    if th is None:
        return "không có thumbnail id=cover"
    fp = th.get("filepath")
    data = Path(fp).read_bytes() if fp and Path(fp).exists() else ydl.urlopen(th["url"]).read()
    (out / "covers" / f"{vid}.{image_ext(data)}").write_bytes(data)
    return ""


def download(urls: list[str], out: Path, args) -> None:
    (out / "covers").mkdir(parents=True, exist_ok=True)
    log_path = out / "download_log.csv"
    done = {}
    if log_path.exists():
        with log_path.open(encoding="utf-8-sig") as f:
            done = {r["video_id"]: r for r in csv.DictReader(f)}
    log = ErrLogger()
    ydl = make_ydl(out, args.browser, args.sleep_min, args.sleep_max, log)
    t0 = time.time()
    with ydl:
        for i, url in enumerate(urls, 1):
            vid = ID_RE.search(url).group(1)
            have = (out / f"{vid}.info.json").exists() and any((out / "covers").glob(f"{vid}.*"))
            if have and done.get(vid, {}).get("status") == "ok":
                continue
            log.last = ""
            t = time.time()
            info = ydl.extract_info(url, download=True)
            row = {"video_id": vid, "url": url, "status": "ok", "reason": "", "seconds": 0}
            if info is None:
                if have:
                    row["reason"] = "đã có trong archive"
                else:
                    row.update(status="error", reason=log.last or "yt-dlp trả None (có thể nằm trong archive)")
            else:
                try:
                    err = save_cover(info, vid, out, ydl)
                except Exception as e:  # noqa: BLE001 — ghi lý do, không dừng cả lô
                    err = f"lưu cover lỗi: {e}"
                if err:
                    row.update(status="cover_error", reason=err)
            row["seconds"] = round(time.time() - t, 1)
            done[vid] = row
            print(f"[{i}/{len(urls)}] {vid} {row['status']} {row['reason'][:100]}", flush=True)
            with log_path.open("w", newline="", encoding="utf-8-sig") as f:  # ghi lại sau mỗi bài
                w = csv.DictWriter(f, fieldnames=["video_id", "url", "status", "reason", "seconds"])
                w.writeheader()
                w.writerows(done.values())
    print(f"tải xong trong {time.time() - t0:.0f}s")


# ---------- 2–4. so ----------
def frames_dhash(mp4: Path, duration: float) -> dict[str, list]:
    """Hai cách cắt: theo ms từ info.duration (spec) và theo chỉ số frame (check_frames.py)."""
    res = {"ms": [None] * 7, "idx": [None] * 7}
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        return res
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for k in range(1, 8):
        for mode in ("ms", "idx"):
            if mode == "ms":
                cap.set(cv2.CAP_PROP_POS_MSEC, duration * k / 8 * 1000)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(int(k / 8 * n), max(n - 1, 0)))
            ok, fr = cap.read()
            if ok and fr is not None:
                res[mode][k - 1] = dhash(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    cap.release()
    return res


def compare_one(r: pd.Series, out: Path) -> dict:
    vid = r["video_id"]
    row = {"video_id": vid, "username": r["username"], "downloaded": False}
    ip = out / f"{vid}.info.json"
    if not ip.exists():
        return row
    info = json.loads(ip.read_text(encoding="utf-8"))
    row["downloaded"] = True
    row["vcodec"] = info.get("vcodec")

    # caption
    ref_c, new_c = r.get("caption"), info.get("description")
    row["caption_exact"] = norm_ws(ref_c) == norm_ws(new_c)
    row["caption_notags"] = strip_tags(ref_c) == strip_tags(new_c)
    # hashtag
    # yt-dlp TikTok không có info["tags"] → fallback: các #tag trong description
    tags = info.get("tags") or re.findall(r"#([^\s#]+)", str(new_c or ""))
    row["hashtag_src"] = "tags" if info.get("tags") else "description"
    ref_t, new_t = parse_tags(r.get("hashtag")), parse_tags(tags)
    row["hashtag_match"] = ref_t == new_t
    row["hashtag_missing"] = "|".join(sorted(ref_t - new_t))
    row["hashtag_extra"] = "|".join(sorted(new_t - ref_t))
    # duration
    row["duration"] = info.get("duration")
    if "duration_manifest" in r and str(r["duration_manifest"]).strip() not in ("", "nan") and info.get("duration"):
        row["duration_diff"] = abs(float(info["duration"]) - float(r["duration_manifest"]))
    # cover
    covers = sorted((out / "covers").glob(f"{vid}.*"))
    if covers:
        data = covers[0].read_bytes()
        row["cover_fmt"] = covers[0].suffix.lstrip(".")
        row["cover_md5_match"] = hashlib.md5(data).hexdigest() == str(r.get("thumb_md5", "")).strip().lower()
        with Image.open(covers[0]) as im:
            row["cover_wh"] = "%dx%d" % im.size
        row["cover_wh_match"] = wh_of(row["cover_wh"]) == wh_of(r.get("thumb_wh"))
    # frames
    mp4 = out / f"{vid}.mp4"
    if info.get("vcodec") not in (None, "none") and mp4.exists():
        fh = frames_dhash(mp4, float(info.get("duration") or 0))
        for mode in ("ms", "idx"):
            ds = [hamming(h, str(r[f"f{k}_dhash"]).strip()) if h else None for k, h in enumerate(fh[mode], 1)]
            for k, d in enumerate(ds, 1):
                row[f"f{k}_{mode}"] = d
            ok = [d for d in ds if d is not None]
            row[f"frames_{mode}_mean"] = round(sum(ok) / len(ok), 2) if ok else None
    return row


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.1f}%)" if d else "0/0"


def summarize(rows: list[dict], log_path: Path, n_urls: int) -> str:
    df = pd.DataFrame(rows)
    dl = df[df["downloaded"]]
    L = [f"Giai đoạn 0 — yt-dlp vs dataset v5 ({len(df)} video trong ref, {n_urls} URL)", ""]
    L.append(f"Tải được (có info.json): {pct(len(dl), len(df))}")
    if log_path.exists():
        lg = pd.read_csv(log_path, dtype=str, keep_default_na=False)
        bad = lg[lg.status != "ok"]
        if len(bad):
            L.append("Lỗi tải:")
            for _, b in bad.iterrows():
                L.append(f"  {b.video_id}: {b.status} — {b.reason[:150]}")
    if not len(dl):
        return "\n".join(L)
    vids = dl[dl["vcodec"].fillna("none") != "none"]
    L += ["", f"Loại: video {len(vids)}, không có luồng hình (slideshow/audio) {len(dl) - len(vids)}", ""]
    L.append(f"Caption khớp tuyệt đối:          {pct(int(dl.caption_exact.sum()), len(dl))}")
    L.append(f"Caption khớp sau khi bỏ #hashtag: {pct(int(dl.caption_notags.sum()), len(dl))}")
    L.append(f"Hashtag khớp (tập hợp):          {pct(int(dl.hashtag_match.sum()), len(dl))}")
    if "duration_diff" in dl and dl["duration_diff"].notna().any():
        d = dl["duration_diff"].dropna()
        L.append(f"Lệch thời lượng: median {d.median():.2f}s, max {d.max():.2f}s")
    else:
        L.append("Thời lượng: ref không có duration_manifest → bỏ qua")
    if "cover_md5_match" in dl:
        c = dl[dl["cover_md5_match"].notna()]
        L += ["", f"Ảnh bìa: {len(c)} ảnh, định dạng {c.cover_fmt.value_counts().to_dict()}"]
        L.append(f"  MD5 khớp (byte-identical): {pct(int(c.cover_md5_match.astype(bool).sum()), len(dl))}")
        L.append(f"  Kích thước khớp:           {pct(int(c.cover_wh_match.astype(bool).sum()), len(dl))}")
    for mode, name in (("ms", "seek ms theo info.duration (spec)"), ("idx", "seek theo chỉ số frame")):
        cols = [f"f{k}_{mode}" for k in range(1, 8) if f"f{k}_{mode}" in vids]
        if not cols:
            continue
        allv = vids[cols].stack().astype(int)
        L += ["", f"Frame dHash — {name}: {len(allv)} frame / {len(vids)} video"]
        L.append(f"  Hamming median {allv.median():.0f}, mean {allv.mean():.2f}, max {allv.max()}")
        for t in (0, 5, 10):
            L.append(f"  ≤ {t:>2}: {pct(int((allv <= t).sum()), len(allv))}")
        per_k = " ".join(f"f{k}={vids[f'f{k}_{mode}'].median():.0f}" for k in range(1, 8) if f"f{k}_{mode}" in vids)
        L.append(f"  median theo vị trí: {per_k}")
        worst = vids.nlargest(5, f"frames_{mode}_mean")[["video_id", f"frames_{mode}_mean"]]
        L.append("  lệch nhiều nhất: " + ", ".join(f"{a} ({b})" for a, b in worst.itertuples(index=False)))
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path(__file__).resolve().parent.parent / "phase0" / "phase0_inputs", help="thư mục chứa phase0_urls.txt / phase0_ref.csv")
    ap.add_argument("--out", type=Path, default=None, help="mặc định phase0/out")
    ap.add_argument("--browser", default="chrome")
    ap.add_argument("--sleep-min", type=float, default=3)
    ap.add_argument("--sleep-max", type=float, default=8)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    out = args.out or Path("phase0") / "out"
    out.mkdir(parents=True, exist_ok=True)
    urls = [u.strip() for u in (args.dir / "phase0_urls.txt").read_text(encoding="utf-8-sig").splitlines()
            if u.strip().startswith("http")]
    if not args.skip_download:
        download(urls, out, args)

    ref = pd.read_csv(args.dir / "phase0_ref.csv", dtype=str, keep_default_na=False)
    rows = []
    for i, (_, r) in enumerate(ref.iterrows(), 1):
        rows.append(compare_one(r, out))
        if i % 25 == 0:
            print(f"so {i}/{len(ref)}", flush=True)
    pd.DataFrame(rows).to_csv(out / "phase0_compare.csv", index=False, encoding="utf-8-sig")
    rep = summarize(rows, out / "download_log.csv", len(urls))
    (out / "phase0_report.txt").write_text(rep, encoding="utf-8")
    print(rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
