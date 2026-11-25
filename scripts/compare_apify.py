"""Giai đoạn 0: đối chiếu metadata yt-dlp (test_out/metadata.csv) với dữ liệu Apify cùng video_id.

    python scripts/compare_apify.py test_out/metadata.csv path/to/apify.json   # export Apify (JSON list)
    python scripts/compare_apify.py test_out/metadata.csv path/to/metadata_v5.csv

Nhận JSON export của Apify (`id`, `text`, `hashtags[].name`, `videoMeta.duration`, `createTime`,
`authorMeta.name`) hoặc CSV phẳng; tên cột được dò theo danh sách ứng viên bên dưới.
So: caption (desc), tập hashtag, thời lượng (lệch ≤ 1s), tên file stem (múi giờ UTC).
Ghi `compare_apify.csv` cạnh metadata.csv và in tóm tắt.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

CANDIDATES = {
    "video_id": ["id", "video_id", "aweme_id"],
    "desc": ["text", "desc", "caption", "description"],
    "hashtags": ["hashtags", "hashtag", "challenges"],
    "duration": ["videoMeta.duration", "duration", "video_duration"],
    "create_time": ["createTime", "create_time", "timestamp"],
    "username": ["authorMeta.name", "username", "author", "uploader"],
}


def flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out


def pick(row: dict, field: str):
    for c in CANDIDATES[field]:
        if c in row and row[c] not in (None, ""):
            return row[c]
    return None


def norm_text(s) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s or ""))).strip()


def norm_tags(v) -> set[str]:
    if v is None:
        return set()
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            v = re.split(r"[|,;\s]+", v)
    tags = [t.get("name") or t.get("title") if isinstance(t, dict) else t for t in v]
    return {unicodedata.normalize("NFC", str(t)).lstrip("#").lower() for t in tags if t}


def load_ref(path: Path) -> dict[str, dict]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = [flat(x) for x in (data if isinstance(data, list) else data.get("items", []))]
    else:
        with path.open(encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    return {str(pick(r, "video_id")): r for r in rows if pick(r, "video_id")}


def main() -> int:
    ours_path, ref_path = Path(sys.argv[1]), Path(sys.argv[2])
    with ours_path.open(encoding="utf-8-sig") as f:
        ours = {r["video_id"]: r for r in csv.DictReader(f)}
    ref = load_ref(ref_path)
    common = sorted(set(ours) & set(ref))
    print(f"yt-dlp: {len(ours)} bài; tham chiếu: {len(ref)} bài; chung video_id: {len(common)}")
    if not common:
        return 1

    out, agg = [], {"desc": 0, "hashtags": 0, "duration": 0, "stem": 0}
    for vid in common:
        a, b = ours[vid], ref[vid]
        r = {"video_id": vid}
        r["desc_eq"] = norm_text(a["desc"]) == norm_text(pick(b, "desc"))
        r["hashtags_eq"] = set(filter(None, a["hashtags"].lower().split("|"))) == norm_tags(pick(b, "hashtags"))
        da, db = a["duration"], pick(b, "duration")
        r["duration_eq"] = bool(da and db) and abs(float(da) - float(db)) <= 1
        ct, user = pick(b, "create_time"), pick(b, "username")
        if ct and user:
            ts = int(ct) if str(ct).isdigit() else int(datetime.fromisoformat(str(ct).replace("Z", "+00:00")).timestamp())
            ref_stem = f"{user}__{datetime.fromtimestamp(ts, timezone.utc):%Y%m%d_%H%M%S}__{vid}"
            r["stem_eq"] = ref_stem == a["stem"]
        else:
            r["stem_eq"] = None
        r.update(ours_desc=a["desc"], ref_desc=pick(b, "desc"), ours_tags=a["hashtags"],
                 ref_tags="|".join(sorted(norm_tags(pick(b, "hashtags")))), ours_dur=da, ref_dur=db)
        for k in agg:
            agg[k] += bool(r[f"{k}_eq"])
        out.append(r)

    dst = ours_path.with_name("compare_apify.csv")
    with dst.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for k, v in agg.items():
        print(f"- {k} khớp: {v}/{len(common)} ({100 * v / len(common):.1f}%)")
    print(f"Chi tiết: {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
