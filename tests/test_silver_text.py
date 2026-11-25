from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

import pytest

from vicap.silver.text import caption_hashtag, caption_hashtags, clean_caption


@pytest.mark.parametrize(
    ("desc", "caption", "tags"),
    [
        ("hello #food #Ăn_ngon", "hello", ["food", "Ăn_ngon"]),
        # giữ xuống dòng; tag giữa câu thành một dấu cách (như v5: "…😋 \n hồi Aqua")
        ("Cá Hồi 😋 \n#Cá hồi Aqua", "Cá Hồi 😋 \n hồi Aqua", ["Cá"]),
        # ​ không thuộc \w: tag dừng trước nó, ​ được giữ
        ("rồi :3 #Honda​ #todidau", "rồi :3 ​", ["Honda", "todidau"]),
        ("Động lực #", "Động lực #", []),  # '#' đứng một mình giữ nguyên
        ("#a #b #a", "", ["a", "b", "a"]),  # giữ trùng như cột hashtag của v5
        ("a\t\t b  c", "a b c", []),
        ("", "", []),
        (None, "", []),
    ],
)
def test_caption_rules(desc: str | None, caption: str, tags: list[str]) -> None:
    assert clean_caption(desc) == caption
    assert caption_hashtags(desc) == tags


def test_caption_hashtag_matches_notebook() -> None:
    assert caption_hashtag("hello", ["food", "vn"]) == "hello food vn"
    assert caption_hashtag("", ["food"]) == "food"
    assert caption_hashtag("hello", []) == "hello"
    assert caption_hashtag(None, None) == ""


# ── Đối chiếu với dataset v5 (dữ liệu local, không có trên git): bỏ qua khi thiếu ──
PHASE0 = Path(os.getenv("VICAP_PHASE0_OUT", "phase0/out"))
V5_CSV = Path(os.getenv("VICAP_V5_CSV", "data/kaggle/v5/dataset.csv"))


def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


@pytest.mark.skipif(not (PHASE0.is_dir() and V5_CSV.is_file()), reason="thiếu phase0/out hoặc dataset v5")
def test_matches_dataset_v5() -> None:
    with V5_CSV.open(encoding="utf-8-sig") as f:
        v5 = {r["video_id"]: r for r in csv.DictReader(f)}
    pairs = []
    for p in sorted(PHASE0.glob("*.info.json")):
        vid = p.name.split(".")[0]
        if vid in v5:
            pairs.append((json.loads(p.read_text(encoding="utf-8")).get("description") or "", v5[vid]))
    assert len(pairs) >= 100
    for desc, row in pairs:
        assert caption_hashtags(desc) == (row["hashtag"].split(";") if row["hashtag"] else []), row["video_id"]
        # yt-dlp của Giai đoạn 0 đổi \n thành dấu cách → so sau khi chuẩn hoá khoảng trắng
        assert _ws(clean_caption(desc)) == _ws(row["caption"]), row["video_id"]
