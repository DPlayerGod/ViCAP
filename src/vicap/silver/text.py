"""Caption / hashtag cho ``silver.videos``: dựng lại đúng cột ``caption`` / ``hashtag`` của dataset v5.

Model P1 học trên text ``caption + " " + " ".join(hashtag)`` của v5 (PLAN mục 4.1), nên Silver phải làm sạch
``desc`` theo đúng cách đã tạo v5. Quy tắc dưới đây đã so với 186 video Giai đoạn 0 (``phase0/out/*.info.json``
so với ``dataset.csv`` v5):

- ``hashtags``: mọi ``#(\\w+)`` trong ``desc`` theo thứ tự xuất hiện, **giữ trùng**, giữ chữ hoa / thường
  (186/186 khớp tuyệt đối, cả thứ tự). ``\\w`` là của ``re`` Python (Unicode); ``\\u200b`` không thuộc ``\\w``
  nên ``#Tag\\u200b`` chỉ lấy ``Tag``.
- ``caption``: thay mỗi ``#\\w+`` bằng một dấu cách, gộp dấu cách / tab liên tiếp, ``strip``; **giữ xuống
  dòng** (v5 có ``"…😋 \\n hồi Aqua"``). Dấu ``#`` đứng một mình được giữ lại như v5. 186/186 khớp sau khi
  chuẩn hoá khoảng trắng (yt-dlp của Giai đoạn 0 đã đổi ``\\n`` thành dấu cách, ``desc`` của itemStruct thì không).

Khác ``vicap.ingestion.parse.extract_hashtags`` (``#([^\\s#]+)``, bỏ trùng): cột ``hashtags`` của Bronze là
metadata để tra cứu, còn text vào model lấy từ đây.

Spark gọi các hàm này qua UDF Python thay cho ``regexp_replace``: ``\\w`` của Java chỉ là ASCII (``#Cá`` chỉ bắt
``C``), thêm ``(?U)`` thì lại khác Python ở dấu kết hợp và ``\\u200d``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

HASHTAG = re.compile(r"#(\w+)")
_SPACES = re.compile(r"[ \t]+")


def clean_caption(desc: str | None) -> str:
    if not desc:
        return ""
    return _SPACES.sub(" ", HASHTAG.sub(" ", desc)).strip()


def caption_hashtags(desc: str | None) -> list[str]:
    return HASHTAG.findall(desc or "")


def caption_hashtag(caption: str | None, hashtags: Sequence[str] | None) -> str:
    """Text cho ``text_emb[0]`` (PLAN mục 4.1), như ``df["caption_hashtag"]`` của notebook."""
    return f"{caption or ''} {' '.join(hashtags or ())}".strip()
