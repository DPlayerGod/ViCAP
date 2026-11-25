"""Log bằng Loguru cho mọi bước của ViCAP.

Code trong ``vicap`` dùng thẳng ``from loguru import logger`` (định dạng ``{}``: ``logger.info("x {}", x)``).
Thư viện (kafka-python, pyiceberg, s3fs, …) vẫn log qua ``logging`` chuẩn; ``setup_logging()`` chuyển các
log đó sang Loguru để mọi dòng cùng một định dạng.

Màu chỉ bật khi stderr là terminal: log trong Airflow / file vẫn là chữ trơn, không dính mã ANSI.
Gọi ``setup_logging()`` một lần ở điểm vào (``main()`` của ``vicap.pipelines.*``), không gọi trong thư viện.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from typing import TextIO

from loguru import logger

FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <7}</level> | "
    "<cyan>{name}</cyan> - <level>{message}</level>"
)
# kafka-python log mỗi lần kết nối / đổi metadata ở mức INFO; py4j (pyspark) log mỗi lệnh gửi JVM
QUIET: Mapping[str, str] = {"kafka": "WARNING", "py4j": "WARNING"}


class InterceptHandler(logging.Handler):
    """Chuyển record của ``logging`` chuẩn sang Loguru, giữ đúng level và vị trí gọi."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 0
        while frame is not None and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(
    level: str = "INFO",
    *,
    colorize: bool | None = None,
    quiet: Mapping[str, str] = QUIET,
    sink: TextIO | None = None,
) -> None:
    """Cấu hình log cho một lần chạy. ``colorize=None``: tự bật màu khi ``sink`` (mặc định stderr) là terminal."""
    for stream in (sys.stdout, sys.stderr):  # console Windows mặc định cp1252: không in được tiếng Việt
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logger.remove()
    logger.add(sink or sys.stderr, level=level, format=FORMAT, colorize=colorize)
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for name, lvl in quiet.items():
        logging.getLogger(name).setLevel(lvl)
