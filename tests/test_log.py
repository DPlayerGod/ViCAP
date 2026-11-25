from __future__ import annotations

import io
import logging
import sys

import pytest
from loguru import logger

from vicap.utils.log import setup_logging


@pytest.fixture
def buf() -> io.StringIO:
    out = io.StringIO()
    yield out
    logger.remove()
    logger.add(sys.stderr)  # trả lại handler mặc định cho các test sau
    logging.basicConfig(handlers=[logging.NullHandler()], force=True)


def test_stdlib_logs_go_through_loguru(buf: io.StringIO) -> None:
    setup_logging(colorize=False, sink=buf)
    logger.info("crawl {} bài", 3)
    logging.getLogger("pyiceberg.io").warning("thư viện %s", "cảnh báo")
    logging.getLogger("kafka.conn").info("kết nối")  # QUIET: kafka chỉ từ WARNING
    out = buf.getvalue()
    assert "| INFO    | test_log - crawl 3 bài" in out
    assert "| WARNING | test_log - thư viện cảnh báo" in out  # vị trí gọi, không phải vicap.utils.log
    assert "kết nối" not in out
    assert "\x1b[" not in out


def test_colorize(buf: io.StringIO) -> None:
    setup_logging(colorize=True, sink=buf)
    logger.error("lỗi")
    assert "\x1b[" in buf.getvalue()
