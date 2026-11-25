from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from vicap import config as cfg


def test_import_does_not_require_env() -> None:
    """Import toàn bộ module trong tiến trình mới, không có biến bí mật nào."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AWS_", "CLICKHOUSE_"))}
    modules = ["vicap.config", "vicap.iceberg.catalog", "vicap.iceberg.writer", "vicap.iceberg.tables",
               "vicap.storage.objects", "vicap.storage.quarantine", "vicap.clickhouse.client", "vicap.quality",
               "vicap.spark"]
    code = "import importlib\n" + "\n".join(f"importlib.import_module({m!r})" for m in modules)
    src = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = str(src)
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=src)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("factory", [cfg.MinioConfig, cfg.IcebergConfig])
def test_missing_secret_raises_at_init(monkeypatch: pytest.MonkeyPatch, factory) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    with pytest.raises(cfg.MissingEnvError, match="AWS_SECRET_ACCESS_KEY"):
        factory()


def test_clickhouse_requires_password(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLICKHOUSE_USER", "vicap")
    monkeypatch.delenv("CLICKHOUSE_PASSWORD", raising=False)
    with pytest.raises(cfg.MissingEnvError, match="CLICKHOUSE_PASSWORD"):
        cfg.ClickHouseConfig()


def test_env_read_at_init_not_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "first")
    assert cfg.MinioConfig().secret_key == "first"
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "second")
    assert cfg.MinioConfig().secret_key == "second"


def test_no_default_passwords_in_config() -> None:
    source = importlib.import_module("vicap.config").__file__
    text = open(source, encoding="utf-8").read()
    assert "admin123" not in text


def test_model_defaults_match_plan() -> None:
    m = cfg.ModelConfig()
    assert (m.threshold, m.at_least_one, m.n_labels) == (0.75, True, 20)
    assert (m.n_text, m.text_dim, m.n_images, m.image_dim) == (2, 768, 8, 768)
    assert cfg.MlflowConfig().model_uri == "models:/kltn-p1@champion"


def test_crawler_minio_config_uses_crawler_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIO_CRAWLER_ACCESS_KEY", "vicap-crawler")
    monkeypatch.setenv("MINIO_CRAWLER_SECRET_KEY", "c-secret")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)  # không cần key pipeline
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    c = cfg.crawler_minio_config()
    assert (c.access_key, c.secret_key, c.raw_bucket) == ("vicap-crawler", "c-secret", "raw")


def test_crawler_minio_config_requires_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIO_CRAWLER_ACCESS_KEY", "vicap-crawler")
    monkeypatch.delenv("MINIO_CRAWLER_SECRET_KEY", raising=False)
    with pytest.raises(cfg.MissingEnvError, match="MINIO_CRAWLER_SECRET_KEY"):
        cfg.crawler_minio_config()
