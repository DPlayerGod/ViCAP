"""Đăng ký P1 (``final_model.pt`` của notebook v12) vào MLflow: ``kltn-p1``, alias ``@champion`` (PLAN mục 5).

Model log theo kiểu "models from code" (``mlflow_p1.py``) + ``code_paths=[kltn_model.py]`` + artifact
``checkpoint``; load lại chỉ cần mlflow, torch, numpy. Run ghi param của checkpoint (kiến trúc, chuẩn hoá,
ngưỡng, ``n_train``), danh sách nhãn và SHA-256 của file.

Idempotent: checkpoint cùng SHA-256 đã có phiên bản thì không đăng ký thêm, chỉ gán alias. Chạy trong gpu-worker
(có torch + mlflow, gọi được ``http://mlflow:5000``)::

    python -m vicap.model.register_p1 --checkpoint /p1/final_model.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from loguru import logger

from vicap.config import MlflowConfig
from vicap.utils.log import setup_logging

HERE = Path(__file__).resolve().parent
EXPERIMENT = "kltn-p1"
SHA_TAG = "checkpoint_sha256"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_params(ck: dict) -> dict[str, object]:
    a = ck["arch"]
    return {
        "name": ck.get("name"), "arch": a["type"], "dims": a.get("dims"), "d": a.get("d"),
        "hidden": a.get("hidden"), "fuse": a.get("fuse"), "proj_layers": a.get("proj_layers"),
        "aux": a.get("aux"), "norm": ck.get("norm"), "threshold": ck["threshold"],
        "n_labels": len(ck["labels"]), "n_train": ck.get("n_train"),
    }


def register(checkpoint: Path, config: MlflowConfig | None = None, *, alias: str | None = None) -> tuple[str, str]:
    """Trả ``(version, run_id)`` của phiên bản đang mang alias."""
    import mlflow
    import torch
    from mlflow import MlflowClient

    config = config or MlflowConfig()
    alias = alias or config.model_alias
    mlflow.set_tracking_uri(config.tracking_uri)
    client = MlflowClient()
    digest = sha256(checkpoint)

    for mv in client.search_model_versions(f"name='{config.model_name}'"):
        if mv.tags.get(SHA_TAG) == digest:
            client.set_registered_model_alias(config.model_name, alias, mv.version)
            logger.info("{} v{} đã có checkpoint này (run {}), gán @{}", config.model_name, mv.version,
                        mv.run_id, alias)
            return mv.version, mv.run_id

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="p1-final-v12") as run:
        mlflow.log_params({k: json.dumps(v) if isinstance(v, list) else v for k, v in checkpoint_params(ck).items()})
        mlflow.log_dict({"labels": ck["labels"], "inputs": ck.get("inputs"),
                         "profile_exclude": ck.get("profile_exclude")}, "p1_meta.json")
        mlflow.set_tags({SHA_TAG: digest, "source": "kaggle cngngquc/mainnotebook (v12) final_model.pt"})
        info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=str(HERE / "mlflow_p1.py"),
            artifacts={"checkpoint": str(checkpoint)},
            code_paths=[str(HERE / "kltn_model.py")],
            pip_requirements=[f"torch=={torch.__version__.split('+')[0]}", "numpy", f"mlflow=={mlflow.__version__}"],
            registered_model_name=config.model_name,
        )
    version = str(info.registered_model_version)
    client.set_model_version_tag(config.model_name, version, SHA_TAG, digest)
    client.set_registered_model_alias(config.model_name, alias, version)
    logger.info("đăng ký {} v{} (run {}), gán @{}", config.model_name, version, run.info.run_id, alias)
    return version, run.info.run_id


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=Path("models/final_model.pt"))
    ap.add_argument("--alias", default=None, help="mặc định champion")
    args = ap.parse_args(argv)

    setup_logging()
    version, run_id = register(args.checkpoint, alias=args.alias)
    print(f"{MlflowConfig().model_name} v{version} run {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
