"""Dự đoán P1 (PLAN mục 4.1, 4.3).

``gold.features`` lưu ``text_emb`` (n, 2, 768) và ``image_emb`` (n, 8, 768); P1 nhận ``(n, 1536)`` và ``(n, 768)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vicap.config import MlflowConfig

TEXT_SLOTS, IMAGE_SLOTS, DIM = 2, 8, 768


def build_p1_inputs(text_emb: np.ndarray, image_emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``text_emb`` (n, 2, 768), ``image_emb`` (n, 8, 768) → ``(n, 1536)``, ``(n, 768)`` float32.

    - Text: ``[caption+hashtag ; ASR]`` đúng thứ tự (``T_capasr``). Đảo thứ tự thì model vẫn chạy nhưng sai.
    - Hình: trung bình các ảnh **có thật** (vector khác 0), chia cho số ảnh có chứ không chia cố định cho 8,
      giống ``siglip_encode`` của notebook (``V_frames``). Không có ảnh nào → vector 0.
    """
    text_emb = np.asarray(text_emb, dtype=np.float32)
    image_emb = np.asarray(image_emb, dtype=np.float32)
    n = len(text_emb)
    if text_emb.shape != (n, TEXT_SLOTS, DIM):
        raise ValueError(f"text_emb phải là (n, {TEXT_SLOTS}, {DIM}), nhận {text_emb.shape}")
    if image_emb.shape != (n, IMAGE_SLOTS, DIM):
        raise ValueError(f"image_emb phải là ({n}, {IMAGE_SLOTS}, {DIM}), nhận {image_emb.shape}")

    xt = text_emb.reshape(n, TEXT_SLOTS * DIM)
    present = np.abs(image_emb).sum(-1) > 0
    cnt = present.sum(1, keepdims=True)
    xv = (image_emb * present[..., None]).sum(1) / np.clip(cnt, 1, None)
    return xt, xv.astype(np.float32)


@dataclass(frozen=True)
class P1:
    """Model P1 đã nạp + phiên bản trong MLflow (để ghi kèm mỗi dự đoán, PLAN mục 1 "truy vết được")."""

    predictor: Any  # kltn_model.Predictor: predict([xt, xv]) → (n, 20), decide(P) → list[list[str]]
    name: str
    version: str
    run_id: str

    @property
    def labels(self) -> list[str]:
        return list(self.predictor.labels)

    @property
    def threshold(self) -> float:
        return float(self.predictor.threshold)


def load_p1(config: MlflowConfig | None = None) -> P1:
    """Nạp ``models:/kltn-p1@champion``: đọc alias ra phiên bản trước, rồi nạp đúng phiên bản đó."""
    import mlflow
    from mlflow import MlflowClient

    config = config or MlflowConfig()
    mlflow.set_tracking_uri(config.tracking_uri)
    mv = MlflowClient().get_model_version_by_alias(config.model_name, config.model_alias)
    model = mlflow.pyfunc.load_model(f"models:/{config.model_name}/{mv.version}")
    return P1(model.unwrap_python_model().predictor, config.model_name, str(mv.version), mv.run_id)


def predict_batch(p1: P1, text_emb: np.ndarray, image_emb: np.ndarray) -> tuple[np.ndarray, list[list[str]]]:
    """Xác suất ``(n, 20)`` và nhãn theo ``decide`` của checkpoint: ``P >= threshold``, luôn có ít nhất nhãn có
    xác suất cao nhất (``at_least_one=True``, PLAN mục 4.3)."""
    if len(text_emb) == 0:
        return np.zeros((0, len(p1.labels)), dtype=np.float32), []
    xt, xv = build_p1_inputs(text_emb, image_emb)
    probs = np.asarray(p1.predictor.predict([xt, xv]), dtype=np.float32)
    return probs, p1.predictor.decide(probs)
