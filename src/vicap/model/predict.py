"""Dự đoán P1 (PLAN mục 4.1, 4.3).

``gold.features`` lưu ``text_emb`` (n, 2, 768) và ``image_emb`` (n, 8, 768); P1 nhận ``(n, 1536)`` và ``(n, 768)``.
"""

from __future__ import annotations

import numpy as np

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
