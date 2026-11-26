"""SigLIP2 cho ``image_emb`` (PLAN mục 4.1), port ``siglip_encode`` / ``extract_pooled`` của notebook (cell 19) đã
sinh cache ``siglip_frames8_seq__v2a.npz`` (``per_frame=True``) và ``siglip_frames8__v2a.npz`` (trung bình).

- ``google/siglip2-base-patch16-224`` fp32, ``eval()``; ``AutoImageProcessor`` trên ảnh ``convert("RGB")``.
- Vector = ``get_image_features``: tensor thì lấy luôn, có ``pooler_output`` thì lấy nó, không thì trung bình
  ``last_hidden_state``. Chưa chuẩn hoá L2.
- ``encode``: mỗi video một list 8 ảnh (thứ tự ``_0 … _7``) → ``(n, 8, 768)``; ảnh thiếu (``None``) → vector 0.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

IMAGE_MODEL = "google/siglip2-base-patch16-224"
BATCH_SIZE = 32


def _pooled(out: Any):
    import torch

    if torch.is_tensor(out):
        return out
    if getattr(out, "pooler_output", None) is not None:
        return out.pooler_output
    return out.last_hidden_state.mean(dim=1)


class SigLip:
    def __init__(self, model: str = IMAGE_MODEL, device: str | None = None):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoImageProcessor.from_pretrained(model)
        self.model: Any = AutoModel.from_pretrained(model).to(self.device).eval()
        self.name = model

    def encode(self, images: Sequence[Sequence[Any]], batch_size: int = BATCH_SIZE) -> np.ndarray:
        """``images[i][k]``: đường dẫn / file-like / ``PIL.Image`` của ảnh ``k`` video ``i`` (``None`` = thiếu)."""
        from PIL import Image

        n, k_max = len(images), max((len(x) for x in images), default=0)
        flat = [(i, k, im) for i, ims in enumerate(images) for k, im in enumerate(ims) if im is not None]
        X: np.ndarray | None = None
        for j in range(0, len(flat), batch_size):
            chunk = flat[j:j + batch_size]
            imgs = [(im if isinstance(im, Image.Image) else Image.open(im)).convert("RGB") for *_, im in chunk]
            inputs = self.proc(images=imgs, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                f = _pooled(self.model.get_image_features(**inputs)).float().cpu().numpy()
            if X is None:
                X = np.zeros((n, k_max, f.shape[1]), dtype=np.float32)
            for (i, k, _), v in zip(chunk, f, strict=True):
                X[i, k] = v
        if X is None:
            dim = int(getattr(self.model.config.vision_config, "hidden_size", 768))
            X = np.zeros((n, k_max, dim), dtype=np.float32)
        return X
