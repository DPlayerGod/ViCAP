"""PhoBERT cho ``text_emb`` (PLAN mục 4.1), port nguyên ``word_segment`` / ``encode_texts`` / ``encode_chunked``
của notebook (cell 16) đã sinh cache ``text_caphash_*.npz`` và ``text_asr_chunked_PhoWhisper_medium_*.npz``.

- Tách từ bằng pyvi (``ViTokenizer.tokenize``); text rỗng → ``""``; pyvi lỗi thì giữ nguyên text.
- ``vinai/phobert-base`` fp32, ``eval()``; vector = mean-pool ``last_hidden_state`` theo ``attention_mask``.
- ``encode_texts`` (``text_emb[0]``, caption + hashtag): ``truncation=True, max_length=256``, batch 32.
- ``encode_chunked`` (``text_emb[1]``, ``transcript_raw``): token không special, chia đoạn 254 token, thêm
  ``<s>`` / ``</s>``, mean-pool từng đoạn rồi trung bình các đoạn; transcript rỗng → vector 0.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

TEXT_MODEL = "vinai/phobert-base"
MAX_LENGTH = 256
BATCH_SIZE = 32


def word_segment(text: str | None) -> str:
    if not text or not text.strip():
        return ""
    from pyvi import ViTokenizer

    try:
        return ViTokenizer.tokenize(text)
    except Exception:  # noqa: BLE001 — như notebook: pyvi lỗi thì giữ nguyên
        return text


class PhoBert:
    def __init__(self, model: str = TEXT_MODEL, device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model: Any = AutoModel.from_pretrained(model).to(self.device).eval()
        self.name = model

    def _pool(self, input_ids, attention_mask) -> np.ndarray:
        with self.torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            m = attention_mask.unsqueeze(-1).float()
            return ((out * m).sum(1) / m.sum(1).clamp(min=1e-9)).cpu().float().numpy()

    def encode_texts(self, texts: Sequence[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        feats = []
        for i in range(0, len(texts), batch_size):
            batch = [word_segment(t) for t in texts[i:i + batch_size]]
            inputs = self.tok(batch, padding=True, truncation=True, max_length=MAX_LENGTH,
                              return_tensors="pt").to(self.device)
            feats.append(self._pool(inputs["input_ids"], inputs["attention_mask"]))
        if not feats:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        return np.concatenate(feats, axis=0)

    def encode_chunked(self, texts: Sequence[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        torch, tok = self.torch, self.tok
        chunks: list[list[int]] = []
        owner: list[int] = []
        for i, t in enumerate(texts):
            if not (t or "").strip():
                continue
            ids = tok(word_segment(t), add_special_tokens=False, truncation=False)["input_ids"]
            for s in range(0, len(ids), MAX_LENGTH - 2):
                chunks.append([tok.cls_token_id, *ids[s:s + MAX_LENGTH - 2], tok.sep_token_id])
                owner.append(i)
        res = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)
        cnt = np.zeros(len(texts))
        for j in range(0, len(chunks), batch_size):
            b = chunks[j:j + batch_size]
            width = max(len(x) for x in b)
            input_ids = torch.full((len(b), width), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(b), width), dtype=torch.long)
            for k, x in enumerate(b):
                input_ids[k, :len(x)] = torch.tensor(x)
                att[k, :len(x)] = 1
            pooled = self._pool(input_ids.to(self.device), att.to(self.device))
            for o, vec in zip(owner[j:j + batch_size], pooled, strict=True):
                res[o] += vec
                cnt[o] += 1
        res[cnt > 0] /= cnt[cnt > 0, None]
        return res
