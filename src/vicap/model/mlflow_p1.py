"""Mô hình P1 dạng MLflow pyfunc, log theo kiểu "models from code" (``register_p1.py``).

File này được MLflow chép nguyên vào model; lúc load, ``kltn_model.py`` (``code_paths``) nằm trên ``sys.path`` nên
``import kltn_model`` chạy được mà không cần package ``vicap`` (PLAN mục 9, lưu ý về ``kltn_model``).

``predict`` nhận ``{"text": (n, 1536), "image": (n, 768)}`` (``vicap.model.predict.build_p1_inputs``) và trả
xác suất ``(n, 20)``. ``self.predictor`` là ``kltn_model.Predictor`` (có ``labels``, ``threshold``, ``decide``).
"""

from __future__ import annotations

import mlflow
import numpy as np
from mlflow.pyfunc import PythonModel


class P1Model(PythonModel):
    def load_context(self, context) -> None:
        import torch

        try:
            import kltn_model  # code_paths của model đã đăng ký
        except ImportError:  # lúc log_model, MLflow load thử ngay trong tiến trình đang đăng ký
            from vicap.model import kltn_model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = kltn_model.load(context.artifacts["checkpoint"], device=device)

    def predict(self, context, model_input, params=None):
        xt = np.asarray(model_input["text"], dtype=np.float32)
        xv = np.asarray(model_input["image"], dtype=np.float32)
        return self.predictor.predict([xt, xv])


mlflow.models.set_model(P1Model())
