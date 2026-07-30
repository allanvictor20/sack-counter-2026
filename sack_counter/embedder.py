"""
embedder.py — Deep appearance feature extractor for Sack Counter v13.

Uses MobileNetV3-Small to produce a 576-dim L2-normalised embedding
for any image crop.  Falls back to a zero vector for crops that are
too small to embed reliably.
"""

import cv2
import numpy as np
import torch
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image


class DeepEmbedder:
    """
    Lightweight MobileNetV3-Small feature extractor.
    Returns a 576-dim L2-normalised embedding for any image crop.
    Falls back to zero vector if the crop is too small.
    """

    _transform = T.Compose([
        T.Resize((128, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self):
        backbone = tv_models.mobilenet_v3_small(
            weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT)
        self.net = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.net.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self._dim = 576

    @torch.no_grad()
    def embed(self, frame: np.ndarray, x1, y1, x2, y2) -> np.ndarray:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return np.zeros(self._dim, dtype=np.float32)
        crop = frame[y1:y2, x1:x2]
        img  = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        t    = self._transform(img).unsqueeze(0).to(self.device)
        feat = self.net(t).squeeze().cpu().numpy()
        norm = np.linalg.norm(feat)
        return feat / (norm + 1e-6)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return 0.5
        return float(np.dot(a, b))
