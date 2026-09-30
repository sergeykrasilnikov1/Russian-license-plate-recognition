"""CRNN (CNN + BiLSTM + CTC) plate-text recogniser.

Input: 1x32x160 grayscale crop of ONE text row (single-row plates: the whole plate; square plates:
each row separately). Output: per-frame log-probabilities over ALPHABET + blank.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .plates import ALPHABET, UNK

IN_H, IN_W = 32, 160
N_CLASSES = len(ALPHABET) + 1     # + CTC blank (last index)
BLANK = len(ALPHABET)


class CRNN(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES, hidden: int = 128):
        super().__init__()

        def block(cin, cout, pool):
            layers = [nn.Conv2d(cin, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
            if pool:
                layers.append(nn.MaxPool2d(pool))
            return layers

        self.cnn = nn.Sequential(
            *block(1, 32, (2, 2)),          # 16 x 80
            *block(32, 64, (2, 2)),         # 8 x 40
            *block(64, 128, None),
            *block(128, 128, (2, 1)),       # 4 x 40
            *block(128, 256, None),
            *block(256, 256, (2, 1)),       # 2 x 40
            nn.Conv2d(256, 256, (2, 1), 1, 0, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),   # 1 x 40
        )
        self.rnn = nn.LSTM(256, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: B,1,32,160 -> B,T,C
        f = self.cnn(x)                                        # B,256,1,T
        f = f.squeeze(2).permute(0, 2, 1)                      # B,T,256
        f, _ = self.rnn(f)
        return self.fc(f)


def preprocess(crop_bgr: np.ndarray) -> np.ndarray:
    """Resize keeping aspect (pad on the right with edge colour), normalise to [-1,1]. Returns 1xHxW float32."""
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    h, w = g.shape[:2]
    s = IN_H / max(1, h)
    nw = int(round(w * s))
    if nw > IN_W:
        g = cv2.resize(g, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
    else:
        g = cv2.resize(g, (max(1, nw), IN_H), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        pad = np.full((IN_H, IN_W - nw), int(np.median(g[:, -2:])), np.uint8)
        g = np.hstack([g, pad])
    return (g.astype(np.float32) / 127.5 - 1.0)[None]


def greedy_decode(logits: torch.Tensor) -> List[Tuple[str, List[float]]]:
    """logits: B,T,C -> list of (text, per-char confidences)."""
    probs = F.softmax(logits, dim=-1)
    conf, idx = probs.max(dim=-1)
    out = []
    for b in range(logits.shape[0]):
        text, confs = [], []
        prev = BLANK
        cur_conf = 0.0
        for t in range(logits.shape[1]):
            k = int(idx[b, t])
            c = float(conf[b, t])
            if k != BLANK and k != prev:
                text.append(ALPHABET[k])
                confs.append(c)
            elif k != BLANK and k == prev:
                confs[-1] = max(confs[-1], c)
            prev = k
        out.append(("".join(text), confs))
    return out


def apply_unknown(text: str, confs: List[float], thr: float) -> Tuple[str, float]:
    chars = [UNK if c < thr else ch for ch, c in zip(text, confs)]
    if not confs:
        return "", 0.0
    score = float(np.exp(np.mean(np.log(np.clip(confs, 1e-4, 1.0)))))
    return "".join(chars), score


class Recognizer:
    def __init__(self, weights: str, device: str = "cpu", unk_thr: float = 0.45, fp16: bool = False):
        self.device = torch.device(device)
        self.model = CRNN().to(self.device).eval()
        state = torch.load(weights, map_location="cpu")
        self.model.load_state_dict(state["model"] if "model" in state else state)
        self.unk_thr = unk_thr
        self.fp16 = fp16 and self.device.type == "cuda"
        if self.fp16:
            self.model.half()

    @torch.inference_mode()
    def read_rows(self, rows: List[np.ndarray]) -> List[Tuple[str, List[float]]]:
        if not rows:
            return []
        x = torch.from_numpy(np.stack([preprocess(r) for r in rows])).to(self.device)
        if self.fp16:
            x = x.half()
        logits = self.model(x).float()
        return greedy_decode(logits)
