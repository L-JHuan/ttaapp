"""GNPR 残差码本使用的 POI 连续向量数据集。"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, data_path: Path | str):
        with Path(data_path).open("rb") as handle:
            raw: dict[Any, Any] = pickle.load(handle)
        if not raw:
            raise ValueError(f"POI embedding 为空: {data_path}")

        self.ids = sorted(int(key) for key in raw)
        self.embeddings = np.asarray(
            [raw[key] if key in raw else raw[str(key)] for key in self.ids],
            dtype=np.float32,
        )
        if self.embeddings.ndim != 2:
            raise ValueError(f"POI embedding 必须是二维矩阵: {self.embeddings.shape}")
        if not np.isfinite(self.embeddings).all():
            raise ValueError("POI embedding 包含 NaN 或 Inf")
        self.dim = int(self.embeddings.shape[1])

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        return self.ids[index], torch.from_numpy(self.embeddings[index])

    def __len__(self) -> int:
        return len(self.ids)
