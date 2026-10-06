"""Learned mask, edge and positional-encoding components."""

from __future__ import annotations

from typing import Any, Optional, Tuple
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: int) -> torch.Tensor:
        h, w = size, size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)


class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.gate = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        feat = self.conv(x)
        gate = torch.sigmoid(self.gate(x))
        return feat * gate


class EdgeCompletionNet(nn.Module):
    def __init__(self, in_channels, base_channels=32):
        super().__init__()
        self.block1 = GatedConv2d(in_channels, base_channels, 3, 1, 1)
        self.block2 = GatedConv2d(base_channels, base_channels, 3, 1, 1)
        self.block3 = nn.Conv2d(base_channels, 1, 3, 1, 1)

    def forward(self, x):
        x = F.relu(self.block1(x))
        x = F.relu(self.block2(x))
        return self.block3(x)


class StructureFiLM(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, embed_dim * 2, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, prior: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma_beta = self.net(prior)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        return gamma, beta


class BoundaryRefineHead(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 32):
        super().__init__()
        self.block1 = GatedConv2d(in_channels, hidden, 3, 1, 1)
        self.block2 = nn.Conv2d(hidden, 1, 3, 1, 1)
        nn.init.zeros_(self.block2.weight)
        nn.init.zeros_(self.block2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.block1(x))
        return self.block2(x)
