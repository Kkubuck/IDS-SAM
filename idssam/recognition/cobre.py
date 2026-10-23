"""Counterfactual Background Residual correction."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def top12_margin_torch(logits: torch.Tensor) -> torch.Tensor:
    k = min(2, int(logits.shape[1]))
    vals = torch.topk(logits, k=k, dim=1).values
    if vals.shape[1] == 1:
        return torch.zeros((logits.shape[0], 1), device=logits.device, dtype=logits.dtype)
    return (vals[:, 0] - vals[:, 1]).unsqueeze(1)


class CoBRe(nn.Module):
    """Counterfactual Background Residual Adapter used as a stage-1 correction module."""

    def __init__(
        self,
        dim: int = 1536,
        n_proto: int = 16,
        gate_hidden: int = 128,
        hard_thr_delta: float = 0.25,
        hard_thr_agree: float = 0.75,
        hard_temp_delta: float = 0.20,
        hard_temp_agree: float = 0.20,
        use_anchor_trust: bool = False,
        trust_thr_m: float = 0.10,
        trust_temp_m: float = 0.05,
    ) -> None:
        super().__init__()
        self.use_anchor_trust = bool(use_anchor_trust)
        self.proto_bg = nn.Parameter(torch.randn(n_proto, dim) / (dim**0.5))
        self.log_tau = nn.Parameter(torch.tensor(np.log(0.07), dtype=torch.float32))
        self.gate = nn.Sequential(
            nn.Linear(6, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 3),
        )
        self.thr_delta = nn.Parameter(torch.tensor(float(hard_thr_delta), dtype=torch.float32))
        self.thr_agree = nn.Parameter(torch.tensor(float(hard_thr_agree), dtype=torch.float32))
        self.log_temp_delta = nn.Parameter(
            torch.tensor(np.log(max(float(hard_temp_delta), 1e-3)), dtype=torch.float32)
        )
        self.log_temp_agree = nn.Parameter(
            torch.tensor(np.log(max(float(hard_temp_agree), 1e-3)), dtype=torch.float32)
        )
        self.thr_m = nn.Parameter(torch.tensor(float(trust_thr_m), dtype=torch.float32))
        self.log_temp_m = nn.Parameter(
            torch.tensor(np.log(max(float(trust_temp_m), 1e-3)), dtype=torch.float32)
        )
        with torch.no_grad():
            self.gate[-1].bias[:] = torch.tensor([-2.0, -2.0, -2.2])

    def forward(
        self,
        base_emb: torch.Tensor,
        global_emb: torch.Tensor,
        local_emb: torch.Tensor,
        base_margin: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor:
        base = F.normalize(base_emb, p=2, dim=-1)
        g = F.normalize(global_emb, p=2, dim=-1)
        l = F.normalize(local_emb, p=2, dim=-1)

        delta_bg = g - l
        delta_norm = delta_bg.norm(dim=-1, keepdim=True)
        d_bg = F.normalize(delta_bg, p=2, dim=-1)

        proj = (l * g).sum(dim=-1, keepdim=True)
        fg_res = F.normalize(l - proj * g, p=2, dim=-1)

        P = F.normalize(self.proto_bg, p=2, dim=-1)
        tau = torch.exp(self.log_tau).clamp(1e-3, 1.0)
        attn = torch.matmul(d_bg, P.t()) / tau
        w = F.softmax(attn, dim=-1)
        bg_hat = torch.matmul(w, P)

        agree = (g * l).sum(dim=-1, keepdim=True)
        bg_align_g = (bg_hat * g).sum(dim=-1, keepdim=True)
        bg_align_l = (bg_hat * l).sum(dim=-1, keepdim=True)
        d_align_g = (d_bg * g).sum(dim=-1, keepdim=True)
        d_align_l = (d_bg * l).sum(dim=-1, keepdim=True)
        feat = torch.cat([agree, delta_norm, bg_align_g, bg_align_l, d_align_g, d_align_l], dim=-1)

        raw = self.gate(feat)
        lambda_bg = F.softplus(raw[:, 0:1]).clamp(0.0, 1.5)
        lambda_fg = F.softplus(raw[:, 1:2]).clamp(0.0, 1.0)

        g_corr = F.normalize(g - lambda_bg * bg_hat + lambda_fg * fg_res, p=2, dim=-1)
        temp_delta = torch.exp(self.log_temp_delta).clamp(1e-3, 2.0)
        temp_agree = torch.exp(self.log_temp_agree).clamp(1e-3, 2.0)
        hard = torch.sigmoid((delta_norm - self.thr_delta) / temp_delta) * torch.sigmoid(
            (self.thr_agree - agree) / temp_agree
        )
        mix = torch.sigmoid(raw[:, 2:3])
        mix = (mix * hard).clamp(0.0, 1.0)
        trust = torch.ones_like(mix)
        if self.use_anchor_trust and base_margin is not None:
            temp_m = torch.exp(self.log_temp_m).clamp(1e-3, 2.0)
            trust = torch.sigmoid((self.thr_m - base_margin) / temp_m)
            mix = (mix * trust).clamp(0.0, 1.0)
        anchor = base if self.use_anchor_trust else g
        out = F.normalize((1.0 - mix) * anchor + mix * g_corr, p=2, dim=-1)
        if return_aux:
            return out, {
                "hard": hard,
                "mix": mix,
                "trust": trust,
                "agree": agree,
                "delta_norm": delta_norm,
            }
        return out
