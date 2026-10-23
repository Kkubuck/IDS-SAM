"""Image-derived structural priors used by IDS-SAM."""

from __future__ import annotations

import math
from typing import Dict
import torch
from torch import nn
from torch.nn import functional as F


class StructurePriorExtractor(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.enable = cfg.get("enable", False)
        self.use_input_norm = cfg.get("use_input_norm", False)
        self.input_mean = cfg.get("input_mean", [0.485, 0.456, 0.406])
        self.input_std = cfg.get("input_std", [0.229, 0.224, 0.225])

        sobel_cfg = cfg.get("sobel", {}) if cfg is not None else {}
        self.sobel_enable = sobel_cfg.get("enable", True)
        self.sobel_sigmas = sobel_cfg.get("sigmas", [0.0, 1.0, 2.0])
        self.sobel_weights = sobel_cfg.get("weights", None)
        self.sobel_window = int(sobel_cfg.get("window", 31))
        if self.sobel_window % 2 == 0:
            self.sobel_window += 1
        self.sobel_clip = sobel_cfg.get("clip", 3.0)
        self.sobel_tanh = sobel_cfg.get("tanh", True)

        phase_cfg = cfg.get("phase", {}) if cfg is not None else {}
        self.phase_enable = phase_cfg.get("enable", False)
        self.pc_size = int(phase_cfg.get("pc_size", 256))
        self.pc_orientations = int(phase_cfg.get("orientations", 4))
        self.pc_min_wavelength = float(phase_cfg.get("min_wavelength", 3.0))
        self.pc_mult = float(phase_cfg.get("mult", 2.1))
        self.pc_sigma_on_f = float(phase_cfg.get("sigma_on_f", 0.55))
        self.pc_scales = phase_cfg.get("scales", [0, 1, 2])
        if isinstance(self.pc_scales, int):
            self.pc_scales = list(range(self.pc_scales))
        self.pc_eps = float(phase_cfg.get("eps", 1e-6))

        self.register_buffer(
            "sobel_kernel_x",
            torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(
                1, 1, 3, 3
            ),
        )
        self.register_buffer(
            "sobel_kernel_y",
            torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(
                1, 1, 3, 3
            ),
        )
        self._gauss_cache = {}
        self._pc_cache = {}

    def _get_gaussian_kernel(self, sigma: float, device, dtype):
        if sigma <= 0:
            return None
        key = (sigma, device, dtype)
        if key in self._gauss_cache:
            return self._gauss_cache[key]
        size = max(3, int(round(sigma * 6.0)) + 1)
        if size % 2 == 0:
            size += 1
        coords = torch.arange(size, device=device, dtype=dtype) - (size // 2)
        g = torch.exp(-(coords**2) / (2 * sigma * sigma))
        g = g / (g.sum() + 1e-12)
        kernel = (g[:, None] * g[None, :]).view(1, 1, size, size)
        self._gauss_cache[key] = kernel
        return kernel

    def _to_luminance(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_input_norm:
            mean = torch.tensor(self.input_mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            std = torch.tensor(self.input_std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            x = x * std + mean
            x = torch.clamp(x, 0.0, 1.0)
        y = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        return y

    def _multi_scale_sobel(self, y: torch.Tensor) -> torch.Tensor:
        grads = []
        weights = self.sobel_weights
        if weights is None:
            weights = [1.0 / max(len(self.sobel_sigmas), 1)] * len(self.sobel_sigmas)
        weight_sum = sum(weights) if weights else 1.0
        weights = [w / weight_sum for w in weights]

        for sigma, weight in zip(self.sobel_sigmas, weights):
            if sigma > 0:
                kernel = self._get_gaussian_kernel(sigma, y.device, y.dtype)
                y_blur = F.conv2d(y, kernel, padding=kernel.shape[-1] // 2)
            else:
                y_blur = y
            gx = F.conv2d(y_blur, self.sobel_kernel_x.to(device=y.device, dtype=y.dtype), padding=1)
            gy = F.conv2d(y_blur, self.sobel_kernel_y.to(device=y.device, dtype=y.dtype), padding=1)
            grad = torch.sqrt(gx * gx + gy * gy + 1e-6)
            grads.append(weight * grad)
        return sum(grads)

    def _local_zscore(self, x: torch.Tensor) -> torch.Tensor:
        if self.sobel_window <= 1:
            return x
        mu = F.avg_pool2d(
            x, kernel_size=self.sobel_window, stride=1, padding=self.sobel_window // 2
        )
        mu2 = F.avg_pool2d(
            x * x, kernel_size=self.sobel_window, stride=1, padding=self.sobel_window // 2
        )
        var = torch.clamp(mu2 - mu * mu, min=0.0)
        std = torch.sqrt(var + 1e-6)
        z = (x - mu) / (std + 1e-6)
        if self.sobel_clip is not None:
            z = torch.clamp(z, -float(self.sobel_clip), float(self.sobel_clip))
        if self.sobel_tanh:
            z = torch.tanh(z)
        return z

    def _build_pc_filters(self, size: int, device, dtype):
        cache_key = (size, device, dtype)
        if cache_key in self._pc_cache:
            return self._pc_cache[cache_key]

        y, x = torch.meshgrid(
            torch.linspace(-0.5, 0.5, steps=size, device=device, dtype=dtype),
            torch.linspace(-0.5, 0.5, steps=size, device=device, dtype=dtype),
            indexing="ij",
        )
        radius = torch.sqrt(x * x + y * y)
        radius[0, 0] = 1.0
        theta = torch.atan2(-y, x)

        log_gabors = []
        for s in range(self.pc_orientations):
            log_gabors.append([])
        for scale in self.pc_scales:
            wavelength = self.pc_min_wavelength * (self.pc_mult ** float(scale))
            fo = 1.0 / wavelength
            log_gabor = torch.exp(
                -(torch.log(radius / fo) ** 2) / (2 * (math.log(self.pc_sigma_on_f) ** 2))
            )
            log_gabor[radius < 1e-6] = 0.0
            for o in range(self.pc_orientations):
                theta0 = o * math.pi / self.pc_orientations
                dtheta = torch.abs(theta - theta0)
                dtheta = torch.minimum(dtheta, math.pi - dtheta)
                theta_sigma = math.pi / self.pc_orientations / 1.5
                spread = torch.exp(-(dtheta**2) / (2 * theta_sigma * theta_sigma))
                log_gabors[o].append(log_gabor * spread)

        self._pc_cache[cache_key] = log_gabors
        return log_gabors

    def _phase_congruency(self, y: torch.Tensor) -> torch.Tensor:
        size = min(self.pc_size, y.shape[-1])
        y_small = F.interpolate(y, size=(size, size), mode="bilinear", align_corners=False)
        f = torch.fft.fft2(y_small.squeeze(1))

        filters = self._build_pc_filters(size, y.device, y.dtype)
        pcs = []
        for o in range(self.pc_orientations):
            sum_even = torch.zeros_like(f.real)
            sum_odd = torch.zeros_like(f.real)
            sum_an = torch.zeros_like(f.real)
            for filt in filters[o]:
                eo = torch.fft.ifft2(f * filt)
                even = eo.real
                odd = eo.imag
                amp = torch.sqrt(even * even + odd * odd + self.pc_eps)
                sum_even = sum_even + even
                sum_odd = sum_odd + odd
                sum_an = sum_an + amp
            energy = torch.sqrt(sum_even * sum_even + sum_odd * sum_odd + self.pc_eps)
            pc = energy / (sum_an + self.pc_eps)
            pcs.append(pc)
        pc_map = torch.stack(pcs, dim=1).max(dim=1).values
        pc_map = pc_map.clamp(0.0, 1.0).unsqueeze(1)
        pc_map = F.interpolate(pc_map, size=y.shape[-2:], mode="bilinear", align_corners=False)
        return pc_map

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enable:
            return None
        y = self._to_luminance(x)
        if self.sobel_enable:
            grad = self._multi_scale_sobel(y)
            grad = self._local_zscore(grad)
        else:
            grad = torch.zeros_like(y)

        priors = [grad]
        if self.phase_enable:
            pc = self._phase_congruency(y)
            priors.append(pc)
        return torch.cat(priors, dim=1)


class StructurePriorBank(StructurePriorExtractor):
    """
    Dynamic Top-2 structure selector.
    Candidates: grad / phase / hf / boundary
    Aux maps for scoring: support / far / contam
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.mode = str(cfg.get("mode", "dynamic_top2")).lower()
        self.candidate_maps = list(cfg.get("candidate_maps", ["grad", "phase", "hf", "boundary"]))
        if len(self.candidate_maps) == 0:
            self.candidate_maps = ["grad", "phase", "hf", "boundary"]
        self.topk_struct = int(cfg.get("topk_struct", 2))
        self.topk_struct = max(1, min(self.topk_struct, len(self.candidate_maps)))
        self.score_use_support_far_contam = bool(cfg.get("score_use_support_far_contam", True))
        self.score_temperature = float(cfg.get("score_temperature", 1.0))
        self.score_temperature = max(1e-6, self.score_temperature)
        score_hidden = int(cfg.get("score_head_hidden", 0))
        score_in = len(self.candidate_maps) + (3 if self.score_use_support_far_contam else 0)
        score_out = len(self.candidate_maps)
        if score_hidden > 0:
            self.score_head = nn.Sequential(
                nn.Conv2d(score_in, score_hidden, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(score_hidden, score_out, kernel_size=1),
            )
        else:
            self.score_head = nn.Conv2d(score_in, score_out, kernel_size=1)

        self.register_buffer(
            "laplacian_kernel",
            torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3),
        )
        self.last_aux: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _robust_norm(
        x: torch.Tensor, q_low: float = 0.05, q_high: float = 0.95, eps: float = 1e-6
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        flat = x.view(b, c, -1)
        lo = torch.quantile(flat, q_low, dim=-1, keepdim=True)
        hi = torch.quantile(flat, q_high, dim=-1, keepdim=True)
        x_n = (flat - lo) / (hi - lo + eps)
        x_n = x_n.clamp(0.0, 1.0).view(b, c, h, w)
        return torch.nan_to_num(x_n, nan=0.0, posinf=1.0, neginf=0.0)

    def _hf_map(self, y: torch.Tensor) -> torch.Tensor:
        lap = F.conv2d(y, self.laplacian_kernel.to(device=y.device, dtype=y.dtype), padding=1)
        hf = torch.abs(lap)
        return self._robust_norm(hf)

    def _boundary_map(self, grad: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(grad, self.sobel_kernel_x.to(device=grad.device, dtype=grad.dtype), padding=1)
        gy = F.conv2d(grad, self.sobel_kernel_y.to(device=grad.device, dtype=grad.dtype), padding=1)
        bmap = torch.sqrt(gx * gx + gy * gy + 1e-6)
        return self._robust_norm(bmap)

    @staticmethod
    def _far_map(support: torch.Tensor, near_ks: int = 17, far_ks: int = 65) -> torch.Tensor:
        near_ks = int(max(3, near_ks))
        far_ks = int(max(near_ks + 2, far_ks))
        if near_ks % 2 == 0:
            near_ks += 1
        if far_ks % 2 == 0:
            far_ks += 1
        support_bin = (support > 0.5).float()
        near = F.max_pool2d(support_bin, kernel_size=near_ks, stride=1, padding=near_ks // 2)
        far = 1.0 - F.max_pool2d(support_bin, kernel_size=far_ks, stride=1, padding=far_ks // 2)
        return torch.clamp((1.0 - near) * far, 0.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enable:
            return None

        y = self._to_luminance(x)

        if self.sobel_enable:
            grad = self._multi_scale_sobel(y)
            grad = self._local_zscore(grad)
            grad = self._robust_norm(grad)
        else:
            grad = torch.zeros_like(y)

        if self.phase_enable:
            phase = self._phase_congruency(y)
            phase = self._robust_norm(phase)
        else:
            phase = torch.zeros_like(y)

        hf = self._hf_map(y)
        boundary = self._boundary_map(grad)

        support = torch.clamp(0.55 * grad + 0.30 * boundary + 0.15 * (1.0 - hf), 0.0, 1.0)
        far = self._far_map(support)
        contam = torch.clamp(far * (1.0 - support) * hf, 0.0, 1.0)

        named = {
            "grad": grad,
            "phase": phase,
            "hf": hf,
            "boundary": boundary,
        }
        cand_list = []
        valid_names = []
        for name in self.candidate_maps:
            if name in named:
                cand_list.append(named[name])
                valid_names.append(name)
        if len(cand_list) == 0:
            cand_list = [grad, phase, hf, boundary]
            valid_names = ["grad", "phase", "hf", "boundary"]
        cand = torch.cat(cand_list, dim=1)  # [B, N, H, W]
        n_cand = cand.shape[1]
        k = max(1, min(self.topk_struct, n_cand))

        if self.mode in {"static_grad_phase", "static_top2", "static_topk", "static"}:
            if self.mode == "static_grad_phase":
                m1 = grad
                m2 = phase if self.phase_enable else boundary
                prior2 = torch.cat([m1, m2], dim=1)[:, :k, ...]
            else:
                # Use globally fixed candidate order from config (e.g., [grad, hf]).
                prior2 = cand[:, :k, ...]
            self.last_aux = {
                "score": torch.zeros(x.shape[0], n_cand, device=x.device, dtype=x.dtype),
                "top2_idx": torch.zeros(x.shape[0], k, device=x.device, dtype=torch.long),
                "top2_weight": torch.full(
                    (x.shape[0], k), 1.0 / float(k), device=x.device, dtype=x.dtype
                ),
            }
            return torch.nan_to_num(prior2, nan=0.0, posinf=0.0, neginf=0.0)

        score_inputs = [cand]
        if self.score_use_support_far_contam:
            score_inputs.extend([support, far, contam])
        score_in = torch.cat(score_inputs, dim=1)
        score_logits = self.score_head(score_in) / self.score_temperature
        score_logits = torch.nan_to_num(score_logits, nan=0.0, posinf=0.0, neginf=0.0)
        score_vec = F.adaptive_avg_pool2d(score_logits, output_size=1).flatten(1)
        score_vec = torch.clamp(score_vec, -12.0, 12.0)

        vals, idx = torch.topk(score_vec, k=k, dim=1, largest=True, sorted=True)
        w = torch.softmax(vals, dim=1)
        gather_idx = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, cand.size(2), cand.size(3))
        top_maps = torch.gather(cand, dim=1, index=gather_idx)
        prior2 = top_maps * w.unsqueeze(-1).unsqueeze(-1)
        prior2 = torch.nan_to_num(prior2, nan=0.0, posinf=0.0, neginf=0.0)

        self.last_aux = {
            "score": score_vec.detach(),
            "top2_idx": idx.detach(),
            "top2_weight": w.detach(),
            "support": support.detach(),
            "far": far.detach(),
            "contam": contam.detach(),
        }
        return prior2
