"""IDS-SAM localization network."""

from __future__ import annotations

import logging
from functools import partial
from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F
from .sam.encoder import ImageEncoderViT
from .sam.decoder import MaskDecoder as MaskDecoder_Edge
from .sam.attention import TwoWayTransformer as TwoWayTransformer_MaskDecoder_Edge
from .layers import PositionEmbeddingRandom, EdgeCompletionNet, StructureFiLM, BoundaryRefineHead
from .priors import StructurePriorExtractor, StructurePriorBank

logger = logging.getLogger(__name__)


class _IDSAMBase(nn.Module):
    def __init__(
        self,
        inp_size=None,
        encoder_mode=None,
        loss=None,
        inst_prompt=None,
        edge_completion=None,
        structure_prior=None,
        structure_film=None,
        attention_bias=None,
        boundary_refine=None,
        loss_extra=None,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.requires_clip = False
        self.embed_dim = encoder_mode["embed_dim"]
        encoder_name = str(encoder_mode.get("name", "sam")).lower()
        self.image_encoder = ImageEncoderViT(
            img_size=inp_size,
            patch_size=encoder_mode["patch_size"],
            in_chans=3,
            embed_dim=encoder_mode["embed_dim"],
            depth=encoder_mode["depth"],
            num_heads=encoder_mode["num_heads"],
            mlp_ratio=encoder_mode["mlp_ratio"],
            out_chans=encoder_mode["out_chans"],
            qkv_bias=encoder_mode["qkv_bias"],
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-06),
            act_layer=nn.GELU,
            use_rel_pos=encoder_mode["use_rel_pos"],
            rel_pos_zero_init=True,
            window_size=encoder_mode["window_size"],
            global_attn_indexes=encoder_mode["global_attn_indexes"],
        )
        self.prompt_embed_dim = encoder_mode["prompt_embed_dim"]
        inst_cfg = inst_prompt or {}
        self.inst_auto_last_cond_neg = bool(inst_cfg.get("auto_last_cond_neg", True))
        self.mask_decoder = MaskDecoder_Edge(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer_MaskDecoder_Edge(
                depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
                auto_last_cond_neg=self.inst_auto_last_cond_neg,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        self.loss_mode = loss
        if self.loss_mode == "bce":
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
        elif self.loss_mode not in ("bce", "iou"):
            raise ValueError("Supported mask objectives are 'bce' and 'iou'.")
        elif self.loss_mode == "iou":
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
            self.criterionIOU = None
        self.pe_layer = PositionEmbeddingRandom(encoder_mode["prompt_embed_dim"] // 2)
        self.inp_size = inp_size
        self.image_embedding_size = inp_size // encoder_mode["patch_size"]
        self.no_mask_embed = nn.Embedding(1, encoder_mode["prompt_embed_dim"])
        self.inst_slice_dim = inst_cfg.get("slice_dim", 1024)
        self.inst_tau = inst_cfg.get("tau", 1.0)
        self.inst_tau_learnable = inst_cfg.get("tau_learnable", False)
        if self.inst_tau_learnable:
            self.inst_tau = nn.Parameter(torch.tensor(float(self.inst_tau)))
        self.inst_ent_weight = inst_cfg.get("ent_weight", 0.0)
        self.inst_ent_eps = inst_cfg.get("ent_eps", 1e-06)
        self.inst_norm = inst_cfg.get("normalize", True)
        self.inst_top_k = inst_cfg.get("top_k", 0)
        if self.inst_top_k is not None:
            self.inst_top_k = int(self.inst_top_k)
        self.inst_alpha_scale = inst_cfg.get("alpha_scale", True)
        self.inst_proj_vis = nn.Linear(self.inst_slice_dim, self.prompt_embed_dim)
        self.inst_proj_text = nn.Linear(self.inst_slice_dim, self.prompt_embed_dim)
        self.inst_norm_vis = nn.LayerNorm(self.prompt_embed_dim)
        self.inst_norm_text = nn.LayerNorm(self.prompt_embed_dim)
        struct_cfg = structure_prior or {}
        film_cfg = structure_film or {}
        attn_cfg = attention_bias or {}
        refine_cfg = boundary_refine or {}
        self.struct_enable = bool(struct_cfg.get("enable", False))
        self.struct_film_enable = bool(film_cfg.get("enable", False))
        self.attn_bias_enable = bool(attn_cfg.get("enable", False))
        self.boundary_refine_enable = bool(refine_cfg.get("enable", False))
        self.struct_enable = (
            self.struct_enable
            or self.struct_film_enable
            or self.attn_bias_enable
            or self.boundary_refine_enable
        )
        if self.struct_enable and (not struct_cfg.get("enable", False)):
            struct_cfg = dict(struct_cfg)
            struct_cfg["enable"] = True
        self.struct_prior = StructurePriorExtractor(struct_cfg) if self.struct_enable else None
        self.struct_prior_channels = 1 + (
            1 if struct_cfg.get("phase", {}).get("enable", False) else 0
        )
        self.struct_film = None
        self.struct_film_gamma_scale = float(film_cfg.get("gamma_scale", 1.0))
        self.struct_film_beta_scale = float(film_cfg.get("beta_scale", 1.0))
        if self.struct_film_enable:
            hidden = int(film_cfg.get("hidden", 64))
            self.struct_film = StructureFiLM(
                self.struct_prior_channels, self.prompt_embed_dim, hidden
            )
        self.attn_bias_conv = None
        self.attn_bias_scale = float(attn_cfg.get("scale", 0.1))
        self.attn_bias_clamp = float(attn_cfg.get("clamp", 2.0))
        self.attn_bias_use_inst = bool(attn_cfg.get("use_inst_weight", True))
        self.attn_bias_warmup_steps = int(attn_cfg.get("warmup_steps", 0))
        self.attn_bias_warmup_start = float(attn_cfg.get("warmup_start", 1.0))
        self.attn_bias_lambda_mode = str(attn_cfg.get("lambda_mode", "guided")).lower()
        self.attn_bias_lambda_const = float(attn_cfg.get("lambda_const", 0.5))
        if self.attn_bias_lambda_mode not in {
            "guided",
            "constant",
            "const",
            "fixed",
            "always_on",
            "always",
            "on",
        }:
            logger.warning(
                "unsupported attention_bias.lambda_mode=%s, fallback to guided",
                self.attn_bias_lambda_mode,
            )
            self.attn_bias_lambda_mode = "guided"
        for layer in getattr(self.mask_decoder.transformer, "layers", []):
            layer.router_lambda_mode = self.attn_bias_lambda_mode
            layer.router_lambda_const = self.attn_bias_lambda_const
        if self.attn_bias_enable:
            self.attn_bias_conv = nn.Conv2d(self.struct_prior_channels, 1, kernel_size=1)
            nn.init.zeros_(self.attn_bias_conv.weight)
            nn.init.zeros_(self.attn_bias_conv.bias)
        self.boundary_refine = None
        self.boundary_eta = float(refine_cfg.get("eta", 0.5))
        self.boundary_use_edge_embed = bool(refine_cfg.get("use_edge_embedding", True))
        self.boundary_warmup_steps = int(refine_cfg.get("warmup_steps", 0))
        self.boundary_warmup_start = float(refine_cfg.get("warmup_start", 1.0))
        if self.boundary_refine_enable:
            hidden = int(refine_cfg.get("hidden", 32))
            edge_ch = self.prompt_embed_dim // 8 if self.boundary_use_edge_embed else 0
            prior_ch = self.struct_prior_channels
            in_ch = edge_ch + prior_ch + 1
            self.boundary_refine = BoundaryRefineHead(in_ch, hidden)
        edge_cfg = edge_completion or {}
        self.edge_comp_enable = edge_cfg.get("enable", False)
        self.edge_comp_threshold = edge_cfg.get("threshold", 0.5)
        self.edge_comp_drop_prob = edge_cfg.get("drop_prob", 0.3)
        self.edge_comp_loss_weight = edge_cfg.get("loss_weight", 0.1)
        self.edge_comp_bce_weight = edge_cfg.get("bce_weight", 1.0)
        self.edge_comp_topo_weight = edge_cfg.get("topo_weight", 0.0)
        self.edge_comp_topo_iter = edge_cfg.get("topo_iter", 10)
        if self.edge_comp_enable:
            edge_in_ch = 2 + self.prompt_embed_dim // 8
            base_ch = edge_cfg.get("base_channels", 32)
            self.edge_completion = EdgeCompletionNet(edge_in_ch, base_ch)
        self._alpha_cache = None
        self._global_step = 0
        self.sam_obj_prior = None
        self.sam_obj_prior_valid = None
        self._support_low_map = None
        self._prior_256_cache = None

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _slice_and_project(
        self, emb: torch.Tensor, proj: nn.Module, norm: nn.Module
    ) -> torch.Tensor:
        if self.inst_norm:
            emb = F.normalize(emb, dim=-1)
        emb = emb[..., : self.inst_slice_dim]
        out = proj(emb)
        out = norm(out)
        return out

    def _build_inst_prompts(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb_v = self.inst_image_emb
        emb_t = self.inst_text_emb
        if emb_t.dim() == 2:
            emb_t = emb_t.unsqueeze(0).expand(emb_v.size(0), -1, -1)
        p_v = self._slice_and_project(emb_v, self.inst_proj_vis, self.inst_norm_vis)
        p_t = self._slice_and_project(emb_t, self.inst_proj_text, self.inst_norm_text)
        scores = (p_v * p_t).sum(dim=-1)
        tau = self.inst_tau
        if not torch.is_tensor(tau):
            tau = torch.tensor(float(tau), device=scores.device)
        scores = scores / torch.clamp(tau, min=1e-06)
        alpha = torch.softmax(scores, dim=-1)
        top_k = self.inst_top_k or 0
        if top_k <= 0 or top_k >= alpha.size(1):
            p_vis = (alpha.unsqueeze(-1) * p_v).sum(dim=1)
            p_text = (alpha.unsqueeze(-1) * p_t).sum(dim=1)
            return (p_text.unsqueeze(1), p_vis.unsqueeze(1), alpha)
        top_k = min(top_k, alpha.size(1))
        (vals, idx) = torch.topk(alpha, k=top_k, dim=1, largest=True, sorted=True)
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, p_v.size(-1))
        p_vis = torch.gather(p_v, 1, gather_idx)
        p_text = torch.gather(p_t, 1, gather_idx)
        if self.inst_alpha_scale:
            weight = vals / (vals.sum(dim=1, keepdim=True) + 1e-06)
            p_vis = p_vis * weight.unsqueeze(-1)
            p_text = p_text * weight.unsqueeze(-1)
        return (p_text, p_vis, alpha)

    def _warmup_mult(self, steps: int, start: float) -> float:
        if not self.training or steps <= 0:
            return 1.0
        s = max(0.0, min(1.0, float(start)))
        prog = float(min(self._global_step + 1, steps)) / float(max(steps, 1))
        return s + (1.0 - s) * prog

    def _decode(
        self,
        features,
        interm_embeddings,
        image_pe,
        dense_embeddings,
        cond_embed,
        routing_map=None,
        router_token=None,
        cond_conf=None,
    ):
        return self.mask_decoder(
            image_embeddings=features,
            interm_embeddings=interm_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=cond_embed,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            neg_prompt_token=None,
            routing_map=routing_map,
            router_token=router_token,
            cond_conf=cond_conf,
        )

    def _edge_completion(self, edge_prob, edge_embedding):
        if not self.edge_comp_enable:
            return edge_prob
        if edge_prob.dim() == 4 and edge_prob.shape[1] != 1:
            edge_prob = edge_prob.mean(dim=1, keepdim=True)
        anchor = (edge_prob > self.edge_comp_threshold).float()
        if self.training and self.edge_comp_drop_prob > 0:
            drop = (torch.rand_like(edge_prob) < self.edge_comp_drop_prob).float()
            anchor = anchor * (1 - drop)
        masked_edge = edge_prob * anchor
        edge_inp = torch.cat([masked_edge, anchor, edge_embedding], dim=1)
        edge_hat = torch.sigmoid(self.edge_completion(edge_inp))
        return edge_hat

    def forward(self, images, instruction_images, instruction_texts):
        self.input = images
        self.inst_image_emb = instruction_images
        self.inst_text_emb = instruction_texts
        bs = self.input.shape[0]
        attn_bias_runtime_scale = self.attn_bias_scale * self._warmup_mult(
            self.attn_bias_warmup_steps, self.attn_bias_warmup_start
        )
        boundary_eta_runtime = self.boundary_eta * self._warmup_mult(
            self.boundary_warmup_steps, self.boundary_warmup_start
        )
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )
        (features, interm_embeddings) = self.image_encoder(self.input, interm=True)
        image_pe = self.get_dense_pe()
        prior_full = None
        prior_64 = None
        prior_256 = None
        if self.struct_prior is not None:
            with torch.no_grad():
                prior_full = self.struct_prior(self.input)
            if prior_full is not None:
                prior_64 = F.interpolate(
                    prior_full,
                    size=(self.image_embedding_size, self.image_embedding_size),
                    mode="bilinear",
                    align_corners=False,
                )
                prior_256 = F.interpolate(
                    prior_full,
                    size=(self.image_embedding_size * 4, self.image_embedding_size * 4),
                    mode="bilinear",
                    align_corners=False,
                )
        self._prior_256_cache = prior_256.detach() if prior_256 is not None else None
        self._support_low_map = None
        if prior_256 is not None:
            ch = prior_256.shape[1]
            if ch >= 2:
                self._support_low_map = torch.clamp(
                    0.7 * prior_256[:, 0:1] + 0.3 * (1.0 - prior_256[:, 1:2]), 0.0, 1.0
                ).detach()
            else:
                self._support_low_map = torch.clamp(prior_256[:, 0:1], 0.0, 1.0).detach()
        if self.struct_film_enable and prior_64 is not None:
            (gamma, beta) = self.struct_film(prior_64)
            features = (
                features * (1.0 + self.struct_film_gamma_scale * gamma)
                + self.struct_film_beta_scale * beta
            )
        (p_text, p_vis, alpha) = self._build_inst_prompts()
        cond_embed = torch.cat([p_text, p_vis], dim=1)
        routing_map = None
        router_token = None
        cond_conf = None
        if self.attn_bias_enable and prior_64 is not None and (self.attn_bias_conv is not None):
            bias = self.attn_bias_conv(prior_64)
            bias = bias - bias.mean(dim=(2, 3), keepdim=True)
            bias_std = bias.std(dim=(2, 3), keepdim=True).clamp(min=1e-06)
            bias = bias / bias_std
            bias = torch.clamp(bias, -self.attn_bias_clamp, self.attn_bias_clamp)
            bias = bias * attn_bias_runtime_scale
            if self.attn_bias_use_inst and alpha is not None:
                alpha_scale = alpha.max(dim=1, keepdim=True).values.view(-1, 1, 1, 1)
                bias = bias * alpha_scale
            routing_map = bias.repeat(1, self.prompt_embed_dim, 1, 1)
            router_token = cond_embed.mean(dim=1, keepdim=True)
            if alpha is not None:
                cond_conf = alpha.max(dim=1, keepdim=True).values
        (low_res_masks, low_res_edges, _, aux) = self._decode(
            features,
            interm_embeddings,
            image_pe,
            dense_embeddings,
            cond_embed,
            routing_map=routing_map,
            router_token=router_token,
            cond_conf=cond_conf,
        )
        edge_embedding = aux.get("edge_embedding") if aux is not None else None
        edge_hat = low_res_edges
        if self.edge_comp_enable and edge_embedding is not None:
            edge_hat = self._edge_completion(low_res_edges, edge_embedding)
        masks_fine = low_res_masks + low_res_masks * edge_hat
        if self.boundary_refine_enable and self.boundary_refine is not None:
            p = torch.sigmoid(masks_fine)
            band = 4.0 * p * (1.0 - p)
            inputs = [band]
            if prior_256 is not None:
                inputs.append(prior_256)
            if self.boundary_use_edge_embed and edge_embedding is not None:
                inputs.append(edge_embedding)
            refine_inp = torch.cat(inputs, dim=1)
            delta = self.boundary_refine(refine_inp)
            masks_fine = masks_fine + boundary_eta_runtime * band * delta
        masks = self.postprocess_masks(masks_fine, self.inp_size, self.inp_size)
        edges = self.postprocess_masks(edge_hat, self.inp_size, self.inp_size)
        self.pred_mask = masks
        self.pred_edge = edges
        self.pred_mask_low = masks_fine
        self.pred_edge_low = edge_hat
        self._alpha_cache = alpha
        return {
            "mask_logits": self.pred_mask,
            "edge_prob": self.pred_edge,
            "view_weights": self._alpha_cache,
        }

    def postprocess_masks(
        self, masks: torch.Tensor, input_size: Tuple[int, ...], original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks, size=(input_size, input_size), mode="bilinear", align_corners=False
        )
        masks = masks[..., :input_size, :input_size]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks


class IDSAM(_IDSAMBase):
    """Instruction-driven structure-aware SAM with checkpoint-compatible parameter names."""

    def __init__(
        self,
        inp_size=None,
        encoder_mode=None,
        loss=None,
        inst_prompt=None,
        edge_completion=None,
        structure_prior=None,
        structure_film=None,
        attention_bias=None,
        boundary_refine=None,
        loss_extra=None,
    ):
        super().__init__(
            inp_size=inp_size,
            encoder_mode=encoder_mode,
            loss=loss,
            inst_prompt=inst_prompt,
            edge_completion=edge_completion,
            structure_prior=structure_prior,
            structure_film=structure_film,
            attention_bias=attention_bias,
            boundary_refine=boundary_refine,
            loss_extra=loss_extra,
        )
        struct_cfg = dict(structure_prior or {})
        struct_cfg["enable"] = True
        self.struct_prior = StructurePriorBank(struct_cfg)
        self.struct_enable = True
        self.struct_prior_channels = int(struct_cfg.get("topk_struct", 2))
        self.struct_prior_channels = max(1, self.struct_prior_channels)
        film_cfg = structure_film or {}
        attn_cfg = attention_bias or {}
        if self.struct_film_enable:
            hidden = int(film_cfg.get("hidden", 64))
            self.struct_film = StructureFiLM(
                self.struct_prior_channels, self.prompt_embed_dim, hidden
            )
        if self.attn_bias_enable:
            self.attn_bias_conv = nn.Conv2d(self.struct_prior_channels, 1, kernel_size=1)
            nn.init.zeros_(self.attn_bias_conv.weight)
            nn.init.zeros_(self.attn_bias_conv.bias)
        if self.boundary_refine_enable and self.boundary_refine is not None:
            expected_in_ch = (
                1
                + self.struct_prior_channels
                + (self.prompt_embed_dim // 8 if self.boundary_use_edge_embed else 0)
            )
            current_in_ch = int(self.boundary_refine.block1.conv.in_channels)
            if current_in_ch != expected_in_ch:
                hidden = int((boundary_refine or {}).get("hidden", 32))
                self.boundary_refine = BoundaryRefineHead(expected_in_ch, hidden)
