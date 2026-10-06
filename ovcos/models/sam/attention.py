# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# licenses/SAM-Apache-2.0.txt file in this repository.

import torch
from torch import Tensor, nn
from torch.nn import functional as F

import math
from typing import Optional, Tuple, Type

from .common import MLPBlock


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        anti_attn_init: float = 1.0,
        auto_last_cond_neg: bool = True,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                    anti_attn_init=anti_attn_init,
                    auto_last_cond_neg=auto_last_cond_neg,
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)
        self.last_router_token: Optional[Tensor] = None

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
        cond_embedding: Tensor,
        routing_map: Optional[Tensor] = None,
        router_token: Optional[Tensor] = None,
        cond_conf: Optional[Tensor] = None,
        routing_query_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys, router_token = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
                cond_embedding=cond_embedding,
                cond_pe=cond_embedding,
                routing_map=routing_map,
                router_token=router_token,
                cond_conf=cond_conf,
                routing_query_mask=routing_query_mask,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        self.last_router_token = router_token

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
        anti_attn_init: float = 1.0,
        auto_last_cond_neg: bool = True,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_cond = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2_cond = nn.LayerNorm(embedding_dim)
        self.anti_attn_scale = nn.Parameter(torch.tensor(anti_attn_init))

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.norm4_cond = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_cond = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.router_attn = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.router_norm = nn.LayerNorm(embedding_dim)
        self.router_bias_scale = nn.Parameter(torch.tensor(0.0))
        self.router_lambda_mode = "guided"
        self.router_lambda_const = 0.5
        self.auto_last_cond_neg = bool(auto_last_cond_neg)

        self.skip_first_layer_pe = skip_first_layer_pe

    @staticmethod
    def _flatten_map(routing_map: Optional[Tensor], key_tokens: Tensor) -> Optional[Tensor]:
        if routing_map is None:
            return None
        if routing_map.dim() == 4:
            return routing_map.flatten(2).permute(0, 2, 1)
        if routing_map.dim() == 3:
            return routing_map
        return None

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        query_pe: Tensor,
        key_pe: Tensor,
        cond_embedding: Tensor,
        cond_pe: Tensor,
        routing_map: Optional[Tensor] = None,
        router_token: Optional[Tensor] = None,
        cond_conf: Optional[Tensor] = None,
        routing_query_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        cond_pos = cond_embedding
        cond_neg = None
        cond_pe_pos = cond_pe
        cond_pe_neg = None
        if self.auto_last_cond_neg and cond_embedding.shape[1] >= 3:
            cond_pos = cond_embedding[:, :-1, :]
            cond_neg = cond_embedding[:, -1:, :]
            cond_pe_pos = cond_pe[:, :-1, :]
            cond_pe_neg = cond_pe[:, -1:, :]

        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        routing_tokens = self._flatten_map(routing_map, keys)
        if (routing_tokens is not None) and (router_token is not None):
            router_kv = F.normalize(routing_tokens, dim=-1)
            router_q = F.normalize(router_token, dim=-1)
            router_upd = self.router_attn(q=router_q, k=router_kv, v=router_kv)
            router_token = self.router_norm(router_token + router_upd)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_bias = None
        if (routing_tokens is not None) and (router_token is not None):
            qn = F.normalize(q, dim=-1)
            rn = F.normalize(router_token, dim=-1)
            kn = F.normalize(routing_tokens, dim=-1)
            q_gate = torch.sigmoid(torch.bmm(qn, rn.transpose(1, 2)))  # [B, Nq, 1]
            if routing_query_mask is not None:
                rq = routing_query_mask
                if rq.dim() == 2:
                    rq = rq.unsqueeze(-1)
                if rq.shape[0] == 1 and q_gate.shape[0] > 1:
                    rq = rq.expand(q_gate.shape[0], -1, -1)
                if rq.shape[1] != q_gate.shape[1]:
                    q_mask = torch.zeros_like(q_gate)
                    n = min(rq.shape[1], q_gate.shape[1])
                    q_mask[:, :n, :] = rq[:, :n, :1]
                else:
                    q_mask = rq[:, :, :1]
                q_gate = q_gate * torch.clamp(q_mask.to(q_gate.dtype), 0.0, 1.0)
            k_bias = torch.bmm(rn, kn.transpose(1, 2))  # [B, 1, Nk]
            mode = str(getattr(self, "router_lambda_mode", "guided")).lower()
            if mode in {"always_on", "always", "on"}:
                lam = torch.ones((q_gate.shape[0], 1, 1), device=q_gate.device, dtype=q_gate.dtype)
            elif mode in {"constant", "const", "fixed"}:
                lam_value = float(getattr(self, "router_lambda_const", 0.5))
                lam_value = min(max(lam_value, 0.0), 1.0)
                lam = torch.full(
                    (q_gate.shape[0], 1, 1), lam_value, device=q_gate.device, dtype=q_gate.dtype
                )
            else:
                lam = torch.sigmoid(self.router_bias_scale).to(dtype=q_gate.dtype).view(1, 1, 1)
                if cond_conf is not None:
                    conf = cond_conf.view(cond_conf.shape[0], -1)[:, :1]
                    lam = lam * (0.5 + 0.5 * (1.0 - torch.clamp(conf, 0.0, 1.0))).view(-1, 1, 1)
                if lam.shape[0] == 1 and q_gate.shape[0] > 1:
                    lam = lam.expand(q_gate.shape[0], -1, -1)
                lam = torch.clamp(lam, 0.0, 1.0)
            attn_bias = torch.clamp(lam * (q_gate * k_bias), -1.5, 1.5).unsqueeze(1)  # [B,1,Nq,Nk]
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys, attn_bias=attn_bias)
        queries = queries + attn_out
        queries = self.norm2(queries)
        # add cond_embedding (positive/visual), subtract hard-negative if present
        q = queries + query_pe
        k = cond_pos + cond_pe_pos
        attn_out = self.cross_attn_token_to_cond(q=q, k=k, v=cond_pos)
        queries = queries + attn_out
        if cond_neg is not None:
            k_neg = cond_neg + cond_pe_neg
            attn_out_neg = self.cross_attn_token_to_cond(q=q, k=k_neg, v=cond_neg)
            queries = queries - torch.sigmoid(self.anti_attn_scale) * attn_out_neg
        queries = self.norm2_cond(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # add cond_embedding (positive/visual only)
        q = cond_pos + cond_pe_pos
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_cond(q=k, k=q, v=cond_pos)
        keys = keys + attn_out
        keys = self.norm4_cond(keys)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys, router_token


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, attn_bias: Optional[Tensor] = None
    ) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        if attn_bias is not None:
            attn = attn + attn_bias
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out
