# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# licenses/SAM-Apache-2.0.txt file in this repository.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masksEE
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        # edge tokens
        self.edge_token = nn.Embedding(1, transformer_dim)
        self.edge_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.neg_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)

        # self.compress_vit_feat = nn.Sequential(
        #     nn.ConvTranspose2d(160, transformer_dim, 2, 2),
        #     LayerNorm2d(transformer_dim),
        #     nn.GELU(),
        #     nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, 2, 2)
        # )
        self.embedding_encoder = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, 2, 2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, 2, 2),
        )
        self.embedding_maskfeature = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        image_embeddings: torch.Tensor,
        interm_embeddings: List[torch.Tensor],
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        neg_prompt_token: torch.Tensor = None,
        routing_map: torch.Tensor = None,
        router_token: torch.Tensor = None,
        cond_conf: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        edge_features = self.embedding_encoder(image_embeddings)  # deep feature
        masks, edge, iou_pred, aux = self.predict_masks(
            image_embeddings=image_embeddings,
            edge_embeddings=edge_features,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            neg_prompt_token=neg_prompt_token,
            routing_map=routing_map,
            router_token=router_token,
            cond_conf=cond_conf,
        )

        # Select the correct mask or masks for outptu
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        if aux is not None:
            aux["edge_features"] = edge_features
        return masks, edge, iou_pred, aux

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        edge_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        neg_prompt_token: torch.Tensor = None,
        routing_map: torch.Tensor = None,
        router_token: torch.Tensor = None,
        cond_conf: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight, self.edge_token.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        if neg_prompt_token is not None:
            if neg_prompt_token.dim() == 2:
                neg_prompt_token = neg_prompt_token.unsqueeze(1)
            output_tokens = torch.cat([output_tokens, neg_prompt_token], dim=1)
        # tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        tokens = output_tokens
        cond_embedding = sparse_prompt_embeddings
        # Expand per-image data in batch direction to be per-mask (only when needed)
        src = image_embeddings
        pos_src = image_pe
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
            pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
            dense_prompt_embeddings = torch.repeat_interleave(
                dense_prompt_embeddings, tokens.shape[0], dim=0
            )
        src = src + dense_prompt_embeddings
        b, c, h, w = src.shape

        routing_query_mask = None
        if routing_map is not None:
            # Routing bias is only applied to mask tokens to avoid destabilizing IoU/edge token paths.
            routing_query_mask = torch.zeros(
                (tokens.shape[0], tokens.shape[1], 1),
                device=tokens.device,
                dtype=tokens.dtype,
            )
            q_start = 1
            q_end = min(q_start + self.num_mask_tokens, tokens.shape[1])
            if q_end > q_start:
                routing_query_mask[:, q_start:q_end, :] = 1.0

        # Run the transformer
        hs, src = self.transformer(
            src,
            pos_src,
            tokens,
            cond_embedding,
            routing_map=routing_map,
            router_token=router_token,
            cond_conf=cond_conf,
            routing_query_mask=routing_query_mask,
        )
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]
        edge_token_out = hs[:, (1 + self.num_mask_tokens) : (2 + self.num_mask_tokens), :]
        neg_token_out = None
        if neg_prompt_token is not None:
            neg_token_out = hs[:, (2 + self.num_mask_tokens) : (3 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)

        if edge_embeddings.shape[0] == b:
            edge_embeddings_rep = edge_embeddings
        else:
            edge_embeddings_rep = edge_embeddings.repeat(b, 1, 1, 1)
        edge_embedding = self.embedding_maskfeature(upscaled_embedding) + edge_embeddings_rep

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)

        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        # Keep edge prediction as [B,1,H,W] regardless of batch size.
        edge_hyper = self.edge_mlp(edge_token_out.squeeze(1)).unsqueeze(1)  # [B,1,C]
        edge = (edge_hyper @ edge_embedding.view(b, c, h * w)).view(b, -1, h, w)
        masks_neg = None
        if neg_token_out is not None:
            neg_hyper = self.neg_mlp(neg_token_out.squeeze(1)).unsqueeze(1)  # [B,1,C]
            masks_neg = (neg_hyper @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        edge = self.sigmoid(edge)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        aux = {
            "mask_tokens": mask_tokens_out,
            "neg_token": neg_token_out,
            "masks_neg": masks_neg,
            "edge_embedding": edge_embedding,
            "router_token_out": getattr(self.transformer, "last_router_token", None),
        }
        return masks, edge, iou_pred, aux


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
