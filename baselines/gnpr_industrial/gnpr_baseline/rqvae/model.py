"""论文基线使用的纯重构式 CRQ-VAE。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .mlp import MLPLayers
from .rq import ResidualVectorQuantizer


class CRQVAE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_emb_list: list[int],
        e_dim: int = 64,
        layers: list[int] | None = None,
        dropout_prob: float = 0.1,
        bn: bool = True,
        loss_type: str = "mse",
        quant_loss_weight: float = 0.5,
        beta: float = 0.25,
        kmeans_init: bool = True,
        kmeans_iters: int = 100,
        sk_epsilons: list[float] | None = None,
        sk_iters: int = 50,
        use_linear: int = 1,
    ):
        super().__init__()
        layers = layers or [512, 256, 128]
        sk_epsilons = sk_epsilons or [0.1] * len(num_emb_list)
        if len(sk_epsilons) != len(num_emb_list):
            raise ValueError("sk_epsilons 数量必须与残差码层数一致")

        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        encoder_dims = [in_dim, *layers, e_dim]
        self.encoder = MLPLayers(encoder_dims, dropout=dropout_prob, bn=bn)
        self.rq = ResidualVectorQuantizer(
            num_emb_list,
            e_dim,
            beta=beta,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            sk_epsilons=sk_epsilons,
            sk_iters=sk_iters,
            use_linear=use_linear,
        )
        self.decoder = MLPLayers(list(reversed(encoder_dims)), dropout=dropout_prob, bn=bn)

    def forward(self, inputs: torch.Tensor, use_sk: bool = False):
        encoded = self.encoder(inputs)
        quantized, quant_loss, codes = self.rq(encoded, use_sk=use_sk)
        reconstructed = self.decoder(quantized)
        return reconstructed, quant_loss, codes

    def compute_loss(
        self,
        quant_loss: torch.Tensor,
        reconstructed: torch.Tensor,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.loss_type == "mse":
            reconstruction_loss = F.mse_loss(reconstructed, inputs)
        elif self.loss_type == "l1":
            reconstruction_loss = F.l1_loss(reconstructed, inputs)
        else:
            raise ValueError(f"不支持的 loss_type: {self.loss_type}")
        total = reconstruction_loss + self.quant_loss_weight * quant_loss
        return total, quant_loss, reconstruction_loss

    @torch.no_grad()
    def get_indices(self, inputs: torch.Tensor):
        encoded = self.encoder(inputs)
        quantized, _, (indices, _) = self.rq(encoded, use_sk=False)
        return quantized.cpu(), indices.cpu()
