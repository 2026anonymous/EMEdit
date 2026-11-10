from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def as_sequence(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() == 3 and x.size(1) == 1:
        return x
    raise ValueError(f"Expected [B, L] or [B, 1, L], got {tuple(x.shape)}")


class ConvBranch(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.BatchNorm1d(dim)
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel_size, padding=padding),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(self.norm(x))


class MultiResolutionTSEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        branch_kernels: Sequence[int] = (3, 5, 9, 15),
        branch_dim: Optional[int] = None,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        branch_dim = branch_dim or max(32, output_dim // len(branch_kernels))
        self.branches = nn.ModuleList(
            [ConvBranch(1, branch_dim, kernel, dropout) for kernel in branch_kernels]
        )
        concat_dim = branch_dim * len(branch_kernels)
        self.res_blocks = nn.Sequential(
            *[
                ResidualTemporalBlock(concat_dim, 3, dropout)
                for _ in range(num_res_blocks)
            ]
        )
        self.projection = nn.Sequential(
            nn.Linear(concat_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = as_sequence(x)
        features = torch.cat([branch(x) for branch in self.branches], dim=1)
        features = self.res_blocks(features)
        pooled = F.adaptive_avg_pool1d(features, 1).squeeze(-1)
        return self.projection(pooled)


class TextProjection(nn.Module):
    def __init__(self, text_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden_dim = max(output_dim, min(4096, text_dim))
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        leading_shape = features.shape[:-1]
        flattened = features.reshape(-1, features.size(-1))
        projected = self.net(flattened.float())
        return projected.reshape(*leading_shape, projected.size(-1))


class EventConditionedMetadataRouter(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        event_embedding: torch.Tensor,
        metadata_embedding: torch.Tensor,
        stable_context_embeddings: Optional[torch.Tensor] = None,
        stable_context_mask: Optional[torch.Tensor] = None,
    ):
        if stable_context_embeddings is None or stable_context_embeddings.numel() == 0:
            return metadata_embedding, None

        if stable_context_embeddings.dim() == 2:
            stable_context_embeddings = stable_context_embeddings.unsqueeze(1)

        batch_size, context_count, _ = stable_context_embeddings.shape
        if stable_context_mask is None:
            stable_context_mask = torch.ones(
                batch_size,
                context_count,
                dtype=torch.bool,
                device=stable_context_embeddings.device,
            )
        else:
            stable_context_mask = stable_context_mask.to(
                device=stable_context_embeddings.device,
                dtype=torch.bool,
            )

        query = self.query(event_embedding).unsqueeze(1)
        keys = self.key(stable_context_embeddings)
        scores = (query * keys).sum(dim=-1) / math.sqrt(event_embedding.size(-1))
        scores = scores.masked_fill(~stable_context_mask, -1e4)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * stable_context_mask.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        context = torch.sum(
            weights.unsqueeze(-1) * stable_context_embeddings,
            dim=1,
        )
        routed = self.value(context)
        has_context = stable_context_mask.any(dim=1, keepdim=True)
        metadata_event = metadata_embedding + routed * has_context.to(routed.dtype)
        return metadata_event, weights


class SourceAnchoredDualPathFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.metadata_gate = nn.Linear(dim * 2, dim)
        self.event_gate = nn.Linear(dim * 3, dim)
        self.metadata_projection = nn.Linear(dim, dim, bias=False)
        self.event_projection = nn.Linear(dim, dim, bias=False)
        self.routed_metadata_projection = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        ts_embedding: torch.Tensor,
        event_embedding: torch.Tensor,
        metadata_embedding: torch.Tensor,
        event_conditioned_metadata: torch.Tensor,
    ):
        gate_metadata = torch.sigmoid(
            self.metadata_gate(torch.cat([ts_embedding, metadata_embedding], dim=-1))
        )
        gate_event = torch.sigmoid(
            self.event_gate(
                torch.cat(
                    [ts_embedding, event_embedding, event_conditioned_metadata],
                    dim=-1,
                )
            )
        )
        z_metadata = ts_embedding + gate_metadata * self.metadata_projection(metadata_embedding)
        event_correction = (
            self.event_projection(event_embedding)
            + self.routed_metadata_projection(event_conditioned_metadata)
        )
        z_event = ts_embedding + gate_event * event_correction
        return z_metadata, z_event


class TransformerEditingDecoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        seq_length: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        if output_dim % num_heads != 0:
            raise ValueError("output_dim must be divisible by num_heads")
        layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=output_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.position = nn.Parameter(torch.zeros(1, 2, output_dim))
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(output_dim * 2),
            nn.Linear(output_dim * 2, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, seq_length),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3 or tokens.size(1) != 2:
            raise ValueError(f"Expected [B, 2, D], got {tuple(tokens.shape)}")
        hidden = self.transformer(tokens + self.position)
        return self.head(hidden.flatten(start_dim=1))


class RiemannianVolumeContrast(nn.Module):
    def __init__(
        self,
        eps: float = 1e-4,
        tau0: float = 0.07,
        tau_min: float = 0.01,
        tau_max: float = 1.0,
        top_k: int = 60,
        eps_s: float = 1e-8,
    ):
        super().__init__()
        self.eps = eps
        self.tau0 = tau0
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.top_k = top_k
        self.eps_s = eps_s

    def log_volume_matrix(
        self,
        ts_embedding: torch.Tensor,
        event_embedding: torch.Tensor,
        metadata_embedding: torch.Tensor,
    ) -> torch.Tensor:
        ts = F.normalize(ts_embedding.float(), dim=-1)
        event = F.normalize(event_embedding.float(), dim=-1)
        metadata = F.normalize(metadata_embedding.float(), dim=-1)

        if ts.size(0) != event.size(0) or ts.size(0) != metadata.size(0):
            raise ValueError("The three modalities must have the same batch size")

        batch_size = ts.size(0)
        a = ts @ event.transpose(0, 1)
        b = ts @ metadata.transpose(0, 1)
        c = (event * metadata).sum(dim=-1).unsqueeze(0).expand(batch_size, -1)
        ones = torch.ones_like(a)

        gram = torch.stack(
            [
                torch.stack([ones, a, b], dim=-1),
                torch.stack([a, ones, c], dim=-1),
                torch.stack([b, c, ones], dim=-1),
            ],
            dim=-2,
        )
        identity = torch.eye(3, device=gram.device, dtype=gram.dtype).view(1, 1, 3, 3)
        spd = gram + self.eps * identity
        sign, logdet = torch.linalg.slogdet(spd)
        return torch.where(sign > 0, logdet, torch.full_like(logdet, -30.0))

    def forward(
        self,
        ts_embedding: torch.Tensor,
        event_embedding: torch.Tensor,
        metadata_embedding: torch.Tensor,
    ) -> torch.Tensor:
        return self.log_volume_matrix(ts_embedding, event_embedding, metadata_embedding)

    def loss(
        self,
        log_volume: torch.Tensor,
        valid_negative_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if log_volume.dim() != 2 or log_volume.size(0) != log_volume.size(1):
            raise ValueError("RVC expects a square [B, B] log-volume matrix")

        batch_size = log_volume.size(0)
        if batch_size <= 1:
            return log_volume.new_zeros(())

        device = log_volume.device
        diagonal = torch.eye(batch_size, dtype=torch.bool, device=device)

        if valid_negative_mask is None:
            valid = ~diagonal
        else:
            if valid_negative_mask.shape != log_volume.shape:
                raise ValueError("valid_negative_mask must have shape [B, B]")
            valid = valid_negative_mask.to(device=device, dtype=torch.bool) & ~diagonal

        tau = self.tau0 * (1.0 + log_volume.detach().abs().mean())
        tau = torch.clamp(tau, min=self.tau_min, max=self.tau_max)
        log_affinity = -log_volume.float() / tau.float()
        candidate_mask = valid | diagonal
        log_affinity = log_affinity.masked_fill(~candidate_mask, float("-inf"))
        row_max = log_affinity.max(dim=1, keepdim=True).values
        row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        affinity = torch.exp(log_affinity - row_max)

        positive = affinity.diagonal()
        negative = affinity.masked_fill(~valid, float("-inf"))
        k = min(self.top_k, batch_size - 1)
        selected = torch.topk(negative, k=k, dim=1).values
        selected = torch.where(torch.isfinite(selected), selected, torch.zeros_like(selected))
        denominator = positive + selected.sum(dim=1) + self.eps_s
        ratio = (positive / denominator).clamp(
            min=self.eps_s,
            max=1.0 - self.eps_s,
        )
        row_loss = -(((1.0 - ratio).pow(2) + self.eps_s) * torch.log(ratio))
        valid_rows = valid.any(dim=1)
        if not valid_rows.any():
            return log_volume.new_zeros(())
        return row_loss[valid_rows].mean().to(log_volume.dtype)


class EMEdit(nn.Module):
    def __init__(
        self,
        seq_length: int,
        text_dim: int,
        hidden_dim: int = 768,
        branch_kernels: Sequence[int] = (3, 5, 9, 15),
        branch_dim: Optional[int] = None,
        num_res_blocks: int = 2,
        decoder_layers: int = 4,
        decoder_heads: int = 8,
        dropout: float = 0.1,
        rvc_eps: float = 1e-4,
        rvc_tau0: float = 0.07,
        rvc_tau_min: float = 0.01,
        rvc_tau_max: float = 1.0,
        rvc_top_k: int = 60,
        rvc_eps_s: float = 1e-8,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.ts_encoder = MultiResolutionTSEncoder(
            output_dim=hidden_dim,
            branch_kernels=branch_kernels,
            branch_dim=branch_dim,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
        )
        self.text_projection = TextProjection(text_dim, hidden_dim, dropout)
        self.metadata_router = EventConditionedMetadataRouter(hidden_dim)
        self.dual_path_fusion = SourceAnchoredDualPathFusion(hidden_dim)
        self.edit_gate = nn.Linear(hidden_dim * 3, hidden_dim)
        self.decoder = TransformerEditingDecoder(
            output_dim=hidden_dim,
            seq_length=seq_length,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            dropout=dropout,
        )
        self.rvc = RiemannianVolumeContrast(
            eps=rvc_eps,
            tau0=rvc_tau0,
            tau_min=rvc_tau_min,
            tau_max=rvc_tau_max,
            top_k=rvc_top_k,
            eps_s=rvc_eps_s,
        )

    def encode_embeddings(
        self,
        ts: torch.Tensor,
        event_features: torch.Tensor,
        metadata_features: torch.Tensor,
        stable_context_features: Optional[torch.Tensor] = None,
    ):
        ts_embedding = self.ts_encoder(ts)
        event_embedding = self.text_projection(event_features)
        metadata_embedding = self.text_projection(metadata_features)
        context_embeddings = None
        if stable_context_features is not None:
            if stable_context_features.numel() > 0:
                context_embeddings = self.text_projection(stable_context_features)
        return ts_embedding, event_embedding, metadata_embedding, context_embeddings

    def forward(
        self,
        ts: torch.Tensor,
        event_features: torch.Tensor,
        metadata_features: torch.Tensor,
        stable_context_features: Optional[torch.Tensor] = None,
        stable_context_mask: Optional[torch.Tensor] = None,
        editing_strength: float | torch.Tensor = 1.0,
        valid_negative_mask: Optional[torch.Tensor] = None,
        compute_rvc: bool = True,
    ):
        ts_embedding, event_embedding, metadata_embedding, context_embeddings = (
            self.encode_embeddings(
                ts,
                event_features,
                metadata_features,
                stable_context_features,
            )
        )

        event_conditioned_metadata, routing_weights = self.metadata_router(
            event_embedding,
            metadata_embedding,
            context_embeddings,
            stable_context_mask,
        )

        z_metadata, z_event = self.dual_path_fusion(
            ts_embedding,
            event_embedding,
            metadata_embedding,
            event_conditioned_metadata,
        )

        latent_gate = torch.sigmoid(
            self.edit_gate(
                torch.cat([z_metadata, z_event, event_conditioned_metadata], dim=-1)
            )
        )

        strength = torch.as_tensor(
            editing_strength,
            device=z_metadata.device,
            dtype=z_metadata.dtype,
        )
        if strength.ndim == 0:
            strength = strength.view(1, 1)
        elif strength.ndim == 1:
            strength = strength.unsqueeze(-1)

        editing_direction = z_event - z_metadata
        edited_token = z_metadata + strength * latent_gate * editing_direction
        tokens = torch.stack([z_metadata, edited_token], dim=1)
        x_hat = self.decoder(tokens)

        log_volume = None
        if compute_rvc:
            log_volume = self.rvc(
                ts_embedding,
                event_embedding,
                metadata_embedding,
            )

        return {
            "x_hat": x_hat,
            "log_volume": log_volume,
            "ts_emb": ts_embedding,
            "event_emb": event_embedding,
            "meta_emb": metadata_embedding,
            "event_conditioned_meta": event_conditioned_metadata,
            "z_metadata": z_metadata,
            "z_event": z_event,
            "latent_gate": latent_gate,
            "routing_weights": routing_weights,
            "tokens": tokens,
        }

    @torch.no_grad()
    def edit(
        self,
        ts: torch.Tensor,
        target_event_features: torch.Tensor,
        metadata_features: torch.Tensor,
        stable_context_features: Optional[torch.Tensor] = None,
        stable_context_mask: Optional[torch.Tensor] = None,
        editing_strength: float = 1.0,
    ) -> torch.Tensor:
        self.eval()
        output = self.forward(
            ts=ts,
            event_features=target_event_features,
            metadata_features=metadata_features,
            stable_context_features=stable_context_features,
            stable_context_mask=stable_context_mask,
            editing_strength=editing_strength,
            compute_rvc=False,
        )
        return output["x_hat"]
