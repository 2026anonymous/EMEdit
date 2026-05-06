"""
EMEdit model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_sequence(x: torch.Tensor) -> torch.Tensor:
    """Return a time-series tensor as [B, 1, L]."""
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() == 3:
        return x
    raise ValueError(f"Expected ts with shape [B, L] or [B, C, L], got {tuple(x.shape)}")


def _flatten_output(x: torch.Tensor) -> torch.Tensor:
    """Return decoder output as [B, L]."""
    if x.dim() == 3 and x.size(1) == 1:
        return x.squeeze(1)
    return x


class ConvBranch(nn.Module):
    """One temporal branch with a distinct receptive field."""

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
    """Residual temporal refinement block B^(l)."""

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
    """
    EMEdit time-series encoder:
        H_ts^(0) = Concat[phi_1(X), ..., phi_R(X)]
        H_ts^(l) = H_ts^(l-1) + B^(l)(H_ts^(l-1))
        E_ts = MLP(Pool(H_ts^(N)))
    """

    def __init__(
        self,
        ts_dim: int,
        output_dim: int,
        in_channels: int = 1,
        branch_kernels: Sequence[int] = (3, 5, 9, 15),
        branch_dim: Optional[int] = None,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ts_dim = ts_dim
        self.output_dim = output_dim
        branch_dim = branch_dim or max(32, output_dim // len(branch_kernels))

        self.branches = nn.ModuleList(
            [ConvBranch(in_channels, branch_dim, k, dropout) for k in branch_kernels]
        )
        concat_dim = branch_dim * len(branch_kernels)
        self.res_blocks = nn.Sequential(
            *[ResidualTemporalBlock(concat_dim, kernel_size=3, dropout=dropout) for _ in range(num_res_blocks)]
        )
        self.proj = nn.Sequential(
            nn.Linear(concat_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_sequence(x)  # [B, C, L]
        features = torch.cat([branch(x) for branch in self.branches], dim=1)  # [B, R*d, L]
        features = self.res_blocks(features)
        pooled = F.adaptive_avg_pool1d(features, output_size=1).squeeze(-1)
        return F.normalize(self.proj(pooled), dim=-1)


class PatchMLPTextEncoder(nn.Module):
    """Patch-MLP projection head for pretrained text embeddings."""

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

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        if text_features.dim() > 2:
            text_features = text_features.flatten(start_dim=1)
        return F.normalize(self.net(text_features.float()), dim=-1)


class GatedFusion(nn.Module):
    """Metadata-guided gated fusion G(x, meta)."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.candidate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x, meta], dim=-1)
        g = self.gate(h)
        z = g * self.candidate(h) + (1.0 - g) * x
        return F.normalize(self.norm(z), dim=-1)


class TransformerEditingDecoder(nn.Module):
    """Transformer decoder that maps two fused tokens Stack(Z_ts, Z_e) to length L."""

    def __init__(
        self,
        ts_dim: int,
        output_dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        if output_dim % num_heads != 0:
            # Keep the model robust for unusual hidden dimensions.
            num_heads = 1
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=output_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.pos = nn.Parameter(torch.zeros(1, 2, output_dim))
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(output_dim * 2),
            nn.Linear(output_dim * 2, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, ts_dim),
        )

    def forward(self, fused_tokens: torch.Tensor) -> torch.Tensor:
        if fused_tokens.dim() != 3 or fused_tokens.size(1) != 2:
            raise ValueError(f"Expected fused tokens [B, 2, D], got {tuple(fused_tokens.shape)}")
        h = self.blocks(fused_tokens + self.pos)
        return self.head(h.flatten(start_dim=1))


class RiemannianVolumeContrast(nn.Module):
    """3-modal SPD Gram log-volume and hard-negative RVC objective."""

    def __init__(
        self,
        eps: float = 1e-6,
        base_tau: float = 0.3,
        tau_min: float = 0.15,
        tau_max: float = 0.9,
        top_k: int = 60,
    ):
        super().__init__()
        self.eps = eps
        self.base_tau = base_tau
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.top_k = top_k

    def log_volume_matrix(self, ts_emb: torch.Tensor, event_emb: torch.Tensor, meta_emb: torch.Tensor) -> torch.Tensor:
        """
        Construct S_ij = G_ij + eps*I and return V_ij = log det(S_ij).

        Pairing follows the paper: ts_i is paired with event_j and meta_j;
        event_j-meta_j is computed within sample j and broadcast across i.
        """
        ts = F.normalize(ts_emb, dim=-1)
        event = F.normalize(event_emb, dim=-1)
        meta = F.normalize(meta_emb, dim=-1)
        b_ts, b_text = ts.size(0), event.size(0)
        device = ts.device

        ones = torch.ones(b_ts, b_text, device=device, dtype=ts.dtype)
        ts_event = ts @ event.T
        ts_meta = ts @ meta.T
        event_meta = torch.sum(event * meta, dim=-1).unsqueeze(0).expand(b_ts, -1)

        gram = torch.stack(
            [
                torch.stack([ones, ts_event, ts_meta], dim=-1),
                torch.stack([ts_event, ones, event_meta], dim=-1),
                torch.stack([ts_meta, event_meta, ones], dim=-1),
            ],
            dim=-2,
        )
        eye = torch.eye(3, device=device, dtype=ts.dtype).view(1, 1, 3, 3)
        spd = gram + self.eps * eye
        sign, logabsdet = torch.linalg.slogdet(spd.float())
        # In exact arithmetic sign should be positive. Clamp for numerical safety.
        safe_logdet = torch.where(sign > 0, logabsdet, torch.full_like(logabsdet, -30.0))
        return safe_logdet.to(ts.dtype)

    def forward(self, ts_emb: torch.Tensor, event_emb: torch.Tensor, meta_emb: torch.Tensor) -> torch.Tensor:
        return self.log_volume_matrix(ts_emb, event_emb, meta_emb)

    def loss(self, log_volume: torch.Tensor) -> torch.Tensor:
        """RVC loss: small matched log-volume, large hard-negative log-volume."""
        if log_volume.dim() != 2 or log_volume.size(0) != log_volume.size(1):
            raise ValueError(f"RVC expects a square [B, B] matrix, got {tuple(log_volume.shape)}")
        bsz = log_volume.size(0)
        if bsz <= 1:
            return log_volume.new_zeros(())

        tau = self.base_tau * (1.0 + log_volume.detach().abs().mean())
        tau = torch.clamp(tau, min=self.tau_min, max=self.tau_max)
        affinity = torch.exp(-log_volume / tau)

        pos = affinity.diagonal()
        neg = affinity.masked_fill(torch.eye(bsz, dtype=torch.bool, device=affinity.device), float("-inf"))
        k = min(self.top_k, bsz - 1)
        hard_neg = torch.topk(neg, k=k, dim=1).values
        denom = pos + hard_neg.sum(dim=1) + self.eps
        ratio = torch.clamp(pos / denom, min=self.eps, max=1.0 - self.eps)
        weights = (1.0 - ratio).pow(2) + self.eps
        return -(weights * torch.log(ratio)).mean()


class EMEdit(nn.Module):
    """
    Event-Meta Counterfactual Time Series Editor.

    Input:
        ts:             [B, L]
        event_features: [B, text_dim]  # Xe embedding, what to change
        meta_features:  [B, text_dim]  # Xm embedding, what to preserve
    Output dictionary contains x_hat and RVC-related embeddings.
    """

    def __init__(
        self,
        ts_dim: int,
        text_dim: int,
        output_dim: int,
        beta: float = 1.0,
        ts_encoder: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        ts_decoder: Optional[nn.Module] = None,
        variational: bool = False,
        clip_mu: bool = False,
        gen_w_src_text: bool = False,
        decoder_layers: int = 4,
        dropout: float = 0.1,
        **_: object,
    ):
        super().__init__()
        self.ts_dim = ts_dim
        self.text_dim = text_dim
        self.output_dim = output_dim
        self.beta = beta
        self.clip_mu = clip_mu
        self.variational = variational
        self.gen_w_src_text = gen_w_src_text

        self.ts_encoder = ts_encoder or MultiResolutionTSEncoder(ts_dim, output_dim, dropout=dropout)
        # Shared text encoder for event and metadata, consistent with EMEdit.
        self.text_encoder = text_encoder or PatchMLPTextEncoder(text_dim, output_dim, dropout=dropout)
        self.meta_encoder = self.text_encoder

        self.gate_ts = GatedFusion(output_dim, dropout=dropout)
        self.gate_event = GatedFusion(output_dim, dropout=dropout)
        self.rvc = RiemannianVolumeContrast()
        self.ts_decoder = ts_decoder or TransformerEditingDecoder(
            ts_dim=ts_dim,
            output_dim=output_dim,
            num_layers=decoder_layers,
            dropout=dropout,
        )

    def encode(self, ts: torch.Tensor, event_features: torch.Tensor, meta_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        e_ts = self.ts_encoder(ts)
        e_event = self.text_encoder(event_features)
        e_meta = self.meta_encoder(meta_features)
        return {"ts": e_ts, "event": e_event, "meta": e_meta}

    def fuse(self, e_ts: torch.Tensor, e_event: torch.Tensor, e_meta: torch.Tensor, w: float = 1.0) -> torch.Tensor:
        """
        Metadata-guided fusion H(E_ts, E_e, E_m) = Stack(G_ts(E_ts,E_m), G_e(E_e,E_m)).
        w controls the latent editing strength of the event branch.
        """
        w = float(w)
        # Latent interpolation: w=0 keeps event branch close to source; w=1 uses event semantics fully.
        event_control = F.normalize((1.0 - w) * e_ts + w * e_event, dim=-1)
        z_ts = self.gate_ts(e_ts, e_meta)
        z_event = self.gate_event(event_control, e_meta)
        return torch.stack([z_ts, z_event], dim=1)  # [B, 2, D]

    def forward(
        self,
        ts: torch.Tensor,
        event_features: torch.Tensor,
        meta_features: torch.Tensor,
        w: float = 1.0,
        return_dict: bool = False,
    ):
        emb = self.encode(ts, event_features, meta_features)
        log_volume = self.rvc(emb["ts"], emb["event"], emb["meta"])
        fused = self.fuse(emb["ts"], emb["event"], emb["meta"], w=w)
        x_hat = _flatten_output(self.ts_decoder(fused))

        out = {
            "x_hat": x_hat,
            "log_volume": log_volume,
            "ts_emb": emb["ts"],
            "event_emb": emb["event"],
            "meta_emb": emb["meta"],
            "fused": fused,
        }
        if return_dict:
            return out

        # Drop-in compatibility with the original training code:
        # logits, ts_hat, mean, log_var = model(ts, text_features, var_features)
        dummy_log_var = torch.zeros_like(emb["ts"])
        return log_volume, x_hat, emb["ts"], dummy_log_var

    @torch.no_grad()
    def generate(
        self,
        w: float,
        ts: torch.Tensor,
        tx_f_tgt: torch.Tensor,
        tx_f_src: Optional[torch.Tensor] = None,
        var: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generation API kept"""
        if var is None:
            raise ValueError("EMEdit requires metadata features `var` during generation.")
        e_ts = self.ts_encoder(ts)
        e_event = self.text_encoder(tx_f_tgt)
        e_meta = self.meta_encoder(var)
        e_src = self.text_encoder(tx_f_src) if tx_f_src is not None else e_event
        if self.gen_w_src_text:
            e_event = e_src
        fused = self.fuse(e_ts, e_event, e_meta, w=w)
        x_hat = _flatten_output(self.ts_decoder(fused))
        # Return names follow the previous code: ts_hat, ts_emb_tgt, tx_emb_tgt, tx_emb_src
        return x_hat, fused[:, 1, :], e_event, e_src


