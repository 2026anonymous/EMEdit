"""
Training utilities for EMEdit.
"""

from __future__ import annotations

import copy
import os
import random
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def rvc_loss_from_matrix(log_volume: torch.Tensor, base_tau: float = 0.3, top_k: int = 60) -> torch.Tensor:
    """Standalone RVC loss, matching model.rvc.loss(log_volume)."""
    if log_volume.dim() != 2 or log_volume.size(0) != log_volume.size(1):
        raise ValueError(f"RVC expects square [B, B] log-volume matrix, got {tuple(log_volume.shape)}")
    bsz = log_volume.size(0)
    if bsz <= 1:
        return log_volume.new_zeros(())

    eps = 1e-8
    tau = base_tau * (1.0 + log_volume.detach().abs().mean())
    tau = torch.clamp(tau, min=0.15, max=0.9)
    affinity = torch.exp(-log_volume / tau)

    pos = affinity.diagonal()
    neg = affinity.masked_fill(torch.eye(bsz, dtype=torch.bool, device=affinity.device), float("-inf"))
    k = min(top_k, bsz - 1)
    hard_neg = torch.topk(neg, k=k, dim=1).values
    denom = pos + hard_neg.sum(dim=1) + eps
    ratio = torch.clamp(pos / denom, min=eps, max=1.0 - eps)
    weights = (1.0 - ratio).pow(2) + eps
    return -(weights * torch.log(ratio)).mean()


def compute_reconstruction_loss_mine(ts: torch.Tensor, ts_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Hybrid temporal-frequency reconstruction loss in the EMEdit paper."""
    ts = ts.float()
    ts_hat = ts_hat.float()
    temporal_loss = F.l1_loss(ts_hat, ts)
    freq_loss = torch.mean(torch.abs(torch.fft.rfft(ts_hat, dim=-1) - torch.fft.rfft(ts, dim=-1)))
    return temporal_loss, freq_loss


def compute_loss(
    model,
    ts: torch.Tensor,
    text_features: torch.Tensor,
    labels: torch.Tensor | None = None,
    targets: torch.Tensor | None = None,
    var_f: torch.Tensor | None = None,
    target_type: str = "by_target",
    train_type: str = "joint",
    alpha: float = 1.0,
    beta: float = 0.1,
    epoch: int = 0,
    lambda_rvc: float = 1.0,
    lambda_time: float = 0.5,
    lambda_freq: float = 0.5,
):
    """
    Return: loss, rvc_loss, temporal_loss, freq_loss.

    `alpha` is kept for compatibility with existing calls; reconstruction weights are
    controlled by lambda_time/lambda_freq.
    """
    if var_f is None:
        raise ValueError("EMEdit requires metadata features `var_f`.")

    # forward returns log_volume, ts_hat, mean, log_var for compatibility
    log_volume, ts_hat, _, _ = model(ts, text_features, var_f)
    if hasattr(model, "rvc") and hasattr(model.rvc, "loss"):
        loss_rvc = model.rvc.loss(log_volume)
    else:
        loss_rvc = rvc_loss_from_matrix(log_volume)

    temporal_loss, freq_loss = compute_reconstruction_loss_mine(ts, ts_hat)

    if train_type in {"clip", "align", "alignment", "stage1"}:
        loss = lambda_rvc * loss_rvc
        temporal_loss = temporal_loss.detach() * 0.0
        freq_loss = freq_loss.detach() * 0.0
    elif train_type in {"joint", "stage2"}:
        recon = lambda_time * temporal_loss + lambda_freq * freq_loss
        loss = lambda_rvc * loss_rvc + alpha * recon
    elif train_type in {"vae", "recon", "reconstruction"}:
        loss = lambda_time * temporal_loss + lambda_freq * freq_loss
        loss_rvc = loss_rvc.detach() * 0.0
    else:
        raise ValueError(f"Unknown train_type: {train_type}")

    return loss, loss_rvc, temporal_loss, freq_loss


def _unpack_batch(batch):
    """Compatible with the existing dataloader tuple."""
    if len(batch) == 6:
        idx, ts, text_features, labels, var_f, targets = batch
        return idx, ts, text_features, labels, var_f, targets
    if len(batch) == 3:
        ts, text_features, var_f = batch
        return None, ts, text_features, None, var_f, None
    raise ValueError(f"Unsupported batch format of length {len(batch)}")


def train_epoch(model, train_dataloader, optimizer, target_type="by_target", train_type="joint", alpha=1.0, beta=1.0):
    model.train()
    total_loss = total_rvc = total_temp = total_freq = 0.0
    num_batches = 0

    for batch in train_dataloader:
        _, ts, text_features, labels, var_f, targets = _unpack_batch(batch)
        loss, loss_rvc, temporal_loss, freq_loss = compute_loss(
            model, ts, text_features, labels, targets, var_f, target_type, train_type, alpha, beta
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())
        total_rvc += float(loss_rvc.item())
        total_temp += float(temporal_loss.item())
        total_freq += float(freq_loss.item())
        num_batches += 1

    denom = max(num_batches, 1)
    return total_loss / denom, total_rvc / denom, total_temp / denom, total_freq / denom


@torch.no_grad()
def test_epoch(model, test_dataloader, target_type="by_target", train_type="joint", alpha=1.0, beta=0.1):
    model.eval()
    total_loss = total_rvc = total_temp = total_freq = 0.0
    num_batches = 0

    for batch in test_dataloader:
        _, ts, text_features, labels, var_f, targets = _unpack_batch(batch)
        loss, loss_rvc, temporal_loss, freq_loss = compute_loss(
            model, ts, text_features, labels, targets, var_f, target_type, train_type, alpha, beta
        )
        total_loss += float(loss.item())
        total_rvc += float(loss_rvc.item())
        total_temp += float(temporal_loss.item())
        total_freq += float(freq_loss.item())
        num_batches += 1

    denom = max(num_batches, 1)
    return total_loss / denom, total_rvc / denom, total_temp / denom, total_freq / denom


def train_emedit(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    scheduler,
    num_epochs,
    target_type="by_target",
    train_type="joint",
    alpha_init=None,
    beta=0.0,
    es_patience=200,
    target_ratio=100,
    output_dir="",
):
    """Backward-compatible trainer name used by the existing project."""
    torch.manual_seed(333)
    random.seed(333)
    np.random.seed(333)

    os.makedirs(output_dir, exist_ok=True) if output_dir else None
    alpha = 1.0 if alpha_init is None else alpha_init
    best_test_loss = float("inf")
    best_model_state = None
    counter = 0

    train_losses, test_losses = [], []
    train_rvc_losses, test_rvc_losses = [], []
    train_temp_losses, test_temp_losses = [], []
    train_freq_losses, test_freq_losses = [], []

    print(f"\t----------------------- EMEdit train_type={train_type} -----------------------")
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss, train_rvc, train_temp, train_freq = train_epoch(
            model, train_dataloader, optimizer, target_type, train_type, alpha, beta
        )
        test_loss, test_rvc, test_temp, test_freq = test_epoch(
            model, test_dataloader, target_type, train_type, alpha, beta
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_rvc_losses.append(train_rvc)
        test_rvc_losses.append(test_rvc)
        train_temp_losses.append(train_temp)
        test_temp_losses.append(test_temp)
        train_freq_losses.append(train_freq)
        test_freq_losses.append(test_freq)

        if scheduler is not None:
            scheduler.step(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = copy.deepcopy(model.state_dict())
            ckpt_name = "model_clip.pth" if train_type in {"clip", "align", "stage1"} else "model.pth"
            if output_dir:
                torch.save(model.state_dict(), os.path.join(output_dir, ckpt_name))
            print(f"\tBest {ckpt_name} saved at epoch {epoch}; val={test_loss:.6f}")
            counter = 0
        else:
            counter += 1
            if counter >= es_patience:
                print(f"\nEarly stopping at epoch {epoch + 1}; no improvement for {es_patience} epochs.")
                break

        current_lr = optimizer.param_groups[0]["lr"]
        min_lrs = getattr(scheduler, "min_lrs", [0.0]) if scheduler is not None else [0.0]
        if scheduler is not None and current_lr <= min_lrs[0]:
            print("Learning rate too small. Stopping training.")
            break

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch + 1:04d}/{num_epochs} | "
            f"train={train_loss:.6f} val={test_loss:.6f} | "
            f"RVC={train_rvc:.6f}/{test_rvc:.6f} | "
            f"Temp={train_temp:.6f}/{test_temp:.6f} | "
            f"Freq={train_freq:.6f}/{test_freq:.6f} | "
            f"lr={current_lr:.2e} | {elapsed:.1f}s"
        )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, test_losses, alpha


def recalibrate_alpha(clip_loss, reconstruction_loss, target_ratio=10):
    if reconstruction_loss == 0:
        return 1.0
    return target_ratio * clip_loss / reconstruction_loss
