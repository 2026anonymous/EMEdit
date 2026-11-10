from dataclasses import dataclass, replace
from typing import Tuple


DATASET_SPECS = {
    "synthetic": {"seq_length": 200, "samples": 18000},
    "air": {"seq_length": 168, "samples": 3684},
    "air_quality": {"seq_length": 168, "samples": 3684},
    "nicu": {"seq_length": 300, "samples": 36679},
    "news": {"seq_length": 144, "samples": 5826},
    "news_popularity": {"seq_length": 144, "samples": 5826},
    "indices": {"seq_length": 60, "samples": 2113},
    "global_indices": {"seq_length": 60, "samples": 2113},
}


@dataclass(frozen=True)
class EMEditConfig:
    dataset_name: str = "air"
    seq_length: int = 168
    text_dim: int = 2560
    hidden_dim: int = 768
    batch_size: int = 512
    learning_rate: float = 1e-4
    stage1_epochs: int = 500
    stage2_epochs: int = 2500
    early_stopping_patience: int = 200
    scheduler_patience: int = 500
    scheduler_factor: float = 0.9
    scheduler_min_lr: float = 1e-6
    grad_clip: float = 1.0
    branch_kernels: Tuple[int, ...] = (3, 5, 9, 15)
    branch_dim: int = 192
    num_res_blocks: int = 2
    decoder_layers: int = 4
    decoder_heads: int = 8
    dropout: float = 0.1
    rvc_eps: float = 1e-4
    rvc_tau0: float = 0.07
    rvc_tau_min: float = 0.01
    rvc_tau_max: float = 1.0
    rvc_top_k: int = 60
    rvc_eps_s: float = 1e-8
    lambda_event: float = 1.0
    lambda_meta: float = 1.0
    lambda_recon: float = 1.0
    lambda_time: float = 1.0
    lambda_freq: float = 1.0
    surrogate_epochs: int = 100
    surrogate_learning_rate: float = 1e-3
    random_state: int = 333
    num_runs: int = 3
    device: str = "cuda"


def get_config(dataset_name: str = "air", **overrides) -> EMEditConfig:
    key = dataset_name.lower().replace(" ", "_")
    if key not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    config = EMEditConfig(
        dataset_name=key,
        seq_length=DATASET_SPECS[key]["seq_length"],
    )
    if "device" not in overrides:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            config = replace(config, device=device)
        except Exception:
            config = replace(config, device="cpu")
    return replace(config, **overrides)
