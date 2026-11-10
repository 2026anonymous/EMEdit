import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import get_config
from data import EMEditDataset, collate_emedit
from EMEdit import EMEdit
from surrogates import fit_surrogates
from text_encoder import FrozenQwenTextEncoder
from train import train_emedit


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="air")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--validation-jsonl", required=True)
    parser.add_argument("--output-dir", default="results/emedit_paper")
    parser.add_argument("--text-model", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--extract-metadata", action="store_true")
    parser.add_argument("--metadata-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--runs", type=int, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.device:
        overrides["device"] = args.device
    if args.runs is not None:
        overrides["num_runs"] = args.runs
    config = get_config(args.dataset, **overrides)
    device = config.device

    text_encoder = FrozenQwenTextEncoder(args.text_model, device)
    contract_extractor = None
    if args.extract_metadata:
        from metadata import Qwen3ContractExtractor
        contract_extractor = Qwen3ContractExtractor(args.metadata_model)

    train_set = EMEditDataset.from_jsonl(
        args.train_jsonl,
        text_encoder,
        config.seq_length,
        contract_extractor=contract_extractor,
    )
    validation_set = EMEditDataset.from_jsonl(
        args.validation_jsonl,
        text_encoder,
        config.seq_length,
        contract_extractor=contract_extractor,
    )

    num_event_classes = max(
        max(int(row["observed_event_label"].item()), int(row["target_event_label"].item()))
        for row in train_set.rows
    ) + 1
    num_metadata_classes = max(
        int(row["metadata_label"].item()) for row in train_set.rows
    ) + 1

    results = []
    for run_index in range(config.num_runs):
        seed = config.random_state + run_index
        set_seed(seed)
        generator = torch.Generator()
        generator.manual_seed(seed)

        train_loader = DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_emedit,
            generator=generator,
        )
        validation_loader = DataLoader(
            validation_set,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_emedit,
        )

        surrogates = fit_surrogates(
            train_loader,
            config.seq_length,
            num_event_classes,
            num_metadata_classes,
            device,
            config.surrogate_epochs,
            config.surrogate_learning_rate,
        )

        model = EMEdit(
            seq_length=config.seq_length,
            text_dim=text_encoder.embedding_dimension,
            hidden_dim=config.hidden_dim,
            branch_kernels=config.branch_kernels,
            branch_dim=config.branch_dim,
            num_res_blocks=config.num_res_blocks,
            decoder_layers=config.decoder_layers,
            decoder_heads=config.decoder_heads,
            dropout=config.dropout, 
            rvc_eps=config.rvc_eps,
            rvc_tau0=config.rvc_tau0,
            rvc_tau_min=config.rvc_tau_min,
            rvc_tau_max=config.rvc_tau_max,
            rvc_top_k=config.rvc_top_k,
            rvc_eps_s=config.rvc_eps_s,
        )

        run_dir = Path(args.output_dir) / f"run_{run_index + 1}"
        result = train_emedit(
            model,
            train_loader,
            validation_loader,
            surrogates,
            config,
            run_dir,
        )
        result["seed"] = seed
        results.append(result)
        print(result)


if __name__ == "__main__":
    main()
