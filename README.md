# EMEdit

**Event-Meta Counterfactual Time Series Editing via Riemannian Volume Contrast**

EMEdit is a language-guided framework for counterfactual time series editing. Given a source time series, an event description, and a metadata description, EMEdit generates an edited sequence that reflects the target event while preserving the intrinsic identity of the original series.

The key idea is **asymmetric event-meta conditioning**:

- **Event description**: specifies what should be changed.
- **Metadata description**: specifies what should be preserved.

EMEdit further introduces **Riemannian Volume Contrast (RVC)** to align time-series, event, and metadata representations on the SPD manifold, followed by metadata-guided gated fusion and a Transformer-based editing decoder.

---

## Overview

Given an input time series `X_ts`, an event description `X_e`, and a metadata description `X_m`, EMEdit performs:

```text
X_hat_ts = D(H(E_ts(X_ts), E_text(X_e), E_text(X_m)))
```

where:

- `E_ts` is a multi-resolution time-series encoder,
- `E_text` is a shared text encoder for event and metadata prompts,
- `H` is a metadata-guided gated fusion module,
- `D` is a Transformer-based editing decoder.

During inference, an editing-strength parameter `w in [0, 1]` controls the strength of the event intervention. Smaller `w` produces more conservative edits, while larger `w` increases the influence of the target event.

---

## Main Features

- Language-guided counterfactual editing for time series.
- Asymmetric event-meta conditioning to separate editability and preservability.
- Riemannian Volume Contrast (RVC) for geometry-aware multimodal alignment.
- SPD Gram matrix construction over time-series, event, and metadata embeddings.
- Metadata-guided gated fusion to preserve source-specific identity.
- Transformer editing decoder for generating edited sequences.
- Hybrid temporal-frequency reconstruction objective.
- Two-stage training:
  1. Feature alignment with RVC.
  2. Joint optimization for alignment and event-meta conditioned editing.

---

## Repository Structure

```text
EMEdit/
├── EMEdit.py              # EMEdit model implementation
├── train.py               # Training utilities and RVC/reconstruction losses
├── data.py                # Dataset loading, feature extraction, and dataloaders
├── eval.py                # Evaluation entry utilities
├── generation.py          # Generation-related utilities
├── config.py              # Default configuration
├── main.py                # Main experiment entry point
├── requirements.txt       # Python dependencies
└── script/
    ├── run/               # Dataset-specific running scripts
    ├── data/              # Dataset files or placeholders
    ├── data_utils/        # Data preprocessing utilities
    ├── eval_utils/        # Evaluation utilities
    └── model_utils/       # Encoder/decoder utility modules
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/2026anonymous/EMEdit.git
cd EMEdit
```

Create a new environment:

```bash
conda create -n emedit python=3.10
conda activate emedit
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If the full `requirements.txt` contains machine-specific paths or fails on your platform, install the main dependencies manually:

```bash
pip install torch torchvision torchaudio
pip install numpy pandas scipy scikit-learn matplotlib tqdm
pip install transformers sentence-transformers timm torchinfo tslearn ruptures
```

---

## Data Format

The default data loader expects a table-like dataset, where each row corresponds to one sample. A typical CSV file should contain:

| Column | Description |
| --- | --- |
| `1`, `2`, ..., `L` | Time-series values of length `L` |
| `text` | Event description, i.e., what should be changed |
| `var` | Metadata description, i.e., what should be preserved |
| `label` | Attribute or condition label for training/evaluation |

Example:

```text
1,2,3,...,300,text,var,label
0.12,0.15,0.18,...,0.31,"sudden policy tightening shock","Nasdaq trading volume",negative_macro
```

The sequence length is controlled by `seq_length` in `config.py`.

---

## Training

Run EMEdit with the default setting:

```bash
python main.py --dataset_name air
```

Run with open-ended event descriptions:

```bash
python main.py --dataset_name air --open_vocab
```

Common arguments:

```bash
python main.py \
  --dataset_name air \
  --attr_suffix "" \
  --suffix "" \
  --gamma 10
```

The default training configuration includes:

- optimizer: Adam
- initial learning rate: `1e-4`
- batch size: `512`
- hidden dimension: `768`
- early stopping patience: `200`
- text encoder: `Qwen/Qwen3-Embedding-4B`

---

