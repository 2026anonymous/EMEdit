## Files

- `EMEdit.py`: multi-resolution temporal encoder, shared text projection, event-conditioned metadata routing, source-anchored asymmetric dual-path fusion, RVC, gated residual editing and Transformer decoder.
- `train.py`: two-stage training, factual temporal-frequency reconstruction, target-event consistency and source-metadata consistency.
- `metadata.py`: schema-role validation, evidence-pointer verification and source metadata serialization.
- `text_encoder.py`: frozen Qwen3-Embedding-4B feature extraction.
- `surrogates.py`: separate temporal and frequency surrogate classifiers for event and metadata labels.
- `data.py`: JSONL loading, text feature extraction, stable-context padding and duplicate-negative masking.
- `eval.py`: RaTS, |RaTS|, ΔDTW, MSE and MAE.
- `config.py`: paper-reported settings and dataset sequence lengths.
- `main_paper.py`: standalone training entry point.

## JSONL record format

Each line is one independent editing instance:

```json
{
  "ts": [0.1, 0.2, 0.3],
  "observed_event_text": "observed source event",
  "target_event_text": "target counterfactual event",
  "metadata_text": "verified source identity and stable context",
  "stable_context": [
    {"key": "market", "value": "US"}
  ],
  "observed_event_label": 0,
  "target_event_label": 1,
  "metadata_label": 0,
  "source_contract_id": "source-001",
  "event_condition_id": "event-001"
}
```

## Training

```bash
pip install -r requirements.txt
python main_paper.py --dataset air --train-jsonl data/train.jsonl --validation-jsonl data/validation.jsonl
```

For schema-guided metadata extraction from raw source records:

```bash
python main_paper.py --dataset air --train-jsonl data/train.jsonl --validation-jsonl data/validation.jsonl --extract-metadata
```

The model expects Qwen3-Embedding-4B text features. The pretrained text model is frozen; the projection layer inside EMEdit is trainable.

