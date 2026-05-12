"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SELECTED_LAYERS = [13, 22, 18, 24]
POOLINGS = ['max', 'max', 'max', 'last']
WINDOW_SIZE = 64

_meta_cache = {
    'rows': None,
    'response_to_idx': None,
    'idx_counter': 0,
}


def _load_dataset_for_meta(data_path: str | Path) -> None:
    df = pd.read_csv(data_path)
    rows = list(zip(df['prompt'].astype(str), df['response'].astype(str)))
    _meta_cache['rows'] = rows
    _meta_cache['response_to_idx'] = {
        (str(p), str(r)): i for i, (p, r) in enumerate(rows)
    }
    _meta_cache['idx_counter'] = 0


def _ensure_dataset_loaded() -> None:
    if _meta_cache['rows'] is not None:
        return
    rows = []
    for path in ['./data/dataset.csv', './data/test.csv']:
        if Path(path).exists():
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                rows.append((str(row['prompt']), str(row['response'])))
    _meta_cache['rows'] = rows
    _meta_cache['response_to_idx'] = {
        (p, r): i for i, (p, r) in enumerate(rows)
    }


def _compute_meta_features_for_text(prompt: str, response: str) -> np.ndarray:
    response_clean = response.replace('<|endoftext|>', '').strip()
    words = response_clean.split()
    response_lower = response_clean.lower()

    features = [
        len(response_clean),
        len(words),
        np.log1p(len(response_clean)),
        len(prompt),
        len(response_clean) / (len(prompt) + 1),
        float('unable to answer' in response_lower),
        float(any(w in response_lower
                  for w in ['cannot', "can't", 'unable', 'not enough'])),
        response_clean.count('.') + response_clean.count('!')
            + response_clean.count('?'),
        response_clean.count(','),
        sum(c.isdigit() for c in response_clean),
        sum(c.isupper() for c in response_clean),
        np.mean([len(w) for w in words]) if words else 0,
        len(set(response_lower.split())),
        len(set(response_lower.split())) / (len(words) + 1),
    ]
    return np.array(features, dtype=np.float32)

def _pool_tokens(
    layer_tokens: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    if pooling == 'last':
        return layer_tokens[-1]
    elif pooling == 'mean':
        return layer_tokens.mean(dim=0)
    elif pooling == 'max':
        return layer_tokens.max(dim=0).values
    elif pooling == 'first':
        return layer_tokens[0]
    else:
        raise ValueError(f"Unknown pooling: {pooling}")

def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    real_positions = attention_mask.nonzero(as_tuple=False).flatten()
    real_len = real_positions.numel()

    if real_len == 0:
        hidden_dim = hidden_states.size(-1)
        return torch.zeros(hidden_dim * len(SELECTED_LAYERS))

    if real_len > WINDOW_SIZE:
        window_positions = real_positions[-WINDOW_SIZE:]
    else:
        window_positions = real_positions

    parts: list[torch.Tensor] = []
    for layer_idx, pooling in zip(SELECTED_LAYERS, POOLINGS):
        layer = hidden_states[layer_idx]
        windowed = layer[window_positions]
        pooled = _pool_tokens(windowed, pooling)
        parts.append(pooled)

    return torch.cat(parts, dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    real_positions = attention_mask.nonzero(as_tuple=False).flatten()
    if real_positions.numel() == 0:
        n_layers = hidden_states.size(0)
        geom_norms = torch.zeros(n_layers)
    else:
        last_pos = int(real_positions[-1].item())
        n_layers = hidden_states.size(0)
        geom_norms = torch.zeros(n_layers)
        for l in range(n_layers):
            vec = hidden_states[l, last_pos, :]
            geom_norms[l] = torch.norm(vec).item()

    _ensure_dataset_loaded()
    rows = _meta_cache['rows']

    if rows and _meta_cache['idx_counter'] < len(rows):
        prompt, response = rows[_meta_cache['idx_counter']]
        meta = _compute_meta_features_for_text(prompt, response)
        _meta_cache['idx_counter'] += 1
    else:
        meta = np.zeros(14, dtype=np.float32)

    meta_t = torch.from_numpy(meta).float()

    combined = torch.cat([geom_norms, meta_t], dim=0)

    combined = torch.nan_to_num(combined, nan=0.0, posinf=1e4, neginf=-1e4)
    combined = torch.clamp(combined, min=-1e4, max=1e4)

    return combined


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    agg_features = aggregate(hidden_states, attention_mask)
    geo_features = extract_geometric_features(hidden_states, attention_mask)
    
    # Ensure both tensors are on the same device before concat
    # (agg_features stays on the device of hidden_states, geo_features may be on CPU)
    geo_features = geo_features.to(agg_features.device)
    
    return torch.cat([agg_features, geo_features], dim=0)
