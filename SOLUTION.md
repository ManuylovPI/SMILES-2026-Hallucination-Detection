# Solution Report

## Result

5-fold cross-validation on `data/dataset.csv`:

- Test accuracy: **74.31%**
- Test F1: **83.43%**
- Test AUROC: **0.7577**
- Majority-class baseline accuracy: 70.10%

Primary metric (test AUROC): **0.7577**, improvement over baseline: +4.2pp accuracy.

## How to reproduce

```bash
git clone https://github.com/ManuylovPI/SMILES-2026-Hallucination-Detection.git
cd SMILES-2026-Hallucination-Detection
pip install -r requirements.txt
pip install xgboost
python solution.py
```

Generates `results.json` and `predictions.csv`. Takes ~3-5 min on a T4 GPU.

Note: `aggregation.py` reads `dataset.csv` and `test.csv` directly to compute meta features, relying on `solution.py` processing samples in dataset order (train, then test). If that order changes, meta features become misaligned.

## Approach

**Aggregation.** Concatenation of 4 layers from Qwen2.5-0.5B with mixed pooling:
- Layer 13 — max pooling
- Layer 22 — max pooling
- Layer 18 — max pooling
- Layer 24 — last-token pooling

Pooling is restricted to the last 64 real tokens (where the response lives). Total hidden-state features: 3584 dims.

**Extra features.** Two small blocks added:
- 25 dims — L2 norm of last-token vector at every layer
- 14 dims — text features computed from prompt+response (length, sentence count, punctuation, refusal markers like "unable to answer", lexical diversity)

**Probe.** Soft-voting ensemble of:
- Logistic Regression (C=0.01, class_weight='balanced')
- XGBoost (200 trees, depth 4, lr=0.05, scale_pos_weight matched to class imbalance)

Both are trained on the same features plus 4 centroid-distance features (L2 + cosine to truthful/hallucinated centroids on layer 24, computed inside `fit()` to avoid leakage). Final probability = average of both classifiers.

**Splitting.** Stratified 5-fold (instead of single train/val/test). Each fold: ~448 train, 103 val, 138 test.

## What worked, in order of contribution

1. **Multi-layer concat instead of last layer only** (~+3pp accuracy). Layer 13 was the strongest single layer in the sweep — surprising for a "truthfulness" task, since literature usually points at late layers.
2. **Strong L2 regularisation** (C=0.01, two orders of magnitude below sklearn default). With 689 samples and 3.6k features, default C overfits badly.
3. **Soft voting LogReg + XGBoost** (~+0.5pp). LogReg had best AUROC, XGBoost had best accuracy — averaging probabilities captured both.
4. **Meta features** (~+0.3pp). EDA showed hallucinated responses are ~2x longer than truthful ones — a strong signal that hidden states already partly encode, but adding it explicitly helped slightly.
5. **Geometric features** and **centroid distances** (~+0.2pp combined). Small but non-negative contribution.

## What didn't work

- **Longer token window (96 vs 64 tokens).** Lost 1.6pp accuracy. Wider window included shared prompt content that diluted max pooling.
- **PCA (32-512 components).** Best PCA matched no-PCA on accuracy but lost AUROC. Signal is distributed across many directions.
- **MLP probe (1 or 2 hidden layers).** Overfit. With 689 examples linear probe is hard to beat.
- **Stacking with meta-classifier.** Added complexity, no improvement over simple averaging on the small training fold.
- **Removing class_weight='balanced'.** Boosted accuracy but tanked AUROC — classifier just biased toward majority class.
- **Wider ensembles (6+ layers concatenated).** Overfit even with strong L2.
- **Inter-layer drift (cosine similarity between adjacent layers).** Almost no class difference (ratio 1.04-1.07). Kept the norms as weak features but drift was dropped.

## Caveat

The 100% train accuracy reflects XGBoost fitting the training fold exactly — not generalization. The honest number is the 5-fold test accuracy of 74.31%. With more compute time several directions would likely help further: logit-based features from the LLM (`output_scores=True`), Optuna over hyperparameters, LightGBM/CatBoost alongside XGBoost.
