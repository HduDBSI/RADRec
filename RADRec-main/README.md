# Not All Sequences Need Augmentation: Retrieval-Augmented Diffusion with Contrastive Learning for Sequential Recommendation (RADRec)

PyTorch implementation of RADRec, packaged with the Beauty data and the
checkpoints needed to reproduce evaluation and rebuild the sequence partition.

## Package contents

```text
datasets/Beauty.txt             Beauty interaction sequences
pretrained/beauty-0.pt          checkpoint used for sequence partitioning
output/RADRec-Beauty.pt         supplied RADRec checkpoint
output/entropy_cache/Beauty_cache.npz
                                retained legacy partition cache
scripts/eval.sh                 location-independent evaluation script
tests/test_coupling_estimation.py
```

Generated segmented datasets, new entropy caches, and log files are intentionally
excluded and are recreated by the commands below. The retained `Beauty_cache.npz`
comes from a legacy partition configuration; the current configuration rebuilds
a compatible cache when needed.

## Requirements

- Python 3.9 or newer
- A PyTorch-compatible CPU or CUDA environment

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Build the Beauty partition cache

Run this once before training:

```bash
python main.py \
  --data_name Beauty \
  --model_idx 0 \
  --build_entropy_cache_only \
  --entropy_pretrained_path ./pretrained/beauty-0.pt
```

The generated cache is stored under `output/entropy_cache/`.

## Train RADRec

```bash
python main.py \
  --data_name Beauty \
  --model_idx 0 \
  --entropy_pretrained_path ./pretrained/beauty-0.pt
```

Use `--cuda N` to select a CUDA device when needed.

## Evaluate the supplied checkpoint

From the repository root:

```bash
python main.py \
  --data_name Beauty \
  --eval_only \
  --checkpoint_path ./output/RADRec-Beauty.pt
```

The helper script can be called from any directory and accepts additional
arguments, for example `--cuda 1`:

```bash
./scripts/eval.sh --cuda 1
```

## Tests

```bash
python -m unittest -v tests.test_coupling_estimation
```
