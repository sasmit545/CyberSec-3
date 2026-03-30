# SQLi Detector — PyTorch Implementation

PyTorch implementation of **"SQL Injection Detection Based on Lightweight Multi-Head Self-Attention"** (Lo, Hwang & Tai, *Applied Sciences*, 2025).

---

## Architecture Summary

```
Input SQL Query
     │
     ▼
┌─────────────────── Embedding Section ───────────────────┐
│  SQL-Specific Tokenizer  (strip user-defined identifiers) │
│  Token Embedding   f1 : token_id  → R^M1  (M1=10)        │
│  Semantic Embedding f2: label_id  → R^M2  (M2=1)         │
│  Concatenate          : → R^11 per position  [F matrix]   │
│  Sinusoidal Pos. Enc. : F + C                [U matrix]   │
└──────────────────────────────────────────────────────────┘
     │  U ∈ R^(512 × 11)
     ▼
┌─────────────────── Detection Section ───────────────────┐
│  CNN Stage     : Conv1d(11→64) + MaxPool                  │
│                  Conv1d(64→128) + MaxPool                 │
│  Self-Attention: 4-head scaled dot-product attention      │
│                  + LayerNorm + residual                   │
│  Output Stage  : GlobalAvgPool → Dense → Sigmoid → z     │
└──────────────────────────────────────────────────────────┘
     │  z ∈ [0,1]
     ▼
z ≥ 0.5  →  MALICIOUS
z <  0.5  →  LEGAL
```

**Total parameters: ~69,269** (vs. 66M for DistilBERT)

---

## File Structure

```
sqli_detector/
├── tokenizer.py   — SQL-specific tokenizer (vocab=158, 3 semantic labels)
├── model.py       — Full neural network (embedding + CNN + attention + output)
├── dataset.py     — PyTorch Dataset + DataLoader builder
├── trainer.py     — Training loop, BCE loss, metrics, checkpointing
├── train.py       — Main entry point (CLI)
├── predict.py     — Inference on new queries (CLI + interactive REPL)
└── README.md      — This file
```

---

## Setup

### 1. Install dependencies

```bash
pip install torch scikit-learn pandas numpy tqdm
```

### 2. Download the dataset

Go to: https://www.kaggle.com/datasets/rayten/sql-injection-dataset

Download and place the CSV file(s) inside a `data/` directory:

```
sqli_detector/
└── data/
    └── trainingdata.csv
    testingdata.csv
    testinglongdata_500.csv
    testinglongdatav2.csv
```

**Or** provide pre-split files:

```
data/
├── train.csv     ← 98,275 queries
├── test_s.csv    ← 24,707 queries  (length < 1000 chars)
└── test_u.csv    ←    500 queries  (length ≥ 1000 chars)
```

Each CSV must have columns `Query` (str) and `Label` (int: 0=legal, 1=malicious).

---

## Training

```bash
# Default settings from the paper
python train.py --data_dir ./data --epochs 10 --batch_size 64

# All options
python train.py \
    --data_dir    ./data \
    --epochs      10 \
    --batch_size  64 \
    --lr          0.001 \
    --max_len     512 \
    --m1          10 \
    --m2          1 \
    --cnn_mid     64 \
    --cnn_out     128 \
    --num_heads   4 \
    --threshold   0.5 \
    --save_dir    ./checkpoints \
    --device      cuda           # or cpu / mps
```

Checkpoints are saved to `./checkpoints/`:
- `best_model.pt`  — highest F1 on Test-S
- `final_model.pt` — end of training
- `training_history.json` — per-epoch metrics

---

## Evaluation Only

```bash
python train.py --eval_only --checkpoint checkpoints/best_model.pt --data_dir ./data
```

---

## Inference on New Queries

```bash
# Single query
python predict.py \
    --checkpoint checkpoints/best_model.pt \
    --query "SELECT * FROM users WHERE id=1 OR '1'='1'"

# Batch from file (one query per line)
python predict.py \
    --checkpoint checkpoints/best_model.pt \
    --input_file queries.txt

# Interactive REPL
python predict.py \
    --checkpoint checkpoints/best_model.pt \
    --interactive
```

---

## Expected Results

Reproducing Table 4 / Table 5 from the paper:

| Split  | Accuracy | Precision | Recall | F1     |
|--------|----------|-----------|--------|--------|
| Test S | 98.98%   | 98.01%    | 99.84% | 98.92% |
| Test U | 96.60%   | 99.23%    | 96.50% | 97.85% |

Average inference time: **~51 ms/query** on a PC (CPU).

---

## Citation

```bibtex
@article{lo2025sqli,
  title   = {SQL Injection Detection Based on Lightweight Multi-Head Self-Attention},
  author  = {Lo, Rui-Teng and Hwang, Wen-Jyi and Tai, Tsung-Ming},
  journal = {Applied Sciences},
  volume  = {15},
  number  = {2},
  pages   = {571},
  year    = {2025},
  doi     = {10.3390/app15020571}
}
```