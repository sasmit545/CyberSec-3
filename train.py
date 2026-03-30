"""
train.py  --  Entry point for training & evaluating the SQLi detector.
===========================================================================
Usage:
    python train.py --data_dir ./data --epochs 10 --batch_size 64

Place these four files inside --data_dir before running:
    trainingdata.csv          (training set T)
    testingdata.csv           (test set S  -- queries < 1000 chars)
    testinglongdata_500.csv   (test set U  -- 500 long queries)
    testinglongdatav2.csv     (optional extra long-query set, use --include_u2)

Download from: https://www.kaggle.com/datasets/rayten/sql-injection-dataset
"""

import argparse
import os
import sys
import json
import torch

from tokenizer import SQLTokenizer
from dataset   import build_dataloaders
from model     import SQLiDetector
from trainer   import Trainer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train the lightweight SQL injection detector (Lo et al., 2025)"
    )

    # Data
    p.add_argument("--data_dir",    type=str, default="./data",
                   help="Folder containing the four Kaggle CSV files")
    p.add_argument("--include_u2",  action="store_true",
                   help="Also evaluate on testinglongdatav2.csv")

    # Model hyperparameters (paper defaults)
    p.add_argument("--max_len",     type=int,   default=512)
    p.add_argument("--m1",          type=int,   default=10,
                   help="Token embedding dim (M1, default 10)")
    p.add_argument("--m2",          type=int,   default=1,
                   help="Semantic label embedding dim (M2, default 1)")
    p.add_argument("--cnn_mid",     type=int,   default=64)
    p.add_argument("--cnn_out",     type=int,   default=128)
    p.add_argument("--num_heads",   type=int,   default=4)
    p.add_argument("--threshold",   type=float, default=0.5)

    # Training
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--num_workers", type=int,   default=0)
    p.add_argument("--device",      type=str,   default=None,
                   help="'cuda', 'cpu', or 'mps'. Auto-detected if omitted.")

    # Output
    p.add_argument("--save_dir",    type=str,   default="./checkpoints")

    # Eval-only mode
    p.add_argument("--eval_only",   action="store_true")
    p.add_argument("--checkpoint",  type=str,   default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # -- Tokenizer -----------------------------------------------------------
    print("\n[Setup] Building SQL-specific tokenizer …")
    tokenizer = SQLTokenizer(max_len=args.max_len)
    print(f"  Vocabulary size : {tokenizer.vocab_size}")

    # -- Data ----------------------------------------------------------------
    print("\n[Setup] Loading dataset …")
    if not os.path.isdir(args.data_dir):
        print(f"[ERROR] data_dir '{args.data_dir}' does not exist.")
        print("  Create the folder and place these files inside it:")
        print("    trainingdata.csv")
        print("    testingdata.csv")
        print("    testinglongdata_500.csv")
        print("    testinglongdatav2.csv   (optional)")
        sys.exit(1)

    result = build_dataloaders(
        data_dir    = args.data_dir,
        tokenizer   = tokenizer,
        max_len     = args.max_len,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        include_u2  = args.include_u2,
    )

    if args.include_u2:
        train_loader, test_s_loader, test_u_loader, test_u2_loader, label_counts = result
    else:
        train_loader, test_s_loader, test_u_loader, label_counts = result
        test_u2_loader = None

    # -- Model ---------------------------------------------------------------
    print("\n[Setup] Building model …")
    model = SQLiDetector(
        max_len   = args.max_len,
        m1        = args.m1,
        m2        = args.m2,
        cnn_mid   = args.cnn_mid,
        cnn_out   = args.cnn_out,
        num_heads = args.num_heads,
        threshold = args.threshold,
    )
    print(f"  Total parameters: {model.count_parameters():,}")

    # -- Eval-only mode ------------------------------------------------------
    if args.eval_only:
        if args.checkpoint is None:
            args.checkpoint = os.path.join(args.save_dir, "best_model.pt")
        print(f"\n[Eval-only] Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        trainer = Trainer(model, train_loader, test_s_loader, test_u_loader,
                          save_dir=args.save_dir, device=args.device)
        print("\n--- Test Set S (testingdata.csv) ---")
        print(json.dumps(trainer._evaluate(test_s_loader, "Test-S"), indent=2))
        print("\n--- Test Set U (testinglongdata_500.csv) ---")
        print(json.dumps(trainer._evaluate(test_u_loader, "Test-U"), indent=2))
        if test_u2_loader:
            print("\n--- Test Set U2 (testinglongdatav2.csv) ---")
            print(json.dumps(trainer._evaluate(test_u2_loader, "Test-U2"), indent=2))
        return

    # -- Train ---------------------------------------------------------------
    trainer = Trainer(
        model         = model,
        train_loader  = train_loader,
        test_s_loader = test_s_loader,
        test_u_loader = test_u_loader,
        lr            = args.lr,
        epochs        = args.epochs,
        save_dir      = args.save_dir,
        device        = args.device,
    )
    history = trainer.train()

    # Optionally evaluate on U2 after training
    if test_u2_loader:
        print("\n--- Test Set U2 (testinglongdatav2.csv) ---")
        print(json.dumps(trainer._evaluate(test_u2_loader, "Test-U2"), indent=2))

    # -- Final summary -------------------------------------------------------
    best = max(history, key=lambda r: r["test_s"]["f1"])
    print("\n" + "="*60)
    print(" BEST EPOCH RESULTS (Test-S  /  testingdata.csv)")
    print("="*60)
    print(f"  Epoch     : {best['epoch']}")
    print(f"  Accuracy  : {best['test_s']['accuracy']:.2f}%")
    print(f"  Precision : {best['test_s']['precision']:.2f}%")
    print(f"  Recall    : {best['test_s']['recall']:.2f}%")
    print(f"  F1 Score  : {best['test_s']['f1']:.2f}%")
    print(f"  Inference : {best['test_s']['avg_inference_ms']:.2f} ms/query")
    print("\n BEST EPOCH RESULTS (Test-U  /  testinglongdata_500.csv)")
    print(f"  Accuracy  : {best['test_u']['accuracy']:.2f}%")
    print(f"  F1 Score  : {best['test_u']['f1']:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()