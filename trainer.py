"""
Trainer
=======
Implements the training loop described in Section 3.5 of the paper.

Loss function : Binary Cross-Entropy  (Equations 7–9)
Optimiser     : Adam  (standard choice for transformer-style models)
Metrics       : Accuracy, Precision, Recall, F1-score  (Equations 10–12)
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_metrics(targets: list, preds: list) -> dict:
    """
    Computes the four metrics from Equations 10-12 of the paper.
    Returns a dict with keys: accuracy, precision, recall, f1.
    """
    acc  = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, zero_division=0)
    rec  = recall_score(targets, preds, zero_division=0)
    f1   = f1_score(targets, preds, zero_division=0)
    return {
        "accuracy" : round(acc  * 100, 4),
        "precision": round(prec * 100, 4),
        "recall"   : round(rec  * 100, 4),
        "f1"       : round(f1   * 100, 4),
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Encapsulates one full training + evaluation run.

    Args:
        model         : SQLiDetector instance
        train_loader  : DataLoader for training set T
        test_s_loader : DataLoader for test set S  (len < 1000)
        test_u_loader : DataLoader for test set U  (len >= 1000)
        lr            : learning rate  (default 1e-3)
        epochs        : number of training epochs
        save_dir      : directory for checkpoints and results
        device        : 'cuda', 'cpu' or 'mps' (auto-detected if None)
    """

    def __init__(self,
                 model,
                 train_loader:  DataLoader,
                 test_s_loader: DataLoader,
                 test_u_loader: DataLoader,
                 lr:            float = 1e-3,
                 epochs:        int   = 10,
                 save_dir:      str   = "checkpoints",
                 device:        str   = None):

        # Device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        print(f"[Trainer] Using device: {self.device}")

        self.model         = model.to(self.device)
        self.train_loader  = train_loader
        self.test_s_loader = test_s_loader
        self.test_u_loader = test_u_loader
        self.epochs        = epochs
        self.save_dir      = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Loss: Binary Cross-Entropy (Equation 7)
        self.criterion = nn.BCELoss()

        # Optimiser
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Learning rate scheduler (reduce on plateau)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2
        )

        self.history = []   # list of per-epoch dicts
        self.best_f1 = 0.0

    # ------------------------------------------------------------------
    def _move_batch(self, batch):
        return (
            batch["token_ids"].to(self.device),
            batch["label_ids"].to(self.device),
            batch["target"].to(self.device),
        )

    # ------------------------------------------------------------------
    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(self.train_loader, desc="  Train", leave=False):
            tok, lab, tgt = self._move_batch(batch)
            self.optimizer.zero_grad()
            z    = self.model(tok, lab)           # (B,)
            loss = self.criterion(z, tgt)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * len(tgt)
        return total_loss / len(self.train_loader.dataset)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _evaluate(self, loader: DataLoader,
                  split_name: str) -> dict:
        """Runs inference over a DataLoader and returns metric dict."""
        self.model.eval()
        all_targets, all_preds = [], []
        total_time = 0.0
        n_queries  = 0

        for batch in tqdm(loader, desc=f"  Eval [{split_name}]", leave=False):
            tok, lab, tgt = self._move_batch(batch)
            t0 = time.perf_counter()
            z  = self.model(tok, lab)
            total_time += time.perf_counter() - t0

            preds = (z >= self.model.threshold).long().cpu().tolist()
            all_preds   += preds
            all_targets += tgt.long().cpu().tolist()
            n_queries   += len(tgt)

        metrics = compute_metrics(all_targets, all_preds)
        # Average inference time per query (ms) — matches Table 6 metric
        metrics["avg_inference_ms"] = round(
            (total_time / n_queries) * 1000, 4
        )
        return metrics

    # ------------------------------------------------------------------
    def train(self):
        """Full training loop with checkpointing."""
        print(f"\n{'='*60}")
        print(f" Training for {self.epochs} epoch(s)")
        print(f" Model parameters : {self.model.count_parameters():,}")
        print(f"{'='*60}\n")

        for epoch in range(1, self.epochs + 1):
            t_start = time.time()
            train_loss = self._train_epoch()
            metrics_s  = self._evaluate(self.test_s_loader, "Test-S")
            metrics_u  = self._evaluate(self.test_u_loader, "Test-U")

            elapsed = time.time() - t_start

            # Step LR scheduler on F1 of Test-S
            self.scheduler.step(metrics_s["f1"])

            # Log
            record = {
                "epoch"     : epoch,
                "train_loss": round(train_loss, 6),
                "test_s"    : metrics_s,
                "test_u"    : metrics_u,
                "elapsed_s" : round(elapsed, 2),
            }
            self.history.append(record)
            self._print_epoch(record)

            # Save best checkpoint based on F1 on Test-S
            if metrics_s["f1"] > self.best_f1:
                self.best_f1 = metrics_s["f1"]
                self._save_checkpoint("best_model.pt")

        # Always save final model
        self._save_checkpoint("final_model.pt")

        # Persist history
        hist_path = os.path.join(self.save_dir, "training_history.json")
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\n[Trainer] History saved to {hist_path}")

        return self.history

    # ------------------------------------------------------------------
    def _save_checkpoint(self, filename: str):
        path = os.path.join(self.save_dir, filename)
        torch.save({
            "model_state_dict"    : self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_f1"             : self.best_f1,
            "history"             : self.history,
        }, path)
        print(f"  [✓] Checkpoint saved → {path}")

    # ------------------------------------------------------------------
    @staticmethod
    def _print_epoch(r: dict):
        s = r["test_s"]
        u = r["test_u"]
        print(
            f"Epoch {r['epoch']:>3} | "
            f"Loss {r['train_loss']:.4f} | "
            f"Test-S  Acc={s['accuracy']:.2f}%  "
            f"P={s['precision']:.2f}%  "
            f"R={s['recall']:.2f}%  "
            f"F1={s['f1']:.2f}%  "
            f"({s['avg_inference_ms']:.2f}ms/q) | "
            f"Test-U  Acc={u['accuracy']:.2f}%  "
            f"F1={u['f1']:.2f}% | "
            f"{r['elapsed_s']}s"
        )
