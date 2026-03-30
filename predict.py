"""
predict.py  —  Run inference on new SQL queries using a trained checkpoint.
===========================================================================
Usage:
    # Single query on command line
    python predict.py --checkpoint checkpoints/best_model.pt \
                      --query "SELECT * FROM users WHERE id=1 OR '1'='1'"

    # Batch from a text file (one query per line)
    python predict.py --checkpoint checkpoints/best_model.pt \
                      --input_file queries.txt

    # Interactive REPL
    python predict.py --checkpoint checkpoints/best_model.pt --interactive
"""

import argparse
import time
import torch
from tokenizer import SQLTokenizer
from model     import SQLiDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str = "cpu") -> SQLiDetector:
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model = SQLiDetector()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)
    return model


def predict_query(query: str, model: SQLiDetector,
                  tokenizer: SQLTokenizer, device: str) -> dict:
    """Returns prediction dict for a single SQL query string."""
    token_ids, label_ids = tokenizer.tokenize(query)
    tok = torch.tensor([token_ids], dtype=torch.long).to(device)
    lab = torch.tensor([label_ids], dtype=torch.long).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        z = model(tok, lab).item()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    skeleton = tokenizer.decode_skeleton(query)
    return {
        "query"        : query,
        "skeleton"     : skeleton,
        "score"        : round(z, 6),
        "verdict"      : "MALICIOUS" if z >= model.threshold else "LEGAL",
        "inference_ms" : round(elapsed_ms, 3),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SQL injection inference")
    p.add_argument("--checkpoint",   type=str, required=True)
    p.add_argument("--query",        type=str, default=None,
                   help="Single SQL query string to classify")
    p.add_argument("--input_file",   type=str, default=None,
                   help="Text file with one query per line")
    p.add_argument("--interactive",  action="store_true",
                   help="Launch interactive REPL")
    p.add_argument("--device",       type=str, default="cpu")
    p.add_argument("--max_len",      type=int, default=512)
    return p.parse_args()


def main():
    args      = parse_args()
    tokenizer = SQLTokenizer(max_len=args.max_len)
    model     = load_model(args.checkpoint, args.device)
    print(f"[Predict] Model loaded from {args.checkpoint}")
    print(f"          Parameters: {model.count_parameters():,}")

    def show(result):
        print(f"\n  Query    : {result['query'][:120]}")
        print(f"  Skeleton : {result['skeleton'][:120]}")
        print(f"  Score    : {result['score']:.4f}")
        print(f"  Verdict  : {result['verdict']}")
        print(f"  Time     : {result['inference_ms']} ms")

    if args.query:
        show(predict_query(args.query, model, tokenizer, args.device))

    elif args.input_file:
        with open(args.input_file) as f:
            queries = [l.strip() for l in f if l.strip()]
        print(f"\n[Predict] Classifying {len(queries)} queries …")
        for q in queries:
            show(predict_query(q, model, tokenizer, args.device))

    elif args.interactive:
        print("\n[Interactive mode] Type a SQL query to classify. Ctrl-C to quit.\n")
        while True:
            try:
                q = input("SQL> ").strip()
                if not q:
                    continue
                show(predict_query(q, model, tokenizer, args.device))
            except KeyboardInterrupt:
                print("\nBye.")
                break
    else:
        # Run a few built-in demo queries
        demos = [
            "SELECT * FROM users WHERE username = 'admin' AND password = 'secret'",
            "SELECT * FROM users WHERE username = 'admin' OR '1'='1' --",
            "SELECT customer_name FROM customer_table WHERE customer_level = 3",
            "SELECT * FROM products; DROP TABLE products; --",
            "INSERT INTO orders (product_id, qty) VALUES (42, 1)",
        ]
        print("\n[Demo] Running built-in example queries:\n")
        for q in demos:
            show(predict_query(q, model, tokenizer, args.device))


if __name__ == "__main__":
    main()
