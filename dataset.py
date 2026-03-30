"""
Dataset
=======
Loads the Kaggle SQL injection dataset (Section 4.1 of paper).

Dataset URL: https://www.kaggle.com/datasets/rayten/sql-injection-dataset

Expected files inside --data_dir:
    trainingdata.csv          -> Training set T  (98,275 queries)
    testingdata.csv           -> Test set S      (24,707 queries, length < 1000)
    testinglongdata_500.csv   -> Test set U      (500 queries,   length >= 1000)
    testinglongdatav2.csv     -> Extra long-query test set (optional)

Each CSV must have columns:  Query (str)  and  Label (int: 0=legal, 1=malicious)
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import SQLTokenizer


# ---------------------------------------------------------------------------
# Exact filenames from the Kaggle dataset
# ---------------------------------------------------------------------------

FILE_TRAIN   = "trainingdata.csv"
FILE_TEST_S  = "testingdata.csv"
FILE_TEST_U  = "testinglongdata_500.csv"
FILE_TEST_U2 = "testinglongdatav2.csv"   # optional second long-query set


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class SQLInjectionDataset(Dataset):
    """
    Wraps a DataFrame of (query, label) pairs.

    Args:
        df        : DataFrame with columns 'Query' and 'Label'
        tokenizer : SQLTokenizer instance
        max_len   : token sequence length (default 512, matches paper)
    """

    def __init__(self, df: pd.DataFrame, tokenizer: SQLTokenizer,
                 max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.queries   = df["Query"].astype(str).tolist()
        self.labels    = df["Label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int):
        token_ids, label_ids = self.tokenizer.tokenize(self.queries[idx])
        return {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "label_ids": torch.tensor(label_ids, dtype=torch.long),
            "target"   : torch.tensor(self.labels[idx], dtype=torch.float),
        }


# ---------------------------------------------------------------------------
# CSV loading helper
# ---------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    """
    Load one of the Kaggle CSVs and normalise column names to 'Query' / 'Label'.
    Handles minor variations in casing or naming that sometimes appear in the files.
    """
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]

    rename_map = {}
    for col in df.columns:
        low = col.lower()
        if low in ("query", "sentence", "sql", "text"):
            rename_map[col] = "Query"
        elif low in ("label", "class", "target", "injectiontype", "injection_type"):
            rename_map[col] = "Label"
    df = df.rename(columns=rename_map)

    if "Query" not in df.columns or "Label" not in df.columns:
        raise ValueError(
            f"Could not find Query/Label columns in {path}.\n"
            f"Columns found: {list(df.columns)}\n"
            "Rename them to 'Query' and 'Label' and try again."
        )

    # Binarise: 0 = legal, anything else = malicious
    df["Label"] = (df["Label"].astype(str).str.strip() != "0").astype(int)
    df = df.dropna(subset=["Query"]).reset_index(drop=True)
    return df[["Query", "Label"]]


def _load_and_report(path: str, split_name: str) -> pd.DataFrame:
    df = load_csv(path)
    mal = int(df["Label"].sum())
    leg = len(df) - mal
    print(f"  {split_name:<12}: {len(df):>7,} queries  "
          f"(malicious={mal:,}, legal={leg:,})")
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_dataloaders(data_dir: str,
                      tokenizer: SQLTokenizer,
                      max_len:      int = 512,
                      batch_size:   int = 64,
                      num_workers:  int = 0,
                      include_u2:   bool = False):
    """
    Builds DataLoaders from the four Kaggle CSV files.

    Args:
        data_dir    : folder containing the four CSV files
        tokenizer   : SQLTokenizer instance
        max_len     : max token length (N in paper)
        batch_size  : mini-batch size
        num_workers : DataLoader workers
        include_u2  : also load testinglongdatav2.csv as a 4th loader

    Returns:
        train_loader, test_s_loader, test_u_loader
        (and optionally test_u2_loader if include_u2=True)
    """

    def path(fname):
        p = os.path.join(data_dir, fname)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Expected file not found: {p}\n"
                f"Make sure '{fname}' is inside --data_dir='{data_dir}'"
            )
        return p

    print("[Dataset] Loading CSVs …")
    df_train  = _load_and_report(path(FILE_TRAIN),  "Train (T)")
    df_test_s = _load_and_report(path(FILE_TEST_S), "Test-S")
    df_test_u = _load_and_report(path(FILE_TEST_U), "Test-U (500)")

    def make_loader(df, shuffle):
        ds = SQLInjectionDataset(df, tokenizer, max_len)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)

    train_loader  = make_loader(df_train,  shuffle=True)
    test_s_loader = make_loader(df_test_s, shuffle=False)
    test_u_loader = make_loader(df_test_u, shuffle=False)

    label_counts = {
        "train_malicious": int(df_train["Label"].sum()),
        "train_legal":     int((df_train["Label"] == 0).sum()),
    }

    if include_u2:
        df_test_u2   = _load_and_report(path(FILE_TEST_U2), "Test-U2")
        test_u2_loader = make_loader(df_test_u2, shuffle=False)
        return train_loader, test_s_loader, test_u_loader, test_u2_loader, label_counts

    return train_loader, test_s_loader, test_u_loader, label_counts