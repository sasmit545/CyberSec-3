"""
SQL-Specific Tokenizer
======================
Implements the SQL-specific tokenizer from Section 3.3 of the paper.

Key design decisions (from paper):
- Only SQL keywords and symbols are kept as tokens (user-defined identifiers removed)
- Three semantic label classes: command (1), expression (2), symbol (3)
- Vocabulary size = 158
- Zero-padding to fixed length N
"""

import re
from typing import Tuple, List

# ---------------------------------------------------------------------------
# Vocabulary: 139 command tokens + 14 expression tokens + 5 symbol tokens = 158
# ---------------------------------------------------------------------------

# Semantic label IDs
LABEL_PAD     = 0   # padding
LABEL_COMMAND = 1   # SQL keywords / commands
LABEL_EXPR    = 2   # expression-related keywords (operators, predicates)
LABEL_SYMBOL  = 3   # punctuation / symbols

# --- 139 command tokens ---
COMMAND_TOKENS = {
    "SELECT", "FROM", "WHERE", "INSERT", "INTO", "VALUES", "UPDATE", "SET",
    "DELETE", "CREATE", "TABLE", "DROP", "ALTER", "ADD", "COLUMN", "INDEX",
    "VIEW", "TRIGGER", "PROCEDURE", "FUNCTION", "DATABASE", "SCHEMA",
    "GRANT", "REVOKE", "COMMIT", "ROLLBACK", "SAVEPOINT", "TRANSACTION",
    "BEGIN", "END", "IF", "ELSE", "THEN", "CASE", "WHEN", "RETURN",
    "DECLARE", "CURSOR", "FETCH", "OPEN", "CLOSE", "EXEC", "EXECUTE",
    "CALL", "DO", "LOOP", "WHILE", "REPEAT", "UNTIL", "LEAVE", "ITERATE",
    "HANDLER", "SIGNAL", "RESIGNAL", "GET", "DIAGNOSTICS", "CONDITION",
    "CONTINUE", "EXIT", "UNDO", "FOR", "EACH", "ROW", "BEFORE", "AFTER",
    "INSTEAD", "OF", "ON", "AS", "JOIN", "INNER", "LEFT", "RIGHT", "FULL",
    "OUTER", "CROSS", "NATURAL", "USING", "WITH", "RECURSIVE", "UNION",
    "INTERSECT", "EXCEPT", "ALL", "DISTINCT", "TOP", "LIMIT", "OFFSET",
    "FETCH", "ROWS", "ONLY", "ORDER", "BY", "GROUP", "HAVING", "ASC",
    "DESC", "PRIMARY", "FOREIGN", "KEY", "REFERENCES", "UNIQUE", "CHECK",
    "DEFAULT", "NOT", "NULL", "AUTO_INCREMENT", "IDENTITY", "CONSTRAINT",
    "ENABLE", "DISABLE", "CASCADE", "RESTRICT", "NO", "ACTION", "MATCH",
    "PARTIAL", "SIMPLE", "FULL", "DEFERRABLE", "DEFERRED", "IMMEDIATE",
    "INITIALLY", "TEMPORARY", "TEMP", "GLOBAL", "LOCAL", "UNLOGGED",
    "MATERIALIZED", "REPLACE", "TRUNCATE", "MERGE", "UPSERT", "LOAD",
    "IMPORT", "EXPORT", "BACKUP", "RESTORE", "SHOW", "DESCRIBE", "EXPLAIN",
    "ANALYZE", "VACUUM", "REINDEX", "CLUSTER", "REFRESH", "LOCK", "UNLOCK",
    "FLUSH", "RESET", "OPTIMIZE", "REPAIR", "CHECK", "CHECKSUM",
    "PARTITION", "SUBPARTITION", "RANGE", "LIST", "HASH", "KEY",
    "TABLESPACE", "DATAFILE", "LOGFILE",
}

# --- 14 expression tokens ---
EXPR_TOKENS = {
    "AND", "OR", "IN", "NOT", "LIKE", "BETWEEN", "IS", "EXISTS",
    "ANY", "SOME", "ALL", "ILIKE", "RLIKE", "REGEXP",
}

# --- 5 symbol tokens ---
SYMBOL_TOKENS = {"=", "<", ">", "(", ")"}

# All tokens with their labels
ALL_TOKENS: dict[str, int] = {}
for t in COMMAND_TOKENS:
    ALL_TOKENS[t] = LABEL_COMMAND
for t in EXPR_TOKENS:
    ALL_TOKENS[t] = LABEL_EXPR
for t in SYMBOL_TOKENS:
    ALL_TOKENS[t] = LABEL_SYMBOL

# Build vocabulary: token -> integer index (1-based; 0 reserved for PAD)
VOCAB: dict[str, int] = {tok: idx + 1 for idx, tok in enumerate(sorted(ALL_TOKENS.keys()))}
VOCAB_SIZE = len(VOCAB) + 1  # +1 for PAD (index 0)

# Number of semantic labels (including PAD=0)
NUM_SEMANTIC_LABELS = 4   # 0=pad, 1=command, 2=expr, 3=symbol


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class SQLTokenizer:
    """
    Converts a raw SQL query string into:
        - token_ids   : List[int]  (indices into VOCAB, 0=pad)
        - label_ids   : List[int]  (semantic label per token, 0=pad)

    The tokenizer strips user-defined identifiers/attribute values so that
    only SQL keywords and the five recognised symbols remain (skeleton query).
    """

    # Regex that splits on whitespace and isolates the 5 symbol characters
    _SPLIT_RE = re.compile(r"([=<>()\s,;'\"])")

    def __init__(self, max_len: int = 512):
        self.max_len = max_len
        self.vocab = VOCAB
        self.vocab_size = VOCAB_SIZE

    # ------------------------------------------------------------------
    def tokenize(self, query: str) -> Tuple[List[int], List[int]]:
        """
        Returns (token_ids, label_ids) each of length max_len (zero-padded).
        """
        raw_tokens = self._split(query)
        token_ids: List[int] = []
        label_ids: List[int] = []

        for tok in raw_tokens:
            upper = tok.upper()
            if upper in ALL_TOKENS:
                token_ids.append(self.vocab[upper])
                label_ids.append(ALL_TOKENS[upper])

        # Truncate or pad to max_len
        token_ids = token_ids[: self.max_len]
        label_ids = label_ids[: self.max_len]

        pad_len = self.max_len - len(token_ids)
        token_ids += [0] * pad_len
        label_ids += [0] * pad_len

        return token_ids, label_ids

    # ------------------------------------------------------------------
    def _split(self, query: str) -> List[str]:
        """Splits query into candidate tokens."""
        parts = self._SPLIT_RE.split(query)
        return [p.strip() for p in parts if p.strip()]

    # ------------------------------------------------------------------
    def decode_skeleton(self, query: str) -> str:
        """Returns the skeleton query (only recognised tokens kept)."""
        raw_tokens = self._split(query)
        skeleton = [t.upper() for t in raw_tokens if t.upper() in ALL_TOKENS]
        return " ".join(skeleton)
