"""
SQLi Detection Model
====================
Implements the full neural network from Section 3 of the paper in PyTorch.

Architecture (Figure 3 & 5):
  Input Embedding Section
  ├── Token Embedding  (f1)  : token_id  -> R^M1
  ├── Semantic Embedding (f2): label_id  -> R^M2
  ├── Concatenate            : -> R^(M1+M2) per position  [F matrix]
  └── Positional Encoding    : sinusoidal C added to F    [U matrix]

  Detection Section
  ├── CNN Stage              : 2× (Conv1d → MaxPool)  extract local features
  ├── Self-Attention Stage   : h-head scaled dot-product attention
  └── Output Stage           : LayerNorm → Dense → Sigmoid → z ∈ [0,1]

Paper hyperparameters (Section 4.1):
  N=512, M1=10, M2=1, M=11, h=4, vocab=158, num_labels=4
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizer import VOCAB_SIZE, NUM_SEMANTIC_LABELS


# ---------------------------------------------------------------------------
# Positional Encoding  (Equation 1 in paper)
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """
    Precomputes C of shape (N, M) using sinusoidal encoding.
    C(i,j) = sin(i / 10000^(j/N))   when j is even
           = cos(i / 10000^((j-1)/N)) when j is odd
    """
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        C = torch.zeros(max_len, d_model)
        positions = torch.arange(1, max_len + 1, dtype=torch.float).unsqueeze(1)  # (N,1)
        dims = torch.arange(d_model, dtype=torch.float)  # (M,)

        # Even indices: sin
        even_mask = (dims % 2 == 0)
        exponent_even = dims[even_mask] / max_len
        C[:, even_mask] = torch.sin(positions / (10000 ** exponent_even))

        # Odd indices: cos
        odd_mask = ~even_mask
        exponent_odd = (dims[odd_mask] - 1) / max_len
        C[:, odd_mask] = torch.cos(positions / (10000 ** exponent_odd))

        # Register as a non-trainable buffer
        self.register_buffer("C", C)  # (N, M)

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        """F: (batch, N, M)  →  U = F + C  (broadcast over batch)"""
        return F + self.C.unsqueeze(0)  # (1, N, M)


# ---------------------------------------------------------------------------
# Input Embedding Section (Section 3.3)
# ---------------------------------------------------------------------------

class InputEmbedding(nn.Module):
    """
    Maps (token_ids, label_ids) → encoded representation matrix U of shape (B, N, M).

    f1 : Embedding(vocab_size,  M1)
    f2 : Embedding(num_labels,  M2)
    Concat → F (B, N, M1+M2)
    + Positional Encoding C → U (B, N, M)
    """
    def __init__(self, vocab_size: int, num_labels: int,
                 max_len: int, m1: int, m2: int):
        super().__init__()
        self.token_embedding    = nn.Embedding(vocab_size,  m1, padding_idx=0)
        self.semantic_embedding = nn.Embedding(num_labels, m2, padding_idx=0)
        self.pos_encoding       = SinusoidalPositionalEncoding(max_len, m1 + m2)

    def forward(self, token_ids: torch.Tensor,
                label_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids : (B, N)  LongTensor
        label_ids : (B, N)  LongTensor
        returns U : (B, N, M)
        """
        tok_emb = self.token_embedding(token_ids)       # (B, N, M1)
        sem_emb = self.semantic_embedding(label_ids)    # (B, N, M2)
        F = torch.cat([tok_emb, sem_emb], dim=-1)       # (B, N, M)
        U = self.pos_encoding(F)                        # (B, N, M)
        return U


# ---------------------------------------------------------------------------
# CNN Stage (Section 3.4)
# ---------------------------------------------------------------------------

class CNNStage(nn.Module):
    """
    Two Conv1d layers with MaxPool to extract local features.
    Input  : (B, N, M)    — sequence-first format
    Output : (B, N', M')  — after two conv+pool operations
    """
    def __init__(self, in_channels: int, mid_channels: int = 64,
                 out_channels: int = 128, kernel_size: int = 3):
        super().__init__()
        # Conv1d expects (B, C, L); we treat the embedding dim as channels
        self.conv1  = nn.Conv1d(in_channels,  mid_channels, kernel_size, padding=kernel_size // 2)
        self.pool1  = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2  = nn.Conv1d(mid_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.pool2  = nn.MaxPool1d(kernel_size=2, stride=2)
        self.out_channels = out_channels

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        """
        U : (B, N, M)
        X : (B, N//4, out_channels)
        """
        x = U.permute(0, 2, 1)          # (B, M, N)  — channels first for Conv1d
        x = F.relu(self.conv1(x))       # (B, mid, N)
        x = self.pool1(x)               # (B, mid, N//2)
        x = F.relu(self.conv2(x))       # (B, out, N//2)
        x = self.pool2(x)               # (B, out, N//4)
        x = x.permute(0, 2, 1)         # (B, N//4, out)
        return x


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention Stage (Section 3.4, Equations 3-6)
# ---------------------------------------------------------------------------

class MultiHeadSelfAttentionStage(nn.Module):
    """
    Scaled dot-product multi-head attention.
    Equations 3-6 from the paper, using PyTorch's built-in MHA for efficiency
    (equivalent formulation).
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn    = nn.MultiheadAttention(embed_dim, num_heads,
                                             dropout=dropout, batch_first=True)
        self.norm    = nn.LayerNorm(embed_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: (B, seq_len, embed_dim)  →  H(X): same shape"""
        attn_out, _ = self.attn(X, X, X)    # Q=K=V=X  (self-attention)
        return self.norm(X + attn_out)       # residual + LayerNorm


# ---------------------------------------------------------------------------
# Output Stage (Section 3.4)
# ---------------------------------------------------------------------------

class OutputStage(nn.Module):
    """
    Global average pool → Dense → Sigmoid → scalar z ∈ [0,1].
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """H: (B, seq, embed)  →  z: (B,)"""
        pooled = H.mean(dim=1)          # (B, embed)  global average pool
        z = torch.sigmoid(self.fc(pooled)).squeeze(-1)   # (B,)
        return z


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class SQLiDetector(nn.Module):
    """
    Complete SQL injection detection model from the paper.

    Default hyperparameters match Table 2 / Section 4.1:
        N=512, M1=10, M2=1  → M=11
        CNN: 11 → 64 → 128 channels
        Self-attention: h=4 heads on 128-dim vectors
        Total weights ≈ 69k
    """
    def __init__(self,
                 vocab_size:   int = VOCAB_SIZE,
                 num_labels:   int = NUM_SEMANTIC_LABELS,
                 max_len:      int = 512,
                 m1:           int = 10,
                 m2:           int = 1,
                 cnn_mid:      int = 64,
                 cnn_out:      int = 128,
                 num_heads:    int = 4,
                 dropout:      float = 0.1,
                 threshold:    float = 0.5):
        super().__init__()
        self.threshold = threshold
        m = m1 + m2   # combined embedding dim (=11)

        self.embedding  = InputEmbedding(vocab_size, num_labels, max_len, m1, m2)
        self.cnn        = CNNStage(in_channels=m,
                                   mid_channels=cnn_mid,
                                   out_channels=cnn_out)
        self.attention  = MultiHeadSelfAttentionStage(embed_dim=cnn_out,
                                                      num_heads=num_heads,
                                                      dropout=dropout)
        self.output     = OutputStage(embed_dim=cnn_out)

    # ------------------------------------------------------------------
    def forward(self, token_ids: torch.Tensor,
                label_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids : (B, N) LongTensor
        label_ids : (B, N) LongTensor
        returns z : (B,)  FloatTensor  — probability of malicious query
        """
        U = self.embedding(token_ids, label_ids)   # (B, N, M)
        X = self.cnn(U)                            # (B, N//4, cnn_out)
        H = self.attention(X)                      # (B, N//4, cnn_out)
        z = self.output(H)                         # (B,)
        return z

    # ------------------------------------------------------------------
    def predict(self, token_ids: torch.Tensor,
                label_ids: torch.Tensor) -> torch.Tensor:
        """Returns binary prediction (1=malicious, 0=legal)."""
        with torch.no_grad():
            z = self.forward(token_ids, label_ids)
        return (z >= self.threshold).long()

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = SQLiDetector()
    print(f"Total trainable parameters: {model.count_parameters():,}")

    B, N = 4, 512
    tok = torch.randint(0, VOCAB_SIZE,  (B, N))
    lab = torch.randint(0, NUM_SEMANTIC_LABELS, (B, N))
    z = model(tok, lab)
    print(f"Output shape: {z.shape}   (expected: [{B}])")
    print(f"Sample outputs: {z.detach().numpy()}")
