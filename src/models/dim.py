"""
=============================================================================
Module 2.3 - Dynamic Interest Modeling (DIM) Architecture
=============================================================================
Arsitektur DIM memproses high-risk alerts sebagai sequential modality.

Komponen:
  1. Embedding Layer    : Diskrit features -> Dense vector space
  2. Long-Term Module   : Transformer Encoder (scaled dot-product attention)
                          untuk preferensi strategi mitigasi jangka panjang
  3. Short-Term Module  : LSTM + Forget Gate untuk evolusi taktik serangan
  4. MLP Fusion         : Gabungkan kedua representasi -> p(match | playbook, alert)

Formulasi:
  Embedding   : e_i = W_e × x_i + b_e
  Attention   : Attn(Q,K,V) = softmax( QK^T / √d_k ) × V
  LSTM gate   : f_t = σ( W_f [h_{t-1}, x_t] + b_f )
  Output      : p = sigmoid( MLP([h_lt ; h_st]) )
=============================================================================
"""

import math
import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ===========================================================================
# Sub-module 1: Embedding Layer
# ===========================================================================

class AlertPlaybookEmbedding(nn.Module):
    """
    Memetakan fitur diskrit (alert_type, severity, tactic_id, playbook_id)
    ke dalam dense vector space melalui shared embedding table.

    e_i = W_e[x_i] + positional_encoding(i)
    """

    def __init__(
        self,
        num_alert_types:  int,
        num_playbooks:    int,
        num_tactics:      int,
        embed_dim:        int = 64,
        max_seq_len:      int = 50,
        dropout:          float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Embedding tables
        self.alert_embed    = nn.Embedding(num_alert_types + 1, embed_dim, padding_idx=0)
        self.playbook_embed = nn.Embedding(num_playbooks   + 1, embed_dim, padding_idx=0)
        self.tactic_embed   = nn.Embedding(num_tactics     + 1, embed_dim, padding_idx=0)
        self.severity_embed = nn.Embedding(6,                   embed_dim, padding_idx=0)  # 0-5

        # Fuse 4 modalities -> embed_dim
        self.fusion = nn.Linear(embed_dim * 4, embed_dim)

        # Positional encoding (sinusoidal)
        self.register_buffer(
            "pos_enc",
            self._sinusoidal_encoding(max_seq_len, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        alert_ids:    torch.Tensor,  # [B, T]
        playbook_ids: torch.Tensor,  # [B, T]
        tactic_ids:   torch.Tensor,  # [B, T]
        severity:     torch.Tensor,  # [B, T]
    ) -> torch.Tensor:               # [B, T, embed_dim]
        T = alert_ids.size(1)
        e_alert    = self.alert_embed(alert_ids)       # [B, T, D]
        e_playbook = self.playbook_embed(playbook_ids) # [B, T, D]
        e_tactic   = self.tactic_embed(tactic_ids)     # [B, T, D]
        e_severity = self.severity_embed(severity)     # [B, T, D]

        # Concatenate & fuse
        fused = torch.cat([e_alert, e_playbook, e_tactic, e_severity], dim=-1)
        fused = self.fusion(fused)  # [B, T, D]

        # Add positional encoding
        fused = fused + self.pos_enc[:T, :].unsqueeze(0)
        return self.dropout(fused)

    @staticmethod
    def _sinusoidal_encoding(max_len: int, d_model: int) -> torch.Tensor:
        """Sinusoidal positional encoding dari 'Attention Is All You Need'."""
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe


# ===========================================================================
# Sub-module 2: Long-Term Interest Modeling (Transformer Encoder)
# ===========================================================================

class LongTermInterestModule(nn.Module):
    """
    Transformer Encoder untuk mengekstraksi preferensi strategi mitigasi
    jangka panjang analis.

    Scaled dot-product attention:
        Q = X W_Q,  K = X W_K,  V = X W_V
        Attn(Q,K,V) = softmax( Q K^T / √d_k ) V
    """

    def __init__(
        self,
        embed_dim:   int = 64,
        num_heads:   int = 4,
        num_layers:  int = 2,
        ff_dim:      int = 256,
        dropout:     float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,       # input: [B, T, D]
            norm_first=True,        # Pre-LN (lebih stabil)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # global average pooling -> [B, D]

    def forward(
        self,
        x:           torch.Tensor,         # [B, T, D]
        src_key_padding_mask: Optional[torch.Tensor] = None,  # [B, T] bool
    ) -> torch.Tensor:                     # [B, D]
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        # Pool across time dimension
        out = self.pool(out.transpose(1, 2)).squeeze(-1)  # [B, D]
        return out


# ===========================================================================
# Sub-module 3: Short-Term Interest Modeling (LSTM + Forget Gate)
# ===========================================================================

class ShortTermInterestModule(nn.Module):
    """
    LSTM dengan forget gate untuk menangkap evolusi taktik serangan jangka pendek.

    Forget gate memudarkan bobot taktik pertahanan yang sudah usang:
        f_t = σ( W_f [h_{t-1}, x_t] + b_f )
        i_t = σ( W_i [h_{t-1}, x_t] + b_i )
        g_t = tanh( W_g [h_{t-1}, x_t] + b_g )
        o_t = σ( W_o [h_{t-1}, x_t] + b_o )
        c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
        h_t = o_t ⊙ tanh(c_t)
    """

    def __init__(
        self,
        embed_dim:   int   = 64,
        hidden_dim:  int   = 128,
        num_layers:  int   = 2,
        dropout:     float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        dirs = 2 if bidirectional else 1
        self.out_dim = hidden_dim * dirs

        # Layer norm untuk stabilisasi
        self.layer_norm = nn.LayerNorm(self.out_dim)

    def forward(
        self,
        x: torch.Tensor,            # [B, T, embed_dim]
        lengths: Optional[torch.Tensor] = None,  # [B] actual sequence lengths
    ) -> torch.Tensor:              # [B, hidden_dim]
        if lengths is not None:
            # Pack untuk efisiensi pada variable-length sequences
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out_packed, (h_n, _) = self.lstm(packed)
            # Ambil hidden state layer terakhir
            h_last = h_n[-1]  # [B, hidden_dim]
        else:
            _, (h_n, _) = self.lstm(x)
            h_last = h_n[-1]  # [B, hidden_dim]

        return self.layer_norm(h_last)


# ===========================================================================
# Sub-module 4: MLP Fusion -> Output Probability
# ===========================================================================

class MLPFusion(nn.Module):
    """
    Multi-Layer Perceptron untuk menggabungkan representasi long-term
    dan short-term menjadi probabilitas kecocokan p.

    Input : [h_lt ; h_st ; e_candidate]  (concatenated)
    Output: p ∈ (0, 1)
    """

    def __init__(
        self,
        lt_dim:    int,
        st_dim:    int,
        cand_dim:  int,
        hidden_sizes: List[int] = [256, 128, 64],
        dropout:   float = 0.2,
    ):
        super().__init__()
        in_dim = lt_dim + st_dim + cand_dim
        layers = []
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(h),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        h_lt:   torch.Tensor,   # [B, lt_dim]
        h_st:   torch.Tensor,   # [B, st_dim]
        e_cand: torch.Tensor,   # [B, cand_dim]
    ) -> torch.Tensor:           # [B] - probabilities
        x = torch.cat([h_lt, h_st, e_cand], dim=-1)
        return torch.sigmoid(self.net(x)).squeeze(-1)


# ===========================================================================
# Main: Dynamic Interest Model
# ===========================================================================

class DynamicInterestModel(nn.Module):
    """
    Dynamic Interest Modeling (DIM) - Arsitektur lengkap.

    Input:
        - Sekuensi historis interaksi playbook H = {x_1, ..., x_{t-1}}
        - Kandidat playbook saat ini t

    Pipeline:
        H -> Embedding -> [Long-Term (Transformer) | Short-Term (LSTM)] -> MLP -> p

    Output:
        p ∈ (0,1): probabilitas kecocokan playbook kandidat dengan alert aktif
    """

    def __init__(
        self,
        num_alert_types:  int,
        num_playbooks:    int,
        num_tactics:      int,
        embed_dim:        int   = 64,
        lt_heads:         int   = 4,
        lt_layers:        int   = 2,
        st_hidden:        int   = 128,
        st_layers:        int   = 2,
        mlp_hidden:       List[int] = [256, 128, 64],
        dropout:          float = 0.1,
        max_seq_len:      int   = 50,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Embedding
        self.embedding = AlertPlaybookEmbedding(
            num_alert_types=num_alert_types,
            num_playbooks=num_playbooks,
            num_tactics=num_tactics,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        # Long-Term (Transformer)
        self.lt_module = LongTermInterestModule(
            embed_dim=embed_dim,
            num_heads=lt_heads,
            num_layers=lt_layers,
            ff_dim=embed_dim * 4,
            dropout=dropout,
        )

        # Short-Term (LSTM)
        self.st_module = ShortTermInterestModule(
            embed_dim=embed_dim,
            hidden_dim=st_hidden,
            num_layers=st_layers,
            dropout=dropout,
        )

        # Candidate playbook embedding (single item)
        self.candidate_embed = nn.Embedding(num_playbooks + 1, embed_dim, padding_idx=0)

        # MLP Fusion
        self.fusion = MLPFusion(
            lt_dim=embed_dim,
            st_dim=self.st_module.out_dim,
            cand_dim=embed_dim,
            hidden_sizes=mlp_hidden,
            dropout=dropout,
        )

        self._init_weights()
        logger.info(
            f"DIM initialized | params={sum(p.numel() for p in self.parameters()):,}"
        )

    def _init_weights(self):
        """Xavier uniform init untuk layer Linear."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        # Sekuensi historis H = {x_1,...,x_{t-1}}
        hist_alert_ids:    torch.Tensor,   # [B, T]
        hist_playbook_ids: torch.Tensor,   # [B, T]
        hist_tactic_ids:   torch.Tensor,   # [B, T]
        hist_severity:     torch.Tensor,   # [B, T]
        # Kandidat playbook saat ini
        cand_playbook_id:  torch.Tensor,   # [B]
        # Optional: mask untuk padding
        padding_mask:      Optional[torch.Tensor] = None,  # [B, T] bool
        seq_lengths:       Optional[torch.Tensor] = None,  # [B]
    ) -> torch.Tensor:                     # [B] probabilities
        # 1) Embed sekuensi historis
        h_embed = self.embedding(
            hist_alert_ids,
            hist_playbook_ids,
            hist_tactic_ids,
            hist_severity,
        )  # [B, T, D]

        # 2) Long-term: Transformer Encoder
        h_lt = self.lt_module(h_embed, src_key_padding_mask=padding_mask)  # [B, D]

        # 3) Short-term: LSTM
        h_st = self.st_module(h_embed, lengths=seq_lengths)  # [B, hidden]

        # 4) Candidate embedding
        e_cand = self.candidate_embed(cand_playbook_id)  # [B, D]

        # 5) MLP Fusion -> probability
        p = self.fusion(h_lt, h_st, e_cand)  # [B]
        return p

    def predict_top_k(
        self,
        hist_alert_ids:    torch.Tensor,
        hist_playbook_ids: torch.Tensor,
        hist_tactic_ids:   torch.Tensor,
        hist_severity:     torch.Tensor,
        num_playbooks:     int,
        k:                 int = 5,
        device:            str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prediksi top-K playbook terbaik untuk alert saat ini.

        Returns:
            (top_k_indices, top_k_scores) - keduanya [B, K]
        """
        self.eval()
        B = hist_alert_ids.size(0)
        all_scores = []

        with torch.no_grad():
            for pid in range(1, num_playbooks + 1):
                cand = torch.full((B,), pid, dtype=torch.long, device=device)
                scores = self.forward(
                    hist_alert_ids, hist_playbook_ids,
                    hist_tactic_ids, hist_severity, cand
                )
                all_scores.append(scores.unsqueeze(1))  # [B, 1]

        all_scores = torch.cat(all_scores, dim=1)  # [B, num_playbooks]
        top_scores, top_idx = torch.topk(all_scores, k=min(k, num_playbooks), dim=1)
        return top_idx + 1, top_scores  # +1 karena playbook ID mulai dari 1


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(42)

    B, T = 4, 20  # batch_size=4, seq_len=20

    model = DynamicInterestModel(
        num_alert_types=50,
        num_playbooks=30,
        num_tactics=14,
        embed_dim=64,
        lt_heads=4,
        lt_layers=2,
        st_hidden=128,
        st_layers=2,
    )

    # Dummy input
    hist_alerts   = torch.randint(1, 51, (B, T))
    hist_playbook = torch.randint(1, 31, (B, T))
    hist_tactic   = torch.randint(1, 15, (B, T))
    hist_sev      = torch.randint(1, 6,  (B, T))
    cand_playbook = torch.randint(1, 31, (B,))

    probs = model(hist_alerts, hist_playbook, hist_tactic, hist_sev, cand_playbook)
    print(f"Output probabilities shape : {probs.shape}")   # [4]
    print(f"Probability values         : {probs.detach()}")

    # Top-K prediction
    top_idx, top_scores = model.predict_top_k(
        hist_alerts, hist_playbook, hist_tactic, hist_sev,
        num_playbooks=30, k=5
    )
    print(f"\nTop-5 playbook indices  : {top_idx}")
    print(f"Top-5 playbook scores   : {top_scores}")
