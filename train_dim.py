"""
Training Script - Dynamic Interest Model (DIM)

Train DIM using Splunk BOTS v3 synthetic sequences with
ranking evaluation metrics (HR, MAP, NDCG).
"""


import os
import sys
import logging
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# Pastikan root project ada di PYTHONPATH
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.dim import DynamicInterestModel
from src.data.dataset_loader import SplunkBOTSLoader, PLAYBOOK_ID_MAP
from src.evaluation.metrics import RankingMetrics, MajorityVoteBaseline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/dim_training.log", mode="a", encoding="utf-8"),

    ],
)
logger = logging.getLogger("DIMTrainer")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG = {
    # Model
    "num_alert_types":  50,
    "num_playbooks":    len(PLAYBOOK_ID_MAP),
    "num_tactics":      14,
    "embed_dim":        64,
    "lt_heads":         4,
    "lt_layers":        2,
    "st_hidden":        128,
    "st_layers":        2,
    "mlp_hidden":       [256, 128, 64],
    "dropout":          0.1,
    "max_seq_len":      20,

    # Training
    "n_synthetic_sequences": 10000,
    "batch_size":       64,
    "epochs":           50,       # max epochs (early stopping akan hentikan lebih awal)
    "lr":               1e-3,
    "weight_decay":     1e-4,
    "neg_sample_ratio": 4,        # negatif per positif
    "k_eval":           [1, 3, 5, 10],

    # Early Stopping
    "early_stopping_patience": 7,   # berhenti jika NDCG@5 tidak naik selama N epoch
    "early_stopping_min_delta": 1e-4,  # minimum improvement yang dianggap signifikan

    # System
    "device":           "cuda" if torch.cuda.is_available() else "cpu",
    "seed":             42,
    "checkpoint_dir":   "checkpoints",
    "log_interval":     10,
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PlaybookSequenceDataset(Dataset):
    """
    Dataset PyTorch untuk pelatihan DIM.
    Setiap item = (sekuensi historis, playbook kandidat, label)
    Label = 1 (positif/match), 0 (negatif/non-match)
    """

    def __init__(
        self,
        data:             Dict,
        num_playbooks:    int,
        neg_sample_ratio: int = 4,
    ):
        self.data             = data
        self.num_playbooks    = num_playbooks
        self.neg_sample_ratio = neg_sample_ratio
        self.n                = data["hist_alert_ids"].size(0)

    def __len__(self):
        # Setiap item positif diikuti neg_sample_ratio item negatif
        return self.n * (1 + self.neg_sample_ratio)

    def __getitem__(self, idx: int) -> Dict:
        base_idx = idx // (1 + self.neg_sample_ratio)
        is_pos   = (idx % (1 + self.neg_sample_ratio)) == 0

        hist_alert   = self.data["hist_alert_ids"][base_idx]
        hist_pb      = self.data["hist_playbook_ids"][base_idx]
        hist_tactic  = self.data["hist_tactic_ids"][base_idx]
        hist_sev     = self.data["hist_severity"][base_idx]
        pad_mask     = self.data["padding_mask"][base_idx]
        seq_len      = self.data["seq_lengths"][base_idx]
        target_pb    = self.data["target_playbook"][base_idx].item()

        if is_pos:
            cand_pb = target_pb
            label   = 1.0
        else:
            # Sampling negatif: pilih playbook yang BUKAN target
            cand_pb = target_pb
            while cand_pb == target_pb:
                cand_pb = random.randint(1, self.num_playbooks)
            label = 0.0

        return {
            "hist_alert_ids":    hist_alert,
            "hist_playbook_ids": hist_pb,
            "hist_tactic_ids":   hist_tactic,
            "hist_severity":     hist_sev,
            "padding_mask":      pad_mask,
            "seq_lengths":       seq_len,
            "cand_playbook_id":  torch.tensor(cand_pb, dtype=torch.long),
            "label":             torch.tensor(label, dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DIMTrainer:
    """Trainer lengkap untuk Dynamic Interest Model."""

    def __init__(self, config: Dict):
        self.config  = config
        self.device  = torch.device(config["device"])
        self._set_seed(config["seed"])

        # Buat direktori
        os.makedirs(config["checkpoint_dir"], exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        # Model
        self.model = DynamicInterestModel(
            num_alert_types=config["num_alert_types"],
            num_playbooks=config["num_playbooks"],
            num_tactics=config["num_tactics"],
            embed_dim=config["embed_dim"],
            lt_heads=config["lt_heads"],
            lt_layers=config["lt_layers"],
            st_hidden=config["st_hidden"],
            st_layers=config["st_layers"],
            mlp_hidden=config["mlp_hidden"],
            dropout=config["dropout"],
            max_seq_len=config["max_seq_len"],
        ).to(self.device)

        logger.info(
            f"Model parameters: "
            f"{sum(p.numel() for p in self.model.parameters()):,}"
        )

        # Loss & Optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config["epochs"],
            eta_min=1e-6,
        )

        # Metrics
        self.rank_metrics = RankingMetrics()
        self.best_ndcg    = 0.0
        self.history: List[Dict] = []

        # Early stopping state
        self._es_patience  = config.get("early_stopping_patience", 7)
        self._es_min_delta = config.get("early_stopping_min_delta", 1e-4)
        self._es_counter   = 0   # jumlah epoch tanpa improvement
        self._es_best_ndcg = 0.0

    def load_data(self) -> Tuple[DataLoader, DataLoader, Dict]:
        """Load dan siapkan dataset."""
        logger.info("Loading Splunk BOTS v3 dataset...")
        loader = SplunkBOTSLoader(
            seq_len=self.config["max_seq_len"],
            random_state=self.config["seed"],
        )
        train_data, test_data = loader.load(
            n_synthetic=self.config["n_synthetic_sequences"]
        )

        train_ds = PlaybookSequenceDataset(
            train_data,
            num_playbooks=self.config["num_playbooks"],
            neg_sample_ratio=self.config["neg_sample_ratio"],
        )
        train_dl = DataLoader(
            train_ds,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )
        logger.info(
            f"Train: {len(train_ds)} samples ({len(train_dl)} batches)"
        )
        return train_dl, test_data, train_data

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Satu epoch pelatihan."""
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for i, batch in enumerate(dataloader):
            # Pindahkan ke device
            hist_alerts    = batch["hist_alert_ids"].to(self.device)
            hist_playbooks = batch["hist_playbook_ids"].to(self.device)
            hist_tactics   = batch["hist_tactic_ids"].to(self.device)
            hist_sev       = batch["hist_severity"].to(self.device)
            cand_pb        = batch["cand_playbook_id"].to(self.device)
            pad_mask       = batch["padding_mask"].to(self.device)
            labels         = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            probs = self.model(
                hist_alert_ids=hist_alerts,
                hist_playbook_ids=hist_playbooks,
                hist_tactic_ids=hist_tactics,
                hist_severity=hist_sev,
                cand_playbook_id=cand_pb,
                padding_mask=pad_mask,
            )

            # BCE Loss
            loss = self.criterion(probs, labels)
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

            if (i + 1) % self.config["log_interval"] == 0:
                avg_loss = total_loss / n_batches
                logger.info(
                    f"  Epoch [{epoch}/{self.config['epochs']}] "
                    f"Step [{i+1}/{len(dataloader)}] "
                    f"Loss: {avg_loss:.6f}"
                )

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self, test_data: Dict) -> Dict:
        """Evaluasi model pada test set menggunakan ranking metrics."""
        self.model.eval()
        n = test_data["hist_alert_ids"].size(0)

        recommendations = []
        ground_truths   = test_data["target_playbook"].tolist()

        # Predict top-K untuk setiap test query
        batch_size = 64
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            top_idx, _ = self.model.predict_top_k(
                hist_alert_ids=test_data["hist_alert_ids"][start:end].to(self.device),
                hist_playbook_ids=test_data["hist_playbook_ids"][start:end].to(self.device),
                hist_tactic_ids=test_data["hist_tactic_ids"][start:end].to(self.device),
                hist_severity=test_data["hist_severity"][start:end].to(self.device),
                num_playbooks=self.config["num_playbooks"],
                k=max(self.config["k_eval"]),
                device=str(self.device),
            )
            recommendations.extend(top_idx.tolist())

        return self.rank_metrics.compute_all(
            recommendations,
            ground_truths,
            k_values=self.config["k_eval"],
        )

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Simpan checkpoint model."""
        ckpt = {
            "epoch":        epoch,
            "model_state":  self.model.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "config":       self.config,
            "metrics":      metrics,
        }
        path = Path(self.config["checkpoint_dir"])
        torch.save(ckpt, path / f"dim_epoch_{epoch:03d}.pt")
        if is_best:
            torch.save(ckpt, path / "dim_best.pt")
            logger.info(f"  [best] NDCG@5={metrics.get('ndcg@5', 0):.4f} - checkpoint saved")

    def _check_early_stopping(self, ndcg5: float) -> bool:
        """
        Cek apakah training harus dihentikan.

        Returns True jika early stopping terpicu.
        Counter di-reset setiap ada improvement >= min_delta.
        """
        if ndcg5 > self._es_best_ndcg + self._es_min_delta:
            self._es_best_ndcg = ndcg5
            self._es_counter   = 0
            return False
        else:
            self._es_counter += 1
            logger.info(
                f"  [EarlyStopping] No improvement for {self._es_counter}/{self._es_patience} epochs "
                f"(best NDCG@5={self._es_best_ndcg:.4f})"
            )
            return self._es_counter >= self._es_patience

    def train(self):
        """Loop pelatihan utama dengan early stopping."""
        train_dl, test_data, _ = self.load_data()

        logger.info(f"\n{'='*60}")
        logger.info(f"  Starting DIM Training on {self.device}")
        logger.info(f"  Max Epochs  : {self.config['epochs']}")
        logger.info(f"  Early Stop  : patience={self._es_patience}, min_delta={self._es_min_delta}")
        logger.info(f"{'='*60}\n")

        stopped_early = False
        for epoch in range(1, self.config["epochs"] + 1):
            t_start = time.time()

            # Train
            train_loss = self.train_epoch(train_dl, epoch)
            self.scheduler.step()

            # Evaluate
            eval_metrics = self.evaluate(test_data)
            ndcg5 = eval_metrics.get("ndcg@5", 0.0)

            elapsed = time.time() - t_start
            logger.info(
                f"Epoch {epoch:3d}/{self.config['epochs']} | "
                f"Loss={train_loss:.6f} | "
                f"HR@5={eval_metrics.get('hit_ratio@5', 0):.4f} | "
                f"MAP@5={eval_metrics.get('map@5', 0):.4f} | "
                f"NDCG@5={ndcg5:.4f} | "
                f"Time={elapsed:.1f}s"
            )

            # Save checkpoint
            is_best = ndcg5 > self.best_ndcg
            if is_best:
                self.best_ndcg = ndcg5
            self.save_checkpoint(epoch, eval_metrics, is_best)

            self.history.append({
                "epoch":      epoch,
                "train_loss": train_loss,
                **eval_metrics,
            })

            # Early stopping check
            if self._check_early_stopping(ndcg5):
                logger.info(
                    f"\n{'='*60}\n"
                    f"  Early stopping triggered at epoch {epoch}\n"
                    f"  Best NDCG@5: {self.best_ndcg:.4f} (epoch {epoch - self._es_patience})\n"
                    f"  Checkpoint  : {self.config['checkpoint_dir']}/dim_best.pt\n"
                    f"{'='*60}\n"
                )
                stopped_early = True
                break

        if not stopped_early:
            logger.info(f"\n{'='*60}")
            logger.info(f"  Training completed (all {self.config['epochs']} epochs).")
            logger.info(f"  Best NDCG@5: {self.best_ndcg:.4f}")
            logger.info(f"{'='*60}\n")

        # ---------------------------------------------------------------
        # Evaluasi akhir: bandingkan DIM vs MajorityVote baseline
        # Membuktikan DIM belajar lebih dari frequency counting.
        # ---------------------------------------------------------------
        logger.info("\nRunning final baseline comparison...")
        best_epoch_metrics = max(self.history, key=lambda x: x.get("ndcg@5", 0))
        baseline = MajorityVoteBaseline()
        baseline_metrics = baseline.evaluate(
            test_data["hist_playbook_ids"],
            test_data["target_playbook"].tolist(),
            k_values=self.config["k_eval"],
        )
        logger.info("  DIM (best epoch) vs MajorityVote Baseline:")
        for k in self.config["k_eval"]:
            dim_hr   = best_epoch_metrics.get(f"hit_ratio@{k}", 0)
            base_hr  = baseline_metrics.get(f"hit_ratio@{k}", 0)
            dim_ndcg = best_epoch_metrics.get(f"ndcg@{k}", 0)
            base_ndcg= baseline_metrics.get(f"ndcg@{k}", 0)
            logger.info(
                f"  K={k:2d} | HR: DIM={dim_hr:.4f} vs Base={base_hr:.4f} (gap={dim_hr-base_hr:+.4f}) "
                f"| NDCG: DIM={dim_ndcg:.4f} vs Base={base_ndcg:.4f} (gap={dim_ndcg-base_ndcg:+.4f})"
            )

        return self.history

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    trainer = DIMTrainer(CONFIG)
    history = trainer.train()

    # Cetak hasil akhir
    print("\n" + "="*70)
    print("  FINAL EVALUATION - DIM vs MAJORITY VOTE BASELINE")
    print("="*70)
    best = max(history, key=lambda x: x.get("ndcg@5", 0))
    
    print("\n[DIM Model] Best Epoch Results:")
    for k, v in best.items():
        if k not in ("train_loss", "epoch"):
            print(f"  {k}: {v:.4f}")
    
    # ========================================================================
    # TAMBAHAN: Evaluasi Majority Vote Baseline untuk anti-trivial validation
    # ========================================================================
    print("\n[Majority Vote Baseline] Evaluation:")
    print("  (Simple argmax(Counter(history)) - tidak ada learning)\n")
    
    _, test_data, train_data = trainer.load_data()
    baseline = MajorityVoteBaseline()
    baseline_metrics = baseline.evaluate(
        hist_playbook_ids=test_data["hist_playbook_ids"],
        ground_truths=test_data["target_playbook"].tolist(),
        k_values=CONFIG["k_eval"],
    )
    
    for k, v in baseline_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # ========================================================================
    # Perbandingan: DIM vs Baseline
    # ========================================================================
    print("\n" + "="*70)
    print("  COMPARISON TABLE: DIM vs Majority Vote Baseline")
    print("="*70)
    print(f"{'Metric':<20} {'DIM':<15} {'Baseline':<15} {'Gap':<10}")
    print("-"*70)
    
    for k in CONFIG["k_eval"]:
        dim_hr    = best.get(f"hit_ratio@{k}", 0.0)
        base_hr   = baseline_metrics.get(f"hit_ratio@{k}", 0.0)
        gap_hr    = dim_hr - base_hr
        
        dim_ndcg  = best.get(f"ndcg@{k}", 0.0)
        base_ndcg = baseline_metrics.get(f"ndcg@{k}", 0.0)
        gap_ndcg  = dim_ndcg - base_ndcg
        
        print(f"HR@{k:<18} {dim_hr:.4f}          {base_hr:.4f}          {gap_hr:+.4f}")
        print(f"NDCG@{k:<16} {dim_ndcg:.4f}          {base_ndcg:.4f}          {gap_ndcg:+.4f}")

    print("=" * 70)
    print("  Note: Evaluation on synthetic sequences derived from BOTS eventcode distribution.")
    print("  Sequences include 40% phase-shift (multi-stage attack simulation),")
    print("  20% multi-campaign, and 40% standard patterns (signal_ratio=0.70).")
    print("=" * 70)

