"""
=============================================================================
Module 2.4 - Analyst Validation Layer & Continuous Learning
=============================================================================
Human-in-the-loop paradigma untuk eliminasi risiko blackbox automation.

Alur:
    1. DIM menghasilkan probabilitas p untuk setiap kandidat playbook
    2. Analis SOC memberikan keputusan akhir y ∈ {0, 1}
       (1 = konfirmasi, 0 = penolakan)
    3. BCE Loss dihitung: L = -[y log(p) + (1-y) log(1-p)]
    4. Backpropagation memperbarui bobot DIM

Sistem terus beradaptasi terhadap preferensi kontekstual analis.
=============================================================================
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.triage.tfidf_filter import Alert
from src.models.dim import DynamicInterestModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PlaybookRecommendation:
    """Rekomendasi playbook yang dihasilkan DIM."""
    playbook_id:   int
    playbook_name: str
    probability:   float        # p dari DIM ∈ (0,1)
    rank:          int          # ranking dalam top-K

    def __repr__(self):
        return (
            f"<Playbook #{self.rank}: '{self.playbook_name}' "
            f"p={self.probability:.4f}>"
        )


@dataclass
class AnalystDecision:
    """Keputusan analis SOC hasil human-in-the-loop."""
    alert_id:       str
    playbook_id:    int
    decision:       int         # y ∈ {0=reject, 1=confirm}
    timestamp:      float = field(default_factory=time.time)
    analyst_notes:  str   = ""

    @property
    def is_confirmed(self) -> bool:
        return self.decision == 1


@dataclass
class FeedbackRecord:
    """Satu record umpan balik untuk training."""
    alert:          Alert
    recommendation: PlaybookRecommendation
    decision:       AnalystDecision
    bce_loss:       float

    # Tensor input (disimpan untuk replay buffer)
    hist_alert_ids:    Optional[torch.Tensor] = None
    hist_playbook_ids: Optional[torch.Tensor] = None
    hist_tactic_ids:   Optional[torch.Tensor] = None
    hist_severity:     Optional[torch.Tensor] = None
    cand_playbook_id:  Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# BCE Loss Trainer
# ---------------------------------------------------------------------------

class ContinuousLearningTrainer:
    """
    Melatih DIM secara berkelanjutan menggunakan umpan balik analis.

    Loss: L(p, y) = -[y log(p) + (1-y) log(1-p)]

    Menggunakan mini-batch dari replay buffer untuk stabilitas training.
    """

    def __init__(
        self,
        model:         DynamicInterestModel,
        lr:            float = 1e-4,
        weight_decay:  float = 1e-5,
        replay_buffer_size: int = 1000,
        batch_size:    int   = 32,
        min_samples_to_train: int = 16,
        device:        str   = "cpu",
    ):
        self.model    = model.to(device)
        self.device   = device
        self.batch_size = batch_size
        self.min_samples_to_train = min_samples_to_train

        self.criterion = nn.BCELoss()
        self.optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # Replay buffer untuk experience replay
        self.replay_buffer: List[FeedbackRecord] = []
        self.max_buffer_size = replay_buffer_size

        # Statistik training
        self.training_stats = {
            "total_feedback":   0,
            "total_updates":    0,
            "cumulative_loss":  0.0,
            "confirmation_rate": 0.0,
            "confirmations":    0,
            "rejections":       0,
        }

    def add_feedback(self, record: FeedbackRecord):
        """Tambahkan record feedback ke replay buffer."""
        self.replay_buffer.append(record)
        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer.pop(0)  # FIFO

        self.training_stats["total_feedback"] += 1
        if record.decision.is_confirmed:
            self.training_stats["confirmations"] += 1
        else:
            self.training_stats["rejections"] += 1

        total = self.training_stats["confirmations"] + self.training_stats["rejections"]
        if total > 0:
            self.training_stats["confirmation_rate"] = (
                self.training_stats["confirmations"] / total
            )

        # Auto-train jika buffer cukup
        if len(self.replay_buffer) >= self.min_samples_to_train:
            loss = self.train_step()
            if loss is not None:
                logger.debug(f"[ContinuousLearning] BCE Loss = {loss:.6f}")

    def train_step(self) -> Optional[float]:
        """Satu langkah backpropagation dari mini-batch replay buffer."""
        if len(self.replay_buffer) < self.min_samples_to_train:
            return None

        # Sample mini-batch
        import random
        batch_size = min(self.batch_size, len(self.replay_buffer))
        batch = random.sample(self.replay_buffer, batch_size)

        # Filter: hanya record dengan tensor tersedia
        valid = [r for r in batch if r.hist_alert_ids is not None]
        if not valid:
            return None

        self.model.train()
        self.optimizer.zero_grad()

        losses = []
        for record in valid:
            p_tensor = torch.tensor(
                [record.recommendation.probability], dtype=torch.float32,
                device=self.device
            )
            y_tensor = torch.tensor(
                [float(record.decision.decision)], dtype=torch.float32,
                device=self.device
            )
            # Reforward untuk gradient computation
            p = self.model(
                record.hist_alert_ids.to(self.device),
                record.hist_playbook_ids.to(self.device),
                record.hist_tactic_ids.to(self.device),
                record.hist_severity.to(self.device),
                record.cand_playbook_id.to(self.device),
            )
            loss = self.criterion(p, y_tensor)
            losses.append(loss)

        total_loss = torch.stack(losses).mean()
        total_loss.backward()

        # Gradient clipping untuk stabilitas
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        loss_val = total_loss.item()
        self.scheduler.step(loss_val)
        self.training_stats["total_updates"]   += 1
        self.training_stats["cumulative_loss"] += loss_val

        return loss_val

    def get_stats(self) -> Dict:
        stats = self.training_stats.copy()
        if stats["total_updates"] > 0:
            stats["avg_loss"] = stats["cumulative_loss"] / stats["total_updates"]
        else:
            stats["avg_loss"] = 0.0
        return stats


# ---------------------------------------------------------------------------
# Human-in-the-Loop Validator
# ---------------------------------------------------------------------------

class HITLValidator:
    """
    Analyst Validation Layer - Human-in-the-Loop interface.

    Alur kerja:
    1. DIM menghasilkan top-K rekomendasi playbook dengan probabilitas p
    2. HITLValidator mempresentasikan rekomendasi kepada analis
    3. Analis memberikan keputusan y ∈ {0, 1}
    4. FeedbackRecord dibuat dan dikirim ke ContinuousLearningTrainer
    5. Trainer melakukan backpropagation untuk update model

    Dalam mode simulasi, keputusan analis dimodelkan menggunakan
    fungsi probabilistik berbasis ground truth label.
    """

    PLAYBOOK_CATALOG = {
        1:  "Malware Containment & Eradication",
        2:  "Ransomware Response",
        3:  "Phishing Investigation",
        4:  "Lateral Movement Containment",
        5:  "Data Exfiltration Response",
        6:  "DDoS Mitigation",
        7:  "Privilege Escalation Response",
        8:  "Credential Compromise Response",
        9:  "Network Intrusion Response",
        10: "APT Investigation",
        11: "Vulnerability Exploitation Response",
        12: "Insider Threat Investigation",
        13: "Ransomware Negotiation",
        14: "C2 Beaconing Response",
        15: "Zero-Day Exploit Response",
        16: "Port Scan Investigation",
        17: "SQL Injection Response",
        18: "XSS Attack Response",
        19: "Brute Force Response",
        20: "DNS Tunneling Response",
    }

    def __init__(
        self,
        model:    DynamicInterestModel,
        trainer:  ContinuousLearningTrainer,
        top_k:    int  = 5,
        device:   str  = "cpu",
        # Callback untuk mode interaktif (CLI/UI)
        decision_callback: Optional[Callable] = None,
    ):
        self.model             = model
        self.trainer           = trainer
        self.top_k             = top_k
        self.device            = device
        self.decision_callback = decision_callback

        self.session_decisions: List[FeedbackRecord] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend_and_validate(
        self,
        alert:             Alert,
        hist_alert_ids:    torch.Tensor,   # [1, T]
        hist_playbook_ids: torch.Tensor,   # [1, T]
        hist_tactic_ids:   torch.Tensor,   # [1, T]
        hist_severity:     torch.Tensor,   # [1, T]
        ground_truth_playbook: Optional[int] = None,  # untuk simulasi
        analyst_id:        str = "analyst_01",
    ) -> Tuple[List[PlaybookRecommendation], Optional[FeedbackRecord]]:
        """
        Jalankan satu siklus rekomendasi + validasi analis.

        Args:
            alert:                  Alert yang sedang ditangani.
            hist_*:                 Tensor sekuensi historis.
            ground_truth_playbook:  Label benar (mode simulasi).
            analyst_id:             ID analis yang memvalidasi.

        Returns:
            (recommendations, feedback_record)
        """
        # 1) Dapatkan top-K rekomendasi dari DIM
        recommendations = self._get_recommendations(
            hist_alert_ids, hist_playbook_ids, hist_tactic_ids, hist_severity
        )

        # 2) Presentasikan ke analis
        self._present_recommendations(alert, recommendations)

        # 3) Dapatkan keputusan analis
        decision = self._get_analyst_decision(
            alert=alert,
            recommendations=recommendations,
            ground_truth=ground_truth_playbook,
            analyst_id=analyst_id,
        )

        if decision is None:
            return recommendations, None

        # 4) Hitung BCE Loss
        top_rec = recommendations[0]  # Fokus pada rekomendasi teratas
        bce_loss = self._compute_bce_loss(top_rec.probability, decision.decision)

        # 5) Buat FeedbackRecord
        record = FeedbackRecord(
            alert=alert,
            recommendation=top_rec,
            decision=decision,
            bce_loss=bce_loss,
            hist_alert_ids=hist_alert_ids,
            hist_playbook_ids=hist_playbook_ids,
            hist_tactic_ids=hist_tactic_ids,
            hist_severity=hist_severity,
            cand_playbook_id=torch.tensor(
                [top_rec.playbook_id], dtype=torch.long, device=self.device
            ),
        )

        # 6) Kirim ke continuous learning trainer
        self.trainer.add_feedback(record)
        self.session_decisions.append(record)

        logger.info(
            f"[HITL] Alert={alert.alert_id} | "
            f"Playbook='{top_rec.playbook_name}' | "
            f"p={top_rec.probability:.4f} | "
            f"y={decision.decision} | "
            f"BCE={bce_loss:.6f}"
        )
        return recommendations, record

    def get_session_stats(self) -> Dict:
        """Statistik sesi validasi saat ini."""
        if not self.session_decisions:
            return {"total": 0}
        confirmed = sum(1 for r in self.session_decisions if r.decision.is_confirmed)
        rejected  = len(self.session_decisions) - confirmed
        avg_prob  = sum(r.recommendation.probability for r in self.session_decisions) / len(self.session_decisions)
        avg_loss  = sum(r.bce_loss for r in self.session_decisions) / len(self.session_decisions)
        return {
            "total":            len(self.session_decisions),
            "confirmed":        confirmed,
            "rejected":         rejected,
            "confirmation_rate": confirmed / len(self.session_decisions),
            "avg_probability":  avg_prob,
            "avg_bce_loss":     avg_loss,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_recommendations(
        self,
        hist_alert_ids:    torch.Tensor,
        hist_playbook_ids: torch.Tensor,
        hist_tactic_ids:   torch.Tensor,
        hist_severity:     torch.Tensor,
    ) -> List[PlaybookRecommendation]:
        """Dapatkan top-K rekomendasi playbook dari DIM."""
        num_playbooks = max(self.PLAYBOOK_CATALOG.keys())
        top_idx, top_scores = self.model.predict_top_k(
            hist_alert_ids.to(self.device),
            hist_playbook_ids.to(self.device),
            hist_tactic_ids.to(self.device),
            hist_severity.to(self.device),
            num_playbooks=num_playbooks,
            k=self.top_k,
            device=self.device,
        )
        recommendations = []
        for rank in range(top_idx.size(1)):
            pid   = top_idx[0, rank].item()
            score = top_scores[0, rank].item()
            name  = self.PLAYBOOK_CATALOG.get(pid, f"Playbook-{pid}")
            recommendations.append(PlaybookRecommendation(
                playbook_id=pid,
                playbook_name=name,
                probability=score,
                rank=rank + 1,
            ))
        return recommendations

    def _present_recommendations(
        self, alert: Alert, recs: List[PlaybookRecommendation]
    ):
        """Log presentasi rekomendasi kepada analis."""
        logger.info(f"\n{'='*60}")
        logger.info(f"[SOC Alert] {alert}")
        logger.info(f"Recommended Playbooks (top-{self.top_k}):")
        for rec in recs:
            bar = "█" * int(rec.probability * 20)
            logger.info(
                f"  #{rec.rank} [{bar:20s}] {rec.probability:.2%} "
                f"- {rec.playbook_name}"
            )
        logger.info(f"{'='*60}")

    def _get_analyst_decision(
        self,
        alert:           Alert,
        recommendations: List[PlaybookRecommendation],
        ground_truth:    Optional[int],
        analyst_id:      str,
    ) -> Optional[AnalystDecision]:
        """
        Dapatkan keputusan analis.
        - Mode callback: panggil fungsi eksternal (CLI/UI)
        - Mode simulasi: gunakan ground truth dengan noise
        """
        if self.decision_callback is not None:
            decision = self.decision_callback(alert, recommendations)
        else:
            # Mode simulasi: keputusan berdasarkan ground truth
            decision = self._simulate_decision(
                alert, recommendations, ground_truth
            )

        if decision is None:
            return None

        return AnalystDecision(
            alert_id=alert.alert_id,
            playbook_id=recommendations[0].playbook_id,
            decision=decision,
            analyst_notes=f"Validated by {analyst_id}",
        )

    def _simulate_decision(
        self,
        alert:           Alert,
        recommendations: List[PlaybookRecommendation],
        ground_truth:    Optional[int],
    ) -> int:
        """
        Simulasikan keputusan analis berdasarkan ground truth.
        Menambahkan noise probabilistik (10%) untuk realisme.
        """
        import random
        if ground_truth is None:
            return random.choice([0, 1])

        top_playbook_id = recommendations[0].playbook_id
        correct = (top_playbook_id == ground_truth)

        # 10% probability of "analyst error" / noise
        if random.random() < 0.1:
            return 0 if correct else 1
        return 1 if correct else 0

    @staticmethod
    def _compute_bce_loss(p: float, y: int) -> float:
        """
        Binary Cross-Entropy Loss:
            L = -[y log(p) + (1-y) log(1-p)]
        Clip p untuk menghindari log(0).
        """
        import math
        p   = max(min(p, 1 - 1e-7), 1e-7)  # clamp
        return -(y * math.log(p) + (1 - y) * math.log(1 - p))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(42)

    from src.models.dim import DynamicInterestModel

    model = DynamicInterestModel(
        num_alert_types=50, num_playbooks=20, num_tactics=14, embed_dim=64
    )
    trainer   = ContinuousLearningTrainer(model=model, lr=1e-4)
    validator = HITLValidator(model=model, trainer=trainer, top_k=5)

    # Simulasi 10 siklus validasi
    from src.triage.tfidf_filter import Alert
    import random

    for i in range(10):
        alert = Alert(
            alert_id=f"ALT-{i:04d}",
            source_ip="10.0.0.1",
            alert_type="Ransomware Activity",
            severity=5,
            raw_text="Detected ransomware encryption activity"
        )
        B, T = 1, 15
        recs, fb = validator.recommend_and_validate(
            alert=alert,
            hist_alert_ids=torch.randint(1, 51, (B, T)),
            hist_playbook_ids=torch.randint(1, 21, (B, T)),
            hist_tactic_ids=torch.randint(1, 15, (B, T)),
            hist_severity=torch.randint(1, 6, (B, T)),
            ground_truth_playbook=2,  # "Ransomware Response"
        )
        if fb:
            print(f"[{i+1:2d}] BCE Loss: {fb.bce_loss:.6f} | "
                  f"Decision: {'✓' if fb.decision.is_confirmed else '✗'}")

    print("\n=== Session Stats ===")
    for k, v in validator.get_session_stats().items():
        print(f"  {k}: {v}")

    print("\n=== Trainer Stats ===")
    for k, v in trainer.get_stats().items():
        print(f"  {k}: {v}")
