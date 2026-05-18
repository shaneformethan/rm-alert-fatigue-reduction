"""
=============================================================================
Module 2.5 – Evaluation Metrics
=============================================================================
Evaluasi dua ranah performa:

1. Kualitas Prediksi Algoritmik:
   - Klasifikasi: Precision, Recall, F1-Score, ROC-AUC
   - Recommendation Ranking: Hit Ratio, MAP (Mean Average Precision), NDCG

2. Efisiensi Operasional SOC:
   - MTTD (Mean Time To Detect)
   - MTTR (Mean Time To Remediate/Respond)
   - Analyst Workload Reduction
=============================================================================
"""

import time
import logging
import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, classification_report,
    confusion_matrix, average_precision_score,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. Classification Metrics
# ===========================================================================

class ClassificationMetrics:
    """
    Metrik evaluasi klasifikasi:
        - Precision, Recall, F1-Score (micro/macro/weighted)
        - ROC-AUC (One-vs-Rest untuk multiclass)
        - Confusion Matrix
        - Classification Report
    """

    def __init__(self, average: str = "weighted"):
        """
        Args:
            average: strategi averaging ('micro', 'macro', 'weighted', 'binary')
        """
        self.average = average
        self.results_history: List[Dict] = []

    def compute(
        self,
        y_true:      np.ndarray,
        y_pred:      np.ndarray,
        y_prob:      Optional[np.ndarray] = None,
        label_names: Optional[List[str]]  = None,
    ) -> Dict:
        """
        Hitung metrik klasifikasi lengkap.

        Args:
            y_true:      Ground truth labels [N]
            y_pred:      Predicted labels [N]
            y_prob:      Prediction probabilities [N, C] untuk ROC-AUC
            label_names: Nama kelas untuk report

        Returns:
            Dict berisi semua metrik
        """
        metrics = {}

        # Precision, Recall, F1
        metrics["precision"] = precision_score(
            y_true, y_pred, average=self.average, zero_division=0
        )
        metrics["recall"] = recall_score(
            y_true, y_pred, average=self.average, zero_division=0
        )
        metrics["f1_score"] = f1_score(
            y_true, y_pred, average=self.average, zero_division=0
        )

        # Per-class metrics
        metrics["precision_per_class"] = precision_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist()
        metrics["recall_per_class"] = recall_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist()
        metrics["f1_per_class"] = f1_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist()

        # ROC-AUC
        if y_prob is not None:
            try:
                n_classes = y_prob.shape[1] if y_prob.ndim > 1 else 2
                if n_classes == 2:
                    prob_pos = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
                    metrics["roc_auc"] = roc_auc_score(y_true, prob_pos)
                else:
                    metrics["roc_auc"] = roc_auc_score(
                        y_true, y_prob, multi_class="ovr", average=self.average
                    )
            except Exception as e:
                logger.warning(f"ROC-AUC gagal: {e}")
                metrics["roc_auc"] = float("nan")
        else:
            metrics["roc_auc"] = float("nan")

        # Confusion Matrix
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

        # Classification report (string)
        metrics["classification_report"] = classification_report(
            y_true, y_pred,
            target_names=label_names,
            zero_division=0
        )

        self.results_history.append(metrics)
        return metrics

    def print_summary(self, metrics: Dict):
        """Cetak ringkasan metrik."""
        print("\n" + "="*60)
        print("CLASSIFICATION METRICS")
        print("="*60)
        print(f"  Precision  ({self.average}): {metrics['precision']:.4f}")
        print(f"  Recall     ({self.average}): {metrics['recall']:.4f}")
        print(f"  F1-Score   ({self.average}): {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC              : {metrics['roc_auc']:.4f}")
        print("\n" + metrics.get("classification_report", ""))


# ===========================================================================
# 2. Ranking Metrics (Recommendation Quality)
# ===========================================================================

class RankingMetrics:
    """
    Metrik evaluasi kualitas rekomendasi:
        - Hit Ratio @ K  : Proporsi rekomendasi yang memuat item relevan
        - MAP @ K        : Mean Average Precision
        - NDCG @ K       : Normalized Discounted Cumulative Gain

    Format input:
        recommendations: List[List[int]] — top-K playbook ID per query
        ground_truths:   List[int]        — playbook ID yang benar per query
    """

    def compute_all(
        self,
        recommendations: List[List[int]],
        ground_truths:   List[int],
        k_values:        List[int] = [1, 3, 5, 10],
    ) -> Dict:
        """
        Hitung semua ranking metrics untuk berbagai nilai K.

        Args:
            recommendations: Daftar top-K rekomendasi per query
            ground_truths:   Label benar per query
            k_values:        Nilai K yang dievaluasi

        Returns:
            Dict berisi metrik per K
        """
        assert len(recommendations) == len(ground_truths), \
            "Jumlah rekomendasi dan ground truth harus sama"

        results = {}
        for k in k_values:
            results[f"hit_ratio@{k}"] = self.hit_ratio_at_k(
                recommendations, ground_truths, k
            )
            results[f"map@{k}"] = self.map_at_k(
                recommendations, ground_truths, k
            )
            results[f"ndcg@{k}"] = self.ndcg_at_k(
                recommendations, ground_truths, k
            )
        return results

    @staticmethod
    def hit_ratio_at_k(
        recommendations: List[List[int]],
        ground_truths:   List[int],
        k:               int,
    ) -> float:
        """
        Hit Ratio @ K:
            HR@K = (1/N) Σ 1[gt_i ∈ top-K_i]

        Proporsi query di mana item relevan ada dalam top-K rekomendasi.
        """
        hits = sum(
            1 for rec, gt in zip(recommendations, ground_truths)
            if gt in rec[:k]
        )
        return hits / len(ground_truths) if ground_truths else 0.0

    @staticmethod
    def map_at_k(
        recommendations: List[List[int]],
        ground_truths:   List[int],
        k:               int,
    ) -> float:
        """
        MAP @ K (Mean Average Precision):
            AP@K = (1/min(K, |rel|)) Σ_{j=1}^{K} P(j) × rel(j)
            MAP@K = (1/N) Σ AP@K_i

        Di mana P(j) = precision at position j,
                rel(j) = 1 jika item pada posisi j relevan, else 0.
        """
        ap_scores = []
        for rec, gt in zip(recommendations, ground_truths):
            top_k = rec[:k]
            hits  = 0
            prec_sum = 0.0
            for j, item in enumerate(top_k, 1):
                if item == gt:
                    hits += 1
                    prec_sum += hits / j
            ap = prec_sum / min(k, 1)  # normalize by min(K, |relevant|)
            ap_scores.append(ap)
        return float(np.mean(ap_scores)) if ap_scores else 0.0

    @staticmethod
    def ndcg_at_k(
        recommendations: List[List[int]],
        ground_truths:   List[int],
        k:               int,
    ) -> float:
        """
        NDCG @ K (Normalized Discounted Cumulative Gain):
            DCG@K  = Σ_{j=1}^{K} rel(j) / log2(j+1)
            IDCG@K = 1 / log2(2) = 1    (ideal: relevant item at pos 1)
            NDCG@K = DCG@K / IDCG@K

        Untuk binary relevance (satu item relevan per query):
            IDCG@K = 1/log2(2) = 1
        """
        ndcg_scores = []
        ideal_dcg = 1.0 / math.log2(2)  # item relevan di posisi 1

        for rec, gt in zip(recommendations, ground_truths):
            top_k = rec[:k]
            dcg = 0.0
            for j, item in enumerate(top_k, 1):
                if item == gt:
                    dcg += 1.0 / math.log2(j + 1)
                    break  # Satu item relevan, hentikan
            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
            ndcg_scores.append(ndcg)

        return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

    def print_summary(self, results: Dict):
        """Cetak ringkasan ranking metrics."""
        print("\n" + "="*60)
        print("RECOMMENDATION RANKING METRICS")
        print("="*60)
        # Kelompokkan per K
        from itertools import groupby
        k_values = sorted(set(
            int(key.split("@")[1]) for key in results if "@" in key
        ))
        for k in k_values:
            hr   = results.get(f"hit_ratio@{k}", 0)
            map_ = results.get(f"map@{k}", 0)
            ndcg = results.get(f"ndcg@{k}", 0)
            print(f"  K={k:2d} | HR@K={hr:.4f} | MAP@K={map_:.4f} | NDCG@K={ndcg:.4f}")


# ===========================================================================
# 3. SOC Operational Efficiency Metrics
# ===========================================================================

@dataclass
class IncidentRecord:
    """Rekaman satu insiden untuk pengukuran efisiensi operasional."""
    incident_id:       str
    detection_start:   float   # timestamp mulai proses algoritma
    detection_end:     float   # timestamp alert ditetapkan sebagai insiden
    response_start:    float   # timestamp DIM mulai proses
    response_end:      float   # timestamp remediasi disetujui analis
    manual_actions_automated: int = 0   # aksi manual yang diotomasi
    manual_actions_total:     int = 0   # total aksi yang sebelumnya dilakukan manual


class SOCOperationalMetrics:
    """
    Metrik efisiensi operasional SOC:
        - MTTD: Mean Time To Detect
        - MTTR: Mean Time To Respond/Remediate
        - Analyst Workload Reduction
    """

    def __init__(self):
        self.incidents: List[IncidentRecord] = []

    def record_incident(self, record: IncidentRecord):
        """Tambahkan rekaman insiden."""
        self.incidents.append(record)

    def compute(self) -> Dict:
        """
        Hitung semua metrik operasional SOC.

        Returns:
            Dict berisi MTTD, MTTR, Analyst Workload Reduction
        """
        if not self.incidents:
            return {"error": "Tidak ada data insiden"}

        # MTTD: rata-rata durasi komputasi algoritma
        mttd_values = [
            r.detection_end - r.detection_start
            for r in self.incidents
            if r.detection_end > r.detection_start
        ]

        # MTTR: rata-rata durasi proses DIM hingga remediasi disetujui
        mttr_values = [
            r.response_end - r.response_start
            for r in self.incidents
            if r.response_end > r.response_start
        ]

        # Analyst Workload Reduction
        workload_reductions = []
        for r in self.incidents:
            if r.manual_actions_total > 0:
                reduction = r.manual_actions_automated / r.manual_actions_total
                workload_reductions.append(reduction)

        results = {
            "mttd_seconds": {
                "mean":   float(np.mean(mttd_values))   if mttd_values else 0.0,
                "median": float(np.median(mttd_values)) if mttd_values else 0.0,
                "std":    float(np.std(mttd_values))    if mttd_values else 0.0,
                "min":    float(np.min(mttd_values))    if mttd_values else 0.0,
                "max":    float(np.max(mttd_values))    if mttd_values else 0.0,
            },
            "mttr_seconds": {
                "mean":   float(np.mean(mttr_values))   if mttr_values else 0.0,
                "median": float(np.median(mttr_values)) if mttr_values else 0.0,
                "std":    float(np.std(mttr_values))    if mttr_values else 0.0,
                "min":    float(np.min(mttr_values))    if mttr_values else 0.0,
                "max":    float(np.max(mttr_values))    if mttr_values else 0.0,
            },
            "analyst_workload_reduction": {
                "mean_pct":   float(np.mean(workload_reductions) * 100)   if workload_reductions else 0.0,
                "median_pct": float(np.median(workload_reductions) * 100) if workload_reductions else 0.0,
                "n_incidents": len(self.incidents),
            },
        }
        return results

    def print_summary(self, results: Optional[Dict] = None):
        """Cetak ringkasan metrik operasional."""
        if results is None:
            results = self.compute()

        print("\n" + "="*60)
        print("SOC OPERATIONAL EFFICIENCY METRICS")
        print("="*60)

        mttd = results.get("mttd_seconds", {})
        mttr = results.get("mttr_seconds", {})
        wlr  = results.get("analyst_workload_reduction", {})

        print(f"\n  MTTD (Mean Time To Detect):")
        print(f"    Mean   : {mttd.get('mean', 0):.2f}s")
        print(f"    Median : {mttd.get('median', 0):.2f}s")
        print(f"    Std    : {mttd.get('std', 0):.2f}s")
        print(f"    Range  : [{mttd.get('min', 0):.2f}s – {mttd.get('max', 0):.2f}s]")

        print(f"\n  MTTR (Mean Time To Respond):")
        print(f"    Mean   : {mttr.get('mean', 0):.2f}s")
        print(f"    Median : {mttr.get('median', 0):.2f}s")
        print(f"    Std    : {mttr.get('std', 0):.2f}s")
        print(f"    Range  : [{mttr.get('min', 0):.2f}s – {mttr.get('max', 0):.2f}s]")

        print(f"\n  Analyst Workload Reduction:")
        print(f"    Mean   : {wlr.get('mean_pct', 0):.1f}%")
        print(f"    Median : {wlr.get('median_pct', 0):.1f}%")
        print(f"    N Incidents: {wlr.get('n_incidents', 0)}")


# ===========================================================================
# Consolidated Evaluator
# ===========================================================================

class SystemEvaluator:
    """
    Evaluator terintegrasi untuk seluruh framework.
    Menggabungkan klasifikasi, ranking, dan metrik operasional.
    """

    def __init__(self, average: str = "weighted"):
        self.clf_metrics  = ClassificationMetrics(average=average)
        self.rank_metrics = RankingMetrics()
        self.soc_metrics  = SOCOperationalMetrics()

    def evaluate_classification(
        self,
        y_true:      np.ndarray,
        y_pred:      np.ndarray,
        y_prob:      Optional[np.ndarray] = None,
        label_names: Optional[List[str]]  = None,
        verbose:     bool = True,
    ) -> Dict:
        results = self.clf_metrics.compute(y_true, y_pred, y_prob, label_names)
        if verbose:
            self.clf_metrics.print_summary(results)
        return results

    def evaluate_ranking(
        self,
        recommendations: List[List[int]],
        ground_truths:   List[int],
        k_values:        List[int] = [1, 3, 5, 10],
        verbose:         bool = True,
    ) -> Dict:
        results = self.rank_metrics.compute_all(recommendations, ground_truths, k_values)
        if verbose:
            self.rank_metrics.print_summary(results)
        return results

    def evaluate_soc_efficiency(
        self,
        incidents: Optional[List[IncidentRecord]] = None,
        verbose:   bool = True,
    ) -> Dict:
        if incidents:
            for inc in incidents:
                self.soc_metrics.record_incident(inc)
        results = self.soc_metrics.compute()
        if verbose:
            self.soc_metrics.print_summary(results)
        return results

    def full_evaluation(
        self,
        # Classification
        y_true: np.ndarray, y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        label_names: Optional[List[str]] = None,
        # Ranking
        recommendations: Optional[List[List[int]]] = None,
        ground_truths:   Optional[List[int]]       = None,
        # SOC
        incidents: Optional[List[IncidentRecord]] = None,
    ) -> Dict:
        """Jalankan evaluasi lengkap semua ranah performa."""
        print("\n" + "█"*60)
        print("  FULL SYSTEM EVALUATION REPORT")
        print("█"*60)

        report = {}
        report["classification"] = self.evaluate_classification(
            y_true, y_pred, y_prob, label_names, verbose=True
        )

        if recommendations and ground_truths:
            report["ranking"] = self.evaluate_ranking(
                recommendations, ground_truths, verbose=True
            )

        if incidents:
            report["soc_efficiency"] = self.evaluate_soc_efficiency(
                incidents, verbose=True
            )

        return report


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)

    # Dummy classification
    n = 200
    n_classes = 5
    y_true = np.random.randint(0, n_classes, n)
    y_pred = np.where(np.random.random(n) > 0.3, y_true,
                      np.random.randint(0, n_classes, n))
    y_prob = np.random.dirichlet(np.ones(n_classes), size=n)

    # Dummy ranking
    n_queries = 50
    num_playbooks = 20
    recommendations = [
        list(np.random.choice(range(1, num_playbooks+1), 10, replace=False))
        for _ in range(n_queries)
    ]
    ground_truths = np.random.randint(1, num_playbooks+1, n_queries).tolist()

    # Dummy SOC incidents
    incidents = []
    for i in range(20):
        t0 = time.time()
        incidents.append(IncidentRecord(
            incident_id=f"INC-{i:04d}",
            detection_start=t0,
            detection_end=t0 + np.random.uniform(0.5, 5.0),
            response_start=t0 + np.random.uniform(5, 10),
            response_end=t0 + np.random.uniform(15, 60),
            manual_actions_automated=np.random.randint(3, 10),
            manual_actions_total=10,
        ))

    evaluator = SystemEvaluator()
    evaluator.full_evaluation(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        label_names=[f"playbook_{i}" for i in range(n_classes)],
        recommendations=recommendations,
        ground_truths=ground_truths,
        incidents=incidents,
    )
