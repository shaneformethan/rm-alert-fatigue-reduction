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
# 2b. Majority Vote Baseline (untuk validasi DIM)
# ===========================================================================

class MajorityVoteBaseline:
    """
    Baseline non-learning untuk perbandingan dengan DIM.

    Merekomendasikan playbook berdasarkan frekuensi kemunculan di riwayat
    historis (argmax Counter). Digunakan untuk membuktikan bahwa DIM belajar
    lebih dari sekedar statistik frekuensi sederhana.

    Secara khusus, pada Phase-Shift sequences (20% dataset), baseline ini
    selalu salah karena target = late_pb yang muncul lebih sedikit secara
    keseluruhan — hanya sequential model (LSTM) yang bisa menjawab dengan benar.
    """

    def predict_top_k(
        self,
        hist_playbook_ids: "torch.Tensor",
        k: int = 5,
    ) -> List[List[int]]:
        """
        Prediksi top-K playbook berdasarkan frekuensi di history.

        Args:
            hist_playbook_ids: Tensor [B, T] riwayat playbook ID
            k: Jumlah rekomendasi

        Returns:
            List of List[int] — top-K playbook per query
        """
        from collections import Counter
        recommendations = []
        for seq in hist_playbook_ids:
            counts = Counter(int(x) for x in seq if int(x) != 0)  # hapus padding
            top_k  = [pb for pb, _ in counts.most_common(k)]
            # Pad jika hist terlalu pendek
            all_pbs = list(counts.keys()) + list(range(1, 20))
            i = 0
            while len(top_k) < k:
                if all_pbs[i] not in top_k:
                    top_k.append(all_pbs[i])
                i += 1
            recommendations.append(top_k[:k])
        return recommendations

    def evaluate(
        self,
        hist_playbook_ids: "torch.Tensor",
        ground_truths:    List[int],
        k_values:         List[int] = [1, 3, 5, 10],
    ) -> Dict:
        """Evaluasi baseline dan return ranking metrics."""
        k_max = max(k_values)
        recs  = self.predict_top_k(hist_playbook_ids, k=k_max)
        rm    = RankingMetrics()
        return rm.compute_all(recs, ground_truths, k_values)

    def compare_with_dim(
        self,
        dim_metrics:      Dict,
        baseline_metrics: Dict,
        k_values:         List[int] = [1, 3, 5, 10],
    ) -> None:
        """Cetak tabel perbandingan DIM vs MajorityVote baseline."""
        print("\n" + "="*65)
        print("  DIM vs MAJORITY VOTE BASELINE (Anti-Trivial Validation)")
        print("="*65)
        print(f"  {'K':<4} {'Metric':<8} {'MajorityVote':>13} {'DIM':>8} {'Gap':>8} {'Status':>8}")
        print(f"  {'-'*4} {'-'*8} {'-'*13} {'-'*8} {'-'*8} {'-'*8}")
        for k in k_values:
            for metric in ["hit_ratio", "ndcg"]:
                key = f"{metric}@{k}"
                base_val = baseline_metrics.get(key, 0)
                dim_val  = dim_metrics.get(key, 0)
                gap      = dim_val - base_val
                status   = "PASS" if gap >= 0 else "FAIL"
                print(
                    f"  {k:<4} {metric:<8} {base_val:>13.4f} {dim_val:>8.4f} "
                    f"{gap:>+8.4f} {status:>8}"
                )
        print(
            "\n  Keterangan: Gap > 0 berarti DIM belajar melebihi frequency counting.\n"
            "  Pada Phase-Shift sequences (20%), majority vote SELALU gagal di K=1\n"
            "  karena target = late_pb yang bukan yang paling sering di hist.\n"
            "="*65
        )


# ===========================================================================
# 3. SOC Operational Efficiency Metrics
# ===========================================================================

@dataclass
class IncidentRecord:
    """
    Rekaman satu insiden untuk pengukuran efisiensi operasional SOC.

    Field waktu:
        detection_start/end : Durasi komputasi TF-IDF filter hingga alert ditetapkan → MTTD
        response_start/end  : Durasi DIM inference hingga playbook disetujui analis → MTTR

    Field otomasi:
        manual_actions_automated : Jumlah langkah manual yang diotomasi sistem.
            Komponen yang diotomasi: NER extraction, KG mapping, TF-IDF triage,
            DIM inference, playbook scoring, dan eksekusi awal via SOAR.
            Langkah yang tetap manual: HITL Confirm/Reject akhir (1 dari N langkah).
        manual_actions_total : Total langkah yang sebelumnya dilakukan manual.
    """
    incident_id:       str
    detection_start:   float   # timestamp mulai proses algoritma TF-IDF
    detection_end:     float   # timestamp alert ditetapkan sebagai insiden high-risk
    response_start:    float   # timestamp DIM mulai proses rekomendasi
    response_end:      float   # timestamp remediasi disetujui analis (HITL)
    manual_actions_automated: int = 0
    manual_actions_total:     int = 0


class SOCOperationalMetrics:
    """
    Metrik efisiensi operasional SOC:
        - MTTD (Mean Time To Detect)       → bandingkan vs MTTD manual dan [10]
        - MTTR (Mean Time To Respond)       → bandingkan vs MTTR manual dan [13][14]
        - Analyst Workload Reduction (%)    → % langkah manual yang diotomasi
        - MTTD Reduction (%) vs baseline    → improvement terhadap SOC manual
        - MTTR Reduction (%) vs baseline    → improvement terhadap SOC manual

    Baseline Reference (dari literatur SOC):
        MTTD baseline  ≈ 180s  (3 menit — investigasi manual awal per alert)
        MTTR baseline  ≈ 480s  (8 menit — triage + response manual sederhana)
        Sumber: IBM Cost of a Data Breach Report; Palo Alto Unit 42 Incident Response Report.

    Pembanding Existing Methods:
        [10] AI-based SIEM+SOAR  : MTTD 3.2s, response accuracy 84%
        [13][14] Validated SOAR  : MTTR reduction 81%, isolation 9.05s
    """

    # Baseline SOC manual dari literatur
    BASELINE_MTTD_SECONDS: float = 180.0   # 3 menit investigasi manual awal
    BASELINE_MTTR_SECONDS: float = 480.0   # 8 menit triage + response manual

    def __init__(
        self,
        baseline_mttd_seconds: float = 180.0,
        baseline_mttr_seconds: float = 480.0,
    ):
        """
        Args:
            baseline_mttd_seconds: MTTD baseline SOC manual (default 180s = 3 menit).
            baseline_mttr_seconds: MTTR baseline SOC manual (default 480s = 8 menit).
                Sesuaikan dengan data historis SOC jika tersedia.
        """
        self.incidents: List[IncidentRecord] = []
        self.baseline_mttd = baseline_mttd_seconds
        self.baseline_mttr = baseline_mttr_seconds

    def record_incident(self, record: IncidentRecord):
        """Tambahkan rekaman insiden."""
        self.incidents.append(record)

    def compute(self) -> Dict:
        """
        Hitung semua metrik operasional SOC, termasuk % reduction vs baseline.

        Returns:
            Dict berisi MTTD, MTTR, Workload Reduction, dan % improvement vs baseline.
        """
        if not self.incidents:
            return {"error": "Tidak ada data insiden"}

        # MTTD: durasi komputasi TF-IDF filter (detection_start -> detection_end)
        mttd_values = [
            r.detection_end - r.detection_start
            for r in self.incidents
            if r.detection_end > r.detection_start
        ]

        # MTTR: durasi DIM inference hingga playbook disetujui (response_start -> response_end)
        mttr_values = [
            r.response_end - r.response_start
            for r in self.incidents
            if r.response_end > r.response_start
        ]

        # Analyst Workload Reduction: % langkah manual yang berhasil diotomasi
        workload_reductions = []
        for r in self.incidents:
            if r.manual_actions_total > 0:
                reduction = r.manual_actions_automated / r.manual_actions_total
                workload_reductions.append(reduction)

        mean_mttd = float(np.mean(mttd_values))   if mttd_values else 0.0
        mean_mttr = float(np.mean(mttr_values))   if mttr_values else 0.0
        mean_wlr  = float(np.mean(workload_reductions) * 100) if workload_reductions else 0.0

        # % reduction vs baseline manual SOC
        mttd_reduction_pct = (
            (self.baseline_mttd - mean_mttd) / self.baseline_mttd * 100
            if mean_mttd < self.baseline_mttd else 0.0
        )
        mttr_reduction_pct = (
            (self.baseline_mttr - mean_mttr) / self.baseline_mttr * 100
            if mean_mttr < self.baseline_mttr else 0.0
        )

        results = {
            "mttd_seconds": {
                "mean":   mean_mttd,
                "median": float(np.median(mttd_values)) if mttd_values else 0.0,
                "std":    float(np.std(mttd_values))    if mttd_values else 0.0,
                "min":    float(np.min(mttd_values))    if mttd_values else 0.0,
                "max":    float(np.max(mttd_values))    if mttd_values else 0.0,
            },
            "mttr_seconds": {
                "mean":   mean_mttr,
                "median": float(np.median(mttr_values)) if mttr_values else 0.0,
                "std":    float(np.std(mttr_values))    if mttr_values else 0.0,
                "min":    float(np.min(mttr_values))    if mttr_values else 0.0,
                "max":    float(np.max(mttr_values))    if mttr_values else 0.0,
            },
            "analyst_workload_reduction": {
                "mean_pct":    mean_wlr,
                "median_pct":  float(np.median(workload_reductions) * 100) if workload_reductions else 0.0,
                "n_incidents": len(self.incidents),
            },
            # Improvement vs baseline manual SOC (dari literatur)
            "vs_baseline": {
                "baseline_mttd_s":    self.baseline_mttd,
                "baseline_mttr_s":    self.baseline_mttr,
                "mttd_reduction_pct": round(mttd_reduction_pct, 2),
                "mttr_reduction_pct": round(mttr_reduction_pct, 2),
                "note": (
                    f"Baseline: MTTD={self.baseline_mttd}s, MTTR={self.baseline_mttr}s "
                    "(manual SOC investigation). "
                    "Sumber: IBM Cost of a Data Breach Report; Palo Alto Unit 42."
                ),
            },
        }
        return results

    def print_summary(self, results: Optional[Dict] = None):
        """Cetak ringkasan metrik operasional dengan tabel perbandingan vs existing methods."""
        if results is None:
            results = self.compute()

        mttd    = results.get("mttd_seconds", {})
        mttr    = results.get("mttr_seconds", {})
        wlr     = results.get("analyst_workload_reduction", {})
        vs_base = results.get("vs_baseline", {})

        print("\n" + "="*68)
        print("  SOC OPERATIONAL EFFICIENCY - SYSTEM vs EXISTING METHODS")
        print("="*68)

        # Tabel perbandingan langsung
        hdr = f"  {'Metrik':<26} {'Baseline Manual':>14} {'[10]':>8} {'[13][14]':>9} {'SISTEM INI':>11}"
        print(hdr)
        print(f"  {'-'*26} {'-'*14} {'-'*8} {'-'*9} {'-'*11}")
        sys_mttd = mttd.get('mean', 0)
        sys_mttr = mttr.get('mean', 0)
        sys_wlr  = wlr.get('mean_pct', 0)
        sys_mttd_red = vs_base.get('mttd_reduction_pct', 0)
        sys_mttr_red = vs_base.get('mttr_reduction_pct', 0)
        print(f"  {'MTTD (mean, detik)':<26} {vs_base.get('baseline_mttd_s', 180):>13.0f}s {'3.2s':>8} {'--':>9} {sys_mttd:>10.2f}s")
        print(f"  {'MTTD Reduction (%)':<26} {'--':>14} {'--':>8} {'--':>9} {sys_mttd_red:>10.1f}%")
        print(f"  {'MTTR (mean, detik)':<26} {vs_base.get('baseline_mttr_s', 480):>13.0f}s {'--':>8} {'9.05s':>9} {sys_mttr:>10.2f}s")
        print(f"  {'MTTR Reduction (%)':<26} {'--':>14} {'--':>8} {'81%':>9} {sys_mttr_red:>10.1f}%")
        print(f"  {'Workload Reduction (%)':<26} {'0% (semua manual)':>14} {'--':>8} {'--':>9} {sys_wlr:>10.1f}%")

        print(f"\n  Detail Sistem Ini:")
        print(f"    MTTD : Mean={sys_mttd:.2f}s | Median={mttd.get('median',0):.2f}s"
              f" | Range=[{mttd.get('min',0):.2f}s-{mttd.get('max',0):.2f}s]")
        print(f"    MTTR : Mean={sys_mttr:.2f}s | Median={mttr.get('median',0):.2f}s"
              f" | Range=[{mttr.get('min',0):.2f}s-{mttr.get('max',0):.2f}s]")
        print(f"    Workload Reduction : Mean={sys_wlr:.1f}% | N={wlr.get('n_incidents',0)} insiden")
        print(f"\n  Baseline: {vs_base.get('note', '')}")
        print("="*68)


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

    # SOC incidents — simulasi sistem yang mengotomasi 80-100% langkah manual
    # Justifikasi: sistem mengotomasi NER, KG, TF-IDF, DIM, scoring, SOAR execution.
    # Hanya satu langkah yang tetap manual: HITL Confirm/Reject akhir.
    incidents = []
    for i in range(50):
        t0 = time.time()
        incidents.append(IncidentRecord(
            incident_id=f"INC-{i:04d}",
            detection_start=t0,
            detection_end=t0 + np.random.uniform(0.1, 3.0),    # MTTD: 0.1–3s (TF-IDF komputasi)
            response_start=t0 + np.random.uniform(3, 8),
            response_end=t0 + np.random.uniform(8, 35),         # MTTR: 8–35s (DIM+HITL)
            manual_actions_automated=np.random.randint(8, 11),  # 8-10 dari 10 diotomasi
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
