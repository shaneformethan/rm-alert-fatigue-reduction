"""
=============================================================================
Module 2.2 – Mathematical Formulation of Automated Triage
=============================================================================
Filter TF-IDF berbasis probabilitas untuk mereduksi alert fatigue.

Formulasi:
    TF-IDF(i, j) = TF(i, j) × log( N / (df_i + 1) )

    Di mana:
        N     = total jumlah alert dalam time window
        df_i  = frekuensi dokumen yang mengandung alert i
        TF(i,j) = frekuensi alert i dalam set j

Kriteria eliminasi (diklasifikasikan sebagai noise/false positive):
    IDF(i) > theta_max_idf   AND
    TF-IDF(i,j) < theta_tfidf

    Peringatan yang lolos filter → diteruskan ke DIM pipeline.
=============================================================================
"""

import math
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Representasi satu security alert."""
    alert_id:   str
    source_ip:  str
    alert_type: str            # tipe peringatan (e.g., "Port Scan", "SQL Injection")
    severity:   int            # 1-5
    raw_text:   str            # deskripsi peringatan
    timestamp:  float = field(default_factory=time.time)
    features:   Dict  = field(default_factory=dict)  # fitur numerik tambahan

    def __repr__(self):
        return f"<Alert [{self.alert_id}] type={self.alert_type!r} sev={self.severity}>"


@dataclass
class TriageResult:
    """Hasil triase satu alert."""
    alert:          Alert
    tfidf_score:    float
    idf_score:      float
    is_noise:       bool          # True = dieliminasi, False = lolos ke DIM
    reason:         str = ""      # alasan keputusan

    def __repr__(self):
        status = "NOISE" if self.is_noise else "HIGH-RISK"
        return (
            f"<TriageResult [{self.alert.alert_id}] "
            f"status={status} tfidf={self.tfidf_score:.4f} idf={self.idf_score:.4f}>"
        )


class TFIDFTriageFilter:
    """
    Filter triase berbasis modifikasi matriks probabilitas TF-IDF.

    Parameter threshold:
        theta_max_idf  : ambang batas maksimum IDF — alert dengan IDF tinggi
                         berarti sangat jarang muncul (kemungkinan noise spesifik)
        theta_tfidf    : ambang batas minimum skor TF-IDF — skor rendah
                         menunjukkan alert tidak cukup informatif dalam konteks

    Sebuah alert diklasifikasikan sebagai noise jika KEDUA kondisi terpenuhi:
        IDF(i) > theta_max_idf  AND  TF-IDF(i,j) < theta_tfidf

    Time window mengumpulkan N alert sebelum batch evaluasi dilakukan.
    """

    def __init__(
        self,
        theta_max_idf:   float = 6.0,
        theta_tfidf:     float = 0.05,
        time_window_sec: int   = 300,     # 5 menit default
        min_alerts_batch: int  = 10,      # minimum alert untuk kalkulasi valid
    ):
        """
        Args:
            theta_max_idf:    Ambang batas IDF maksimum (nilai > threshold = candidate noise).
            theta_tfidf:      Ambang batas TF-IDF minimum (nilai < threshold = low relevance).
            time_window_sec:  Durasi time window dalam detik.
            min_alerts_batch: Jumlah minimum alert dalam satu batch untuk evaluasi.
        """
        self.theta_max_idf    = theta_max_idf
        self.theta_tfidf      = theta_tfidf
        self.time_window_sec  = time_window_sec
        self.min_alerts_batch = min_alerts_batch

        # Internal state
        self._alert_queue:   List[Alert] = []
        self._window_start:  float       = time.time()
        self._df_counter:    Counter     = Counter()  # document frequency per alert_type
        self._N:             int         = 0          # total alert count in window

        # Statistik
        self.stats = {
            "total_processed": 0,
            "noise_eliminated": 0,
            "high_risk_forwarded": 0,
            "batches_evaluated": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_alert(self, alert: Alert) -> Optional[TriageResult]:
        """
        Tambahkan satu alert ke queue.

        Returns:
            TriageResult jika evaluasi batch dipicu, None jika masih mengumpulkan.
        """
        self._alert_queue.append(alert)
        self._df_counter[alert.alert_type] += 1
        self._N += 1
        self.stats["total_processed"] += 1

        # Evaluasi batch jika window penuh
        elapsed = time.time() - self._window_start
        if elapsed >= self.time_window_sec or len(self._alert_queue) >= self.min_alerts_batch:
            return self._evaluate_single(alert)
        return None

    def evaluate_batch(self, alerts: List[Alert]) -> Tuple[List[TriageResult], List[Alert]]:
        """
        Evaluasi batch alert sekaligus.

        Args:
            alerts: List alert dalam satu time window.

        Returns:
            Tuple (semua TriageResult, high-risk alerts yang lolos ke DIM)
        """
        if not alerts:
            return [], []

        N = len(alerts)
        # Hitung document frequency per alert_type
        df: Dict[str, int] = {}
        for alert in alerts:
            df[alert.alert_type] = df.get(alert.alert_type, 0) + 1

        # Hitung TF per (alert_type, position) — disederhanakan per alert_type dalam batch
        tf_counter: Counter = Counter(a.alert_type for a in alerts)

        results       = []
        high_risk     = []

        for alert in alerts:
            tfidf, idf = self._compute_tfidf(
                alert_type=alert.alert_type,
                tf=tf_counter[alert.alert_type] / N,
                df=df[alert.alert_type],
                N=N
            )
            is_noise = self._classify_noise(idf, tfidf)
            reason   = self._get_reason(idf, tfidf, is_noise)

            result = TriageResult(
                alert=alert,
                tfidf_score=tfidf,
                idf_score=idf,
                is_noise=is_noise,
                reason=reason
            )
            results.append(result)

            if is_noise:
                self.stats["noise_eliminated"] += 1
                logger.debug(f"[NOISE] {alert}: {reason}")
            else:
                self.stats["high_risk_forwarded"] += 1
                high_risk.append(alert)

        self.stats["batches_evaluated"] += 1
        self._reset_window()

        logger.info(
            f"[Triage Batch] N={N} | "
            f"noise={self.stats['noise_eliminated']} | "
            f"high-risk={self.stats['high_risk_forwarded']}"
        )
        return results, high_risk

    def compute_tfidf_matrix(
        self, alerts: List[Alert]
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Hitung TF-IDF matrix lengkap untuk visualisasi/analisis.

        Returns:
            (matrix [N_types × N_alerts], alert_types, alert_ids)
        """
        alert_types = list(set(a.alert_type for a in alerts))
        alert_ids   = [a.alert_id for a in alerts]
        N           = len(alerts)

        # Document frequency
        df: Dict[str, int] = {}
        for a in alerts:
            df[a.alert_type] = df.get(a.alert_type, 0) + 1

        matrix = np.zeros((len(alert_types), N))

        for j, alert in enumerate(alerts):
            # TF(i, j) = frekuensi tipe i dalam subset j (disederhanakan)
            tf = 1.0 / N  # uniform TF per alert

            for i, atype in enumerate(alert_types):
                if alert.alert_type == atype:
                    _, idf = self._compute_tfidf(atype, tf, df.get(atype, 0), N)
                    matrix[i][j] = tf * idf

        return matrix, alert_types, alert_ids

    def get_stats(self) -> Dict:
        return self.stats.copy()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_tfidf(
        self,
        alert_type: str,
        tf: float,
        df: int,
        N: int
    ) -> Tuple[float, float]:
        """
        Formulasi TF-IDF:
            IDF(i)      = log( N / (df_i + 1) )
            TF-IDF(i,j) = TF(i,j) × IDF(i)

        Penambahan +1 pada penyebut mencegah ZeroDivisionError saat alert baru.
        """
        idf   = math.log(N / (df + 1))
        tfidf = tf * idf
        return tfidf, idf

    def _classify_noise(self, idf: float, tfidf: float) -> bool:
        """
        Kriteria eliminasi:
            IDF(i) > theta_max_idf  AND  TF-IDF(i,j) < theta_tfidf
        """
        return idf > self.theta_max_idf and tfidf < self.theta_tfidf

    def _get_reason(self, idf: float, tfidf: float, is_noise: bool) -> str:
        if is_noise:
            return (
                f"IDF={idf:.3f} > θ_idf={self.theta_max_idf} "
                f"AND TF-IDF={tfidf:.3f} < θ_tfidf={self.theta_tfidf}"
            )
        return (
            f"IDF={idf:.3f} | TF-IDF={tfidf:.3f} — lolos threshold"
        )

    def _evaluate_single(self, alert: Alert) -> TriageResult:
        """Evaluasi single alert berdasarkan state window saat ini."""
        tf    = self._df_counter[alert.alert_type] / max(self._N, 1)
        df    = self._df_counter[alert.alert_type]
        N     = max(self._N, 1)
        tfidf, idf = self._compute_tfidf(alert.alert_type, tf, df, N)
        is_noise   = self._classify_noise(idf, tfidf)
        return TriageResult(
            alert=alert,
            tfidf_score=tfidf,
            idf_score=idf,
            is_noise=is_noise,
            reason=self._get_reason(idf, tfidf, is_noise)
        )

    def _reset_window(self):
        self._alert_queue  = []
        self._df_counter   = Counter()
        self._N            = 0
        self._window_start = time.time()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    import random, string

    def rand_id():
        return "ALT-" + "".join(random.choices(string.digits, k=6))

    # Simulasi 30 alert dengan distribusi tidak merata
    alert_types = (
        ["Port Scan"] * 15 +          # sangat sering → TF tinggi
        ["SQL Injection"] * 5 +
        ["Brute Force"] * 5 +
        ["DNS Exfiltration"] * 2 +    # jarang → IDF tinggi
        ["Zero-Day Exploit"] * 1 +    # sangat jarang
        ["Ransomware Activity"] * 2
    )
    random.shuffle(alert_types)

    alerts = [
        Alert(
            alert_id=rand_id(),
            source_ip=f"10.0.0.{random.randint(1,254)}",
            alert_type=atype,
            severity=random.randint(1, 5),
            raw_text=f"Detected {atype} from source",
        )
        for atype in alert_types
    ]

    triage = TFIDFTriageFilter(theta_max_idf=2.5, theta_tfidf=0.01)
    results, high_risk = triage.evaluate_batch(alerts)

    print("\n=== Triage Results ===")
    for r in results:
        status = "🔴 NOISE" if r.is_noise else "🟢 HIGH-RISK"
        print(f"  {status} | {r.alert.alert_type:25s} | {r.reason}")

    print(f"\n=== Stats ===")
    for k, v in triage.get_stats().items():
        print(f"  {k}: {v}")

    print(f"\nHigh-risk alerts forwarded to DIM: {len(high_risk)}")
