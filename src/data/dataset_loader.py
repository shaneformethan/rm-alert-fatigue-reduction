"""
=============================================================================
Module 2.5 - Dataset Loader
=============================================================================
Loader untuk hybrid dataset:
  1. Splunk BOTS v3       -> pelatihan DIM (sekuensi playbook)
  2. UNSW-NB15            -> filter triase TF-IDF
  3. CICIDS2017           -> filter triase TF-IDF

Preprocessing:
  - Normalisasi fitur numerik
  - SMOTE untuk class imbalance (serangan << normal)
  - Label encoding alert_type dan playbook_class
=============================================================================
"""

import os
import logging
import re
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
DATASETS_DIR = BASE_DIR / "Datasets"

# UNSW-NB15: gunakan pre-split training/testing set (sudah punya header + attack_cat)
UNSW_TRAIN_CSV = DATASETS_DIR / "UNSW-NB15" / "CSV Files" / "Training and Testing Sets" / "UNSW_NB15_training-set.csv"
UNSW_TEST_CSV  = DATASETS_DIR / "UNSW-NB15" / "CSV Files" / "Training and Testing Sets" / "UNSW_NB15_testing-set.csv"

# CICIDS2017: hanya file yang mengandung serangan signifikan
CICIDS_ATTACK_FILES = [
    DATASETS_DIR / "CICIDS2017" / "archive" / f
    for f in [
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",         # DDoS: 128k
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",     # PortScan: 158k
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",   # XSS, SQLi, BruteForce
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",                # Bot, Infiltration
        "Tuesday-WorkingHours.pcap_ISCX.csv",                       # FTP/SSH Brute Force
    ]
]

# Mapping CICIDS label -> kategori playbook (anotasi manual ke kelas mitigasi)
CICIDS_TO_PLAYBOOK = {
    "BENIGN":                   "normal",
    "DoS Hulk":                 "DDoS Mitigation",
    "PortScan":                 "Port Scan Investigation",
    "DDoS":                     "DDoS Mitigation",
    "DoS GoldenEye":            "DDoS Mitigation",
    "FTP-Patator":              "Brute Force Response",
    "SSH-Patator":              "Brute Force Response",
    "DoS slowloris":            "DDoS Mitigation",
    "DoS Slowhttptest":         "DDoS Mitigation",
    "Bot":                      "Malware Containment & Eradication",
    "Web Attack - Brute Force": "Brute Force Response",
    "Web Attack - XSS":         "XSS Attack Response",
    "Web Attack - Sql Injection": "SQL Injection Response",
    "Infiltration":             "Network Intrusion Response",
    "Heartbleed":               "Vulnerability Exploitation Response",
}

# Mapping UNSW attack_cat -> playbook class
UNSW_TO_PLAYBOOK = {
    "Normal":       "normal",
    "Generic":      "Malware Containment & Eradication",
    "Exploits":     "Vulnerability Exploitation Response",
    "Fuzzers":      "Network Intrusion Response",
    "DoS":          "DDoS Mitigation",
    "Reconnaissance": "Port Scan Investigation",
    "Analysis":     "Network Intrusion Response",
    "Backdoor":     "Malware Containment & Eradication",
    "Shellcode":    "Vulnerability Exploitation Response",
    "Worms":        "Malware Containment & Eradication",
}

# Playbook ID catalog (konsisten dengan HITLValidator)
PLAYBOOK_ID_MAP = {
    "normal":                              0,
    "Malware Containment & Eradication":   1,
    "Ransomware Response":                 2,
    "Phishing Investigation":              3,
    "Lateral Movement Containment":        4,
    "Data Exfiltration Response":          5,
    "DDoS Mitigation":                     6,
    "Privilege Escalation Response":       7,
    "Credential Compromise Response":      8,
    "Network Intrusion Response":          9,
    "APT Investigation":                   10,
    "Vulnerability Exploitation Response": 11,
    "Insider Threat Investigation":        12,
    "Ransomware Negotiation":              13,
    "C2 Beaconing Response":               14,
    "Zero-Day Exploit Response":           15,
    "Port Scan Investigation":             16,
    "SQL Injection Response":              17,
    "XSS Attack Response":                 18,
    "Brute Force Response":                19,
    "DNS Tunneling Response":              20,
}


# ===========================================================================
# UNSW-NB15 Loader
# ===========================================================================

class UNSWDatasetLoader:
    """
    Loader untuk dataset UNSW-NB15.
    Menggunakan pre-split training/testing set yang sudah memiliki header
    dan kolom attack_cat secara eksplisit.
    Digunakan untuk: TF-IDF triage filter & anomaly detection.
    """

    # Fitur numerik dari training-set (exclude id, attack_cat, label)
    NUMERIC_FEATURES = [
        "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
        "sttl", "dttl", "sload", "dload", "sloss", "dloss",
        "sinpkt", "dinpkt", "sjit", "djit", "swin", "stcpb",
        "dtcpb", "dwin", "tcprtt", "synack", "ackdat",
        "smean", "dmean", "trans_depth", "response_body_len",
        "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
        "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
        "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd",
        "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
    ]
    CATEGORICAL_FEATURES = ["proto", "service", "state"]

    def __init__(self, sample_ratio: float = 1.0):
        self.sample_ratio = sample_ratio
        self.le_attack   = LabelEncoder()
        self.le_proto    = LabelEncoder()
        self.le_service  = LabelEncoder()
        self.le_state    = LabelEncoder()
        self.scaler      = StandardScaler()

    def load(
        self,
        apply_smote:  bool  = True,
        random_state: int   = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Load UNSW-NB15 dari pre-split training/testing set resmi.
        File sudah memiliki header + kolom attack_cat.

        Returns:
            X_train, X_test, y_train, y_test, label_names
        """
        if not UNSW_TRAIN_CSV.exists() or not UNSW_TEST_CSV.exists():
            logger.warning("File UNSW training/testing set tidak ditemukan. Menggunakan dummy data.")
            return self._dummy_data()

        logger.info(f"Loading UNSW-NB15 training set: {UNSW_TRAIN_CSV.name}")
        df_train = pd.read_csv(UNSW_TRAIN_CSV, low_memory=False)
        logger.info(f"Loading UNSW-NB15 testing set:  {UNSW_TEST_CSV.name}")
        df_test  = pd.read_csv(UNSW_TEST_CSV,  low_memory=False)

        if self.sample_ratio < 1.0:
            df_train = df_train.sample(frac=self.sample_ratio, random_state=random_state)
            df_test  = df_test.sample(frac=self.sample_ratio,  random_state=random_state)

        logger.info(f"UNSW-NB15 | train={len(df_train):,} rows | test={len(df_test):,} rows")
        return self._preprocess_split(df_train, df_test, apply_smote, random_state)

    def _preprocess_split(
        self,
        df_train:     pd.DataFrame,
        df_test:      pd.DataFrame,
        apply_smote:  bool,
        random_state: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Preprocess UNSW-NB15 menggunakan pre-split train/test resmi."""

        def _encode_and_extract(df: pd.DataFrame, fit: bool) -> Tuple[np.ndarray, np.ndarray]:
            df = df.copy()
            # Map attack_cat -> playbook class
            df["attack_cat"] = df["attack_cat"].fillna("Normal").astype(str).str.strip()
            df["playbook_class"] = df["attack_cat"].map(
                lambda x: UNSW_TO_PLAYBOOK.get(x, "Network Intrusion Response")
            )
            y = (self.le_attack.fit_transform if fit else self.le_attack.transform)(df["playbook_class"])

            # Encode categorical features - 'unknown' selalu di-fit agar test-set
            # yang punya nilai baru tidak crash saat transform
            for col, enc in [("proto", self.le_proto), ("service", self.le_service), ("state", self.le_state)]:
                if col in df.columns:
                    vals = df[col].fillna("unknown").astype(str)
                    if fit:
                        fit_vals = pd.concat([vals, pd.Series(["unknown"])], ignore_index=True)
                        enc.fit(fit_vals)
                        df[col] = enc.transform(vals)
                    else:
                        safe = vals.map(lambda v: v if v in enc.classes_ else "unknown")
                        df[col] = enc.transform(safe)

            # Semua fitur (numerik + encoded categorical)
            all_feats = self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES
            avail = [c for c in all_feats if c in df.columns]
            X = df[avail].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
            return X, y

        X_tr, y_tr = _encode_and_extract(df_train, fit=True)
        X_te, y_te = _encode_and_extract(df_test,  fit=False)

        # Scaling
        X_tr = self.scaler.fit_transform(X_tr)
        X_te = self.scaler.transform(X_te)

        # SMOTE
        if apply_smote:
            X_tr, y_tr = self._apply_smote(X_tr, y_tr, random_state)

        label_names = list(self.le_attack.classes_)
        logger.info(
            f"UNSW-NB15 preprocessed: train={X_tr.shape}, test={X_te.shape}, "
            f"classes={len(label_names)} -> {label_names}"
        )
        return X_tr, X_te, y_tr, y_te, label_names

    @staticmethod
    def _apply_smote(
        X: np.ndarray, y: np.ndarray, random_state: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        SMOTE menggunakan K-Nearest Neighbors untuk mencegah overfitting
        dan bias klasifikasi akibat class imbalance.
        """
        counts = np.bincount(y)
        if counts.min() < 2:
            logger.warning("SMOTE dilewati: ada kelas dengan < 2 sampel.")
            return X, y
        k_neighbors = min(5, counts.min() - 1)
        smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
        try:
            X_res, y_res = smote.fit_resample(X, y)
            logger.info(
                f"SMOTE applied: {X.shape[0]} -> {X_res.shape[0]} samples"
            )
            return X_res, y_res
        except Exception as e:
            logger.warning(f"SMOTE gagal: {e}")
            return X, y

    @staticmethod
    def _dummy_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Dummy data untuk testing tanpa file dataset."""
        np.random.seed(42)
        n, d = 1000, 20
        X = np.random.randn(n, d).astype(np.float32)
        y = np.random.randint(0, 5, n)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_tr, X_te, y_tr, y_te, [f"class_{i}" for i in range(5)]


# ===========================================================================
# CICIDS2017 Loader
# ===========================================================================

class CICIDSDatasetLoader:
    """
    Loader untuk dataset CICIDS2017.
    Hanya menggunakan file yang mengandung serangan signifikan
    (DDoS, PortScan, WebAttacks, Infiltration, BruteForce).
    Digunakan untuk: TF-IDF triage filter & anomaly detection.
    """

    LABEL_COL = " Label"   # kolom label CICIDS2017 memiliki leading space

    def __init__(self, max_rows_per_file: int = 50000, sample_ratio: float = 1.0):
        self.max_rows    = max_rows_per_file
        self.sample_ratio = sample_ratio
        self.le          = LabelEncoder()
        self.scaler      = StandardScaler()

    def load(
        self,
        apply_smote:  bool  = True,
        test_size:    float = 0.2,
        random_state: int   = 42,
        max_files:    int   = 4,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Load dari file-file CICIDS2017 yang mengandung serangan nyata.
        """
        files = [f for f in CICIDS_ATTACK_FILES if Path(f).exists()][:max_files]

        if not files:
            logger.warning("File CICIDS2017 tidak ditemukan. Menggunakan dummy data.")
            return UNSWDatasetLoader._dummy_data()

        dfs = []
        for csv_file in files:
            name = Path(csv_file).name
            logger.info(f"Loading CICIDS2017: {name}")
            try:
                df = pd.read_csv(
                    csv_file,
                    nrows=self.max_rows,
                    low_memory=False,
                    encoding="utf-8",
                    on_bad_lines="skip",
                )
                dfs.append(df)
                vc = df[self.LABEL_COL].value_counts().to_dict() if self.LABEL_COL in df.columns else {}
                logger.info(f"  -> {name}: {len(df):,} rows | labels: {vc}")
            except Exception as e:
                logger.warning(f"Skip {name}: {e}")

        if not dfs:
            return UNSWDatasetLoader._dummy_data()

        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"CICIDS2017 total rows: {len(df):,}")

        if self.sample_ratio < 1.0:
            df = df.sample(frac=self.sample_ratio, random_state=42)

        return self._preprocess(df, apply_smote, test_size, random_state)

    def _preprocess(
        self,
        df: pd.DataFrame,
        apply_smote:  bool,
        test_size:    float,
        random_state: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        # Strip whitespace dari nama kolom
        df.columns = df.columns.str.strip()

        if "Label" not in df.columns:
            logger.warning("Kolom 'Label' tidak ditemukan di CICIDS2017.")
            return UNSWDatasetLoader._dummy_data()

        df["Label"] = df["Label"].astype(str).str.strip()
        # Map ke playbook class
        df["playbook_class"] = df["Label"].map(
            lambda x: CICIDS_TO_PLAYBOOK.get(x, "Network Intrusion Response")
        )
        logger.info(f"CICIDS2017 label distribution:\n{df['Label'].value_counts().to_string()}")
        y = self.le.fit_transform(df["playbook_class"])

        # Fitur numerik (exclude Label)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ("Label", "label")]

        X = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
        X = np.clip(X, -1e9, 1e9)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        X_tr = self.scaler.fit_transform(X_tr)
        X_te = self.scaler.transform(X_te)

        if apply_smote:
            X_tr, y_tr = UNSWDatasetLoader._apply_smote(X_tr, y_tr, random_state)

        label_names = list(self.le.classes_)
        logger.info(
            f"CICIDS2017 preprocessed: "
            f"train={X_tr.shape}, test={X_te.shape}, classes={len(label_names)}"
        )
        return X_tr, X_te, y_tr, y_te, label_names


# ===========================================================================
# Splunk BOTS v3 Loader
# ===========================================================================

class SplunkBOTSLoader:
    """
    Sequence loader for DIM training, derived from Splunk BOTS v3 eventcode distribution.

    Since raw SIEM logs lack response labels, sequences are generated in two steps:
      1. Map Windows Event Codes (from eventcode.csv) to playbook classes via
         EVENTCODE_TO_PLAYBOOK (based on MITRE ATT&CK semantics).
      2. Generate synthetic playbook history sequences using the BOTS eventcode
         frequency distribution as a prior for playbook selection.

    Use describe() for full provenance details.
    """

    LOOKUPS_DIR = DATASETS_DIR / "Splunk BOTS v3" / "botsv3_data_set" / "lookups"

    # Mapping eventcode -> playbook berdasarkan anotasi sistematis
    # Sumber: Windows Security Event Log semantics + MITRE ATT&CK mapping
    EVENTCODE_TO_PLAYBOOK = {
        "4624": "Credential Compromise Response",    # Successful logon
        "4625": "Brute Force Response",              # Failed logon
        "4648": "Lateral Movement Containment",      # Logon with explicit credentials
        "4688": "Malware Containment & Eradication", # New process created
        "4698": "Malware Containment & Eradication", # Scheduled task created (persistence)
        "4720": "Privilege Escalation Response",     # User account created
        "4732": "Privilege Escalation Response",     # User added to privileged group
        "7045": "Malware Containment & Eradication", # New service installed
        "4697": "Malware Containment & Eradication", # Service installed
        "4663": "Data Exfiltration Response",        # File access attempt
        "4657": "Insider Threat Investigation",      # Registry value modified
        "4672": "Privilege Escalation Response",     # Special privileges assigned
    }

    def __init__(self, seq_len: int = 20, random_state: int = 42):
        self.seq_len      = seq_len
        self.random_state = random_state
        self._generation_metadata: Dict = {}   # diisi saat load() dipanggil

    def describe(self) -> Dict:
        """
        Kembalikan ringkasan provenance data yang dihasilkan.

        Berguna untuk transparansi dalam laporan evaluasi: menjelaskan
        bahwa DIM dilatih dari synthetic sequences yang diturunkan dari
        distribusi BOTS, bukan raw BOTS log secara langsung.

        Returns:
            Dict berisi informasi sumber data, metode generasi, dan statistik.
        """
        base = {
            "dataset": "Splunk BOTS v3",
            "dataset_type": "SYNTHETIC_SEQUENCES_DERIVED_FROM_BOTS",
            "generation_method": (
                "Synthetic playbook sequences generated from Splunk BOTS v3 "
                "eventcode distribution. Each sequence represents one simulated "
                "SOC incident response session."
            ),
            "annotation_note": (
                "Raw BOTS data does not inherently contain mitigation response labels. "
                "Windows Event Codes are systematically annotated to playbook classes "
                "via EVENTCODE_TO_PLAYBOOK mapping before sequence generation."
            ),
            "eventcode_to_playbook": self.EVENTCODE_TO_PLAYBOOK,
            "seq_len": self.seq_len,
            "lookups_dir": str(self.LOOKUPS_DIR),
            "eventcode_file_exists": (self.LOOKUPS_DIR / "eventcode.csv").exists(),
        }
        base.update(self._generation_metadata)
        return base

    def load(
        self,
        test_size: float = 0.2,
        n_synthetic: int  = 5000,
    ) -> Tuple[Dict, Dict]:
        """
        Load BOTS dataset untuk pelatihan DIM.

        Proses:
        1. Baca distribusi eventcode dari BOTS lookups/eventcode.csv
           (jika file tidak ada, gunakan distribusi uniform sebagai fallback).
        2. Petakan eventcode -> kelas playbook via EVENTCODE_TO_PLAYBOOK.
        3. Generate n_synthetic sekuensi historis playbook sintetis
           berdasarkan distribusi tersebut.
        4. Split train/test dan pad ke max_seq_len.

        CATATAN: Returned dict menyertakan kunci 'generation_info' yang berisi
        metadata provenance untuk transparansi evaluasi. Gunakan juga
        method describe() untuk ringkasan lengkap.

        Args:
            test_size:   Proporsi data uji (default 0.2).
            n_synthetic: Jumlah sekuensi sintetis yang di-generate (default 5000).

        Returns:
            train_data, test_data (dict of tensors + generation_info)
        """
        eventcode_file = self.LOOKUPS_DIR / "eventcode.csv"
        playbook_dist, dist_source = self._load_eventcode_distribution(eventcode_file)

        logger.info(
            f"Generating {n_synthetic} sequences from {dist_source} | "
            f"seq_len={self.seq_len} | "
            f"playbook_classes={len(set(self.EVENTCODE_TO_PLAYBOOK.values()))}"
        )


        return self._generate_sequences(
            playbook_dist=playbook_dist,
            n_sequences=n_synthetic,
            test_size=test_size,
            dist_source=dist_source,
        )

    def _load_eventcode_distribution(
        self, eventcode_file: Path
    ) -> Tuple[Dict[str, float], str]:
        """
        Baca distribusi eventcode dari BOTS lookup file.

        Returns:
            (dist_dict, source_description) - di mana source menjelaskan
            apakah distribusi berasal dari file BOTS asli atau fallback uniform.
        """
        if not eventcode_file.exists():
            logger.warning(
                f"eventcode.csv tidak ditemukan di {self.LOOKUPS_DIR}. "
                "Menggunakan distribusi uniform sebagai fallback - "
                "semua playbook class mendapat bobot yang sama."
            )
            dist = {k: 1.0 / len(self.EVENTCODE_TO_PLAYBOOK)
                    for k in self.EVENTCODE_TO_PLAYBOOK}
            return dist, "uniform_fallback (eventcode.csv not found)"

        try:
            df = pd.read_csv(eventcode_file)
            df.columns = df.columns.str.strip().str.lower()
            if "eventcode" in df.columns:
                dist = df["eventcode"].astype(str).value_counts(normalize=True).to_dict()
                n_unique = len(dist)
                logger.info(
                    f"Eventcode distribution loaded from BOTS: "
                    f"{n_unique} unique event codes"
                )
                return dist, f"bots_eventcode_csv ({n_unique} unique codes)"
        except Exception as e:
            logger.warning(f"Gagal baca eventcode.csv: {e}")

        dist = {k: 1.0 / len(self.EVENTCODE_TO_PLAYBOOK)
                for k in self.EVENTCODE_TO_PLAYBOOK}
        return dist, "uniform_fallback (parse error)"

    def _generate_sequences(
        self,
        playbook_dist: Dict[str, float],
        n_sequences:   int,
        test_size:     float,
        dist_source:   str = "unknown",
        signal_ratio:  float = 0.70,  # dominant signal strength (0.7 = clear pattern)
    ) -> Tuple[Dict, Dict]:
        """
        Generate synthetic playbook history sequences for DIM training.

        Sequence composition (3 types):
          - Standard    (40%): dominant playbook in 70% of steps, target=dominant
          - Multi-attack (20%): two competing playbooks, target=dominant
          - Phase-shift  (40%): multi-stage attack simulation, target=late-phase playbook

        Sequences are derived from Splunk BOTS v3 eventcode distribution via
        systematic annotation to playbook response classes.
        """
        import random
        random.seed(self.random_state)
        np.random.seed(self.random_state)

        playbook_ids   = list(PLAYBOOK_ID_MAP.values())
        alert_type_ids = list(range(1, 51))  # 50 tipe alert
        tactic_ids     = list(range(1, 15))  # 14 MITRE tactics

        # Hitung distribusi playbook dari distribusi eventcode BOTS
        # (eventcode yang tidak ter-mapping diabaikan)
        playbook_weights: Dict[int, float] = {}
        for ec, weight in playbook_dist.items():
            pb_name = self.EVENTCODE_TO_PLAYBOOK.get(str(ec))
            if pb_name:
                pb_id = PLAYBOOK_ID_MAP.get(pb_name, 0)
                playbook_weights[pb_id] = playbook_weights.get(pb_id, 0) + weight

        # Jika tidak ada eventcode yang ter-mapping, pakai distribusi uniform
        if not playbook_weights:
            playbook_weights = {pid: 1.0 for pid in playbook_ids[1:]}

        weighted_pbs  = list(playbook_weights.keys())
        weighted_vals = list(playbook_weights.values())
        total_w = sum(weighted_vals)
        weighted_vals = [w / total_w for w in weighted_vals]  # normalize

        all_data = []
        for _ in range(n_sequences):
            seq_len = np.random.randint(5, self.seq_len + 1)

            # Pilih playbook dominan berbobot dari distribusi BOTS
            dominant_pb = random.choices(weighted_pbs, weights=weighted_vals, k=1)[0]
            if dominant_pb == 0:  # exclude 'normal'
                dominant_pb = random.choices(playbook_ids[1:])[0]

            hist_alerts   = np.random.choice(alert_type_ids, seq_len).tolist()
            hist_tactic   = np.random.choice(tactic_ids, seq_len).tolist()
            hist_severity = np.random.randint(1, 6, seq_len).tolist()

            # ---------------------------------------------------------------
            # Tiga tipe sequence untuk menghindari trivial pattern
            # yang cukup dijawab oleh argmax(Counter(history)):
            # ---------------------------------------------------------------
            seq_roll = np.random.random()
            seq_type = 0  # 0=standard, 1=multi-campaign, 2=phase-shift

            if seq_roll < 0.40:
                # TIPE 1 - Standard (40%):
                # dominant_pb muncul di signal_ratio% langkah.
                # MajorityVote dapat menjawab ini dengan benar.
                hist_playbook = [
                    dominant_pb if np.random.random() < signal_ratio
                    else random.choice(playbook_ids)
                    for _ in range(seq_len)
                ]
                target_playbook = dominant_pb

            elif seq_roll < 0.60:
                seq_type = 1  # multi-campaign
                # TIPE 2 - Multi-Campaign (20%):
                # Dua playbook bersaing: dominant signal_ratio%, secondary lainnya.
                secondary_pb = random.choice(
                    [pb for pb in playbook_ids[1:] if pb != dominant_pb]
                )
                secondary_ratio = (1 - signal_ratio) / 2
                hist_playbook = []
                for _ in range(seq_len):
                    r = np.random.random()
                    if r < signal_ratio:
                        hist_playbook.append(dominant_pb)
                    elif r < signal_ratio + secondary_ratio:
                        hist_playbook.append(secondary_pb)
                    else:
                        hist_playbook.append(random.choice(playbook_ids))
                target_playbook = dominant_pb

            else:
                seq_type = 2  # phase-shift
                # TIPE 3 - Phase-Shift (40%):
                # Mensimulasikan serangan multi-stage: fase awal (dominant_pb)
                # diikuti fase akhir (late_pb) yang BERBEDA.
                # Target = playbook fase AKHIR - BUKAN yang paling sering di riwayat.
                # MajorityVote SELALU SALAH di tipe ini (prediksi dominant, target=late_pb).
                # Hanya model sequential (LSTM) yang bisa menjawab dengan benar.
                late_pb = random.choice(
                    [pb for pb in playbook_ids[1:] if pb != dominant_pb]
                )
                early_len = max(1, int(seq_len * 0.55))  # 55% fase awal
                late_len  = seq_len - early_len           # 45% fase akhir (lebih panjang)

                hist_early = [
                    dominant_pb if np.random.random() < signal_ratio
                    else random.choice(playbook_ids)
                    for _ in range(early_len)
                ]
                hist_late = [
                    late_pb if np.random.random() < signal_ratio
                    else random.choice(playbook_ids)
                    for _ in range(late_len)
                ]
                hist_playbook   = hist_early + hist_late
                target_playbook = late_pb   # target = fase AKHIR, bukan dominant

            all_data.append({
                "hist_alert_ids":    hist_alerts,
                "hist_playbook_ids": hist_playbook,
                "hist_tactic_ids":   hist_tactic,
                "hist_severity":     hist_severity,
                "target_playbook":   target_playbook,
                "seq_len":           seq_len,
                "seq_type":          seq_type,  # 0=standard, 1=multi, 2=phase-shift
            })

        # Train/test split (sequential split untuk menghindari data leakage)
        split_idx = int(len(all_data) * (1 - test_size))
        train_raw, test_raw = all_data[:split_idx], all_data[split_idx:]

        # Simpan metadata provenance
        self._generation_metadata = {
            "n_sequences_generated": n_sequences,
            "n_train":               len(train_raw),
            "n_test":                len(test_raw),
            "distribution_source":   dist_source,
            "signal_ratio":          signal_ratio,
            "dominant_playbook_prior": {
                PLAYBOOK_ID_MAP.get(k, str(k)): round(v, 4)
                for k, v in zip(weighted_pbs, weighted_vals)
            },
            "sequence_composition": (
                f"40% standard (signal={signal_ratio:.0%}) + "
                "20% multi-campaign + 40% phase-shift"
            ),
        }
        logger.info(
            f"  Sequences generated: {n_sequences} "
            f"(train={len(train_raw)}, test={len(test_raw)})"
        )

        generation_info = {
            "dataset_type": "SYNTHETIC_SEQUENCES_DERIVED_FROM_BOTS",
            **self._generation_metadata,
        }

        train_tensors = self._collate(train_raw, self.seq_len)
        test_tensors  = self._collate(test_raw,  self.seq_len)

        # Sertakan generation_info sebagai metadata non-tensor
        train_tensors["generation_info"] = generation_info
        test_tensors["generation_info"]  = generation_info

        return train_tensors, test_tensors

    @staticmethod
    def _collate(data_list: List[Dict], max_seq_len: int) -> Dict:
        """Pad sequences dan convert ke tensors."""
        import torch

        def pad_seq(seq: List[int], max_len: int, pad_val: int = 0) -> List[int]:
            return seq[:max_len] + [pad_val] * max(0, max_len - len(seq))

        batch = {
            "hist_alert_ids":    [],
            "hist_playbook_ids": [],
            "hist_tactic_ids":   [],
            "hist_severity":     [],
            "target_playbook":   [],
            "seq_lengths":       [],
            "padding_mask":      [],
            "seq_type":          [],  # 0=standard, 1=multi-campaign, 2=phase-shift
        }
        for d in data_list:
            L = d["seq_len"]
            batch["hist_alert_ids"].append(pad_seq(d["hist_alert_ids"],    max_seq_len))
            batch["hist_playbook_ids"].append(pad_seq(d["hist_playbook_ids"], max_seq_len))
            batch["hist_tactic_ids"].append(pad_seq(d["hist_tactic_ids"],  max_seq_len))
            batch["hist_severity"].append(pad_seq(d["hist_severity"],      max_seq_len))
            batch["target_playbook"].append(d["target_playbook"])
            batch["seq_lengths"].append(L)
            batch["seq_type"].append(d.get("seq_type", 0))
            # Padding mask: True = pad position (ignored in attention)
            mask = [False] * L + [True] * (max_seq_len - L)
            batch["padding_mask"].append(mask)

        return {
            "hist_alert_ids":    torch.tensor(batch["hist_alert_ids"],    dtype=torch.long),
            "hist_playbook_ids": torch.tensor(batch["hist_playbook_ids"], dtype=torch.long),
            "hist_tactic_ids":   torch.tensor(batch["hist_tactic_ids"],   dtype=torch.long),
            "hist_severity":     torch.tensor(batch["hist_severity"],     dtype=torch.long),
            "target_playbook":   torch.tensor(batch["target_playbook"],   dtype=torch.long),
            "seq_lengths":       torch.tensor(batch["seq_lengths"],       dtype=torch.long),
            "padding_mask":      torch.tensor(batch["padding_mask"],      dtype=torch.bool),
            "seq_type":          torch.tensor(batch["seq_type"],          dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # UNSW-NB15 (pakai training/testing set resmi)
    print("\n=== UNSW-NB15 ===")
    unsw = UNSWDatasetLoader(sample_ratio=0.3)  # 30% untuk demo cepat
    X_tr, X_te, y_tr, y_te, labels = unsw.load(apply_smote=True)
    print(f"Train: {X_tr.shape}, Test: {X_te.shape}")
    print(f"Classes: {labels}")
    from collections import Counter
    print(f"Train class dist: {dict(sorted(Counter(y_tr.tolist()).items()))}")

    # CICIDS2017 (pakai file attack-only)
    print("\n=== CICIDS2017 ===")
    cicids = CICIDSDatasetLoader(max_rows_per_file=20000, sample_ratio=0.5)
    X_tr, X_te, y_tr, y_te, labels = cicids.load(apply_smote=True, max_files=2)
    print(f"Train: {X_tr.shape}, Test: {X_te.shape}")
    print(f"Classes: {labels}")

    # Splunk BOTS v3
    print("\n=== Splunk BOTS v3 ===")
    bots = SplunkBOTSLoader(seq_len=20)
    train_data, test_data = bots.load(n_synthetic=1000)
    print(f"Train sequences: {train_data['hist_alert_ids'].shape}")
    print(f"Test sequences:  {test_data['hist_alert_ids'].shape}")
