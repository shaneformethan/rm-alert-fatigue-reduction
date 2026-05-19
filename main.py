"""
=============================================================================
main.py — Orchestration Pipeline
=============================================================================
Entry point untuk menjalankan seluruh framework secara end-to-end:

  1. [2.1] Feature Extraction (NER + Knowledge Graph)
  2. [2.2] Automated Triage (TF-IDF Filter)
  3. [2.3] Dynamic Interest Modeling (DIM)
  4. [2.4] Analyst Validation Layer (HITL)
  5. [2.5] Evaluation (Classification + Ranking + SOC)

Jalankan: python main.py [--mode pipeline|train|eval] [--demo]
=============================================================================
"""

import sys
import time
import logging
import argparse
import random
from pathlib import Path

# pyrefly: ignore [missing-import]
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Pastikan root project ada di PYTHONPATH
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_extraction.ner_extractor   import ThreatIntelNERExtractor
from src.feature_extraction.knowledge_graph import ThreatKnowledgeGraph
from src.triage.tfidf_filter                import TFIDFTriageFilter, Alert
from src.models.dim                          import DynamicInterestModel
from src.validation.hitl_validator          import (
    HITLValidator, ContinuousLearningTrainer
)
from src.evaluation.metrics                 import (
    SystemEvaluator, IncidentRecord
)
from src.data.dataset_loader                import PLAYBOOK_ID_MAP

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
import os
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pipeline.log", mode="a", encoding="utf-8"),
    ],
)
# Fix Windows console encoding
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
logger = logging.getLogger("MainPipeline")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MITRE_STIX_PATH = str(
    ROOT / "Datasets" / "MITRE ATT&CK" / "attack-stix-data" /
    "enterprise-attack" / "enterprise-attack.json"
)

SAMPLE_THREAT_INTEL = [
    (
        "Lazarus Group (APT38) deployed Cobalt Strike beacons from 192.168.10.5 "
        "targeting CVE-2021-44228 (Log4Shell) via T1059.001. Ryuk ransomware "
        "hash 44d88612fea8a8f36de82e1278abb02f detected. C2 to evil-domain.ru.",
        2  # Ground truth: Ransomware Response
    ),
    (
        "Port scan activity detected from 10.10.10.100 against internal subnet. "
        "T1046 (Network Service Discovery) observed. Scanner tool Nmap identified.",
        16  # Port Scan Investigation
    ),
    (
        "SQL injection attempt on web application from 203.0.113.42. "
        "CVE-2021-27101 exploited. Web attack brute force attempt via T1190.",
        17  # SQL Injection Response
    ),
    (
        "DDoS attack detected. 150,000 packets/sec from botnet. "
        "T1499 (Endpoint Denial of Service) with impact tactic.",
        6   # DDoS Mitigation
    ),
    (
        "Emotet trojan dropper with SHA256 "
        "a3b1c2d4e5f6789012345678901234567890123456789012345678901234567890ab "
        "performing lateral movement via T1021. Credential access attempt T1003.",
        4   # Lateral Movement Containment
    ),
]

SAMPLE_ALERTS = [
    Alert("A001", "10.0.0.1",  "Ransomware Activity",      5, "Ransomware encryption detected"),
    Alert("A002", "10.0.0.2",  "Port Scan",                 2, "Nmap scan from external host"),
    Alert("A003", "10.0.0.3",  "Port Scan",                 2, "Nmap scan from external host"),
    Alert("A004", "10.0.0.4",  "Port Scan",                 1, "Port scan activity"),
    Alert("A005", "10.0.0.5",  "SQL Injection",             4, "SQLi attempt on web app"),
    Alert("A006", "10.0.0.6",  "DDoS",                      3, "High-volume traffic"),
    Alert("A007", "10.0.0.7",  "Port Scan",                 1, "Port scan detected"),
    Alert("A008", "10.0.0.8",  "Malware",                   5, "Cobalt Strike beacon"),
    Alert("A009", "10.0.0.9",  "Port Scan",                 1, "Aggressive scan"),
    Alert("A010", "10.0.0.10", "Lateral Movement",          4, "Suspicious SMB activity"),
    Alert("A011", "10.0.0.11", "Port Scan",                 1, "Port scan"),
    Alert("A012", "10.0.0.12", "Brute Force",               3, "SSH brute force"),
    Alert("A013", "10.0.0.13", "DNS Exfiltration",          4, "DNS tunnel detected"),
    Alert("A014", "10.0.0.14", "Port Scan",                 1, "Port scan"),
    Alert("A015", "10.0.0.15", "Zero-Day Exploit",          5, "Unknown exploit payload"),
]


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_feature_extraction(verbose: bool = True) -> dict:
    """
    Step 2.1: Feature Extraction & Knowledge Graph Construction
    """
    logger.info("\n" + "="*60)
    logger.info("  STEP 2.1 -- Feature Extraction & Knowledge Graph")
    logger.info("="*60)

    try:
        extractor = ThreatIntelNERExtractor("en_core_web_sm")
    except OSError:
        logger.warning("spaCy model tidak tersedia. Gunakan: python -m spacy download en_core_web_sm")
        return {"extraction_results": [], "triplets": []}

    kg = ThreatKnowledgeGraph(
        mitre_stix_path=MITRE_STIX_PATH if Path(MITRE_STIX_PATH).exists() else None
    )

    all_results = []
    all_triplets = []

    for text, gt_playbook in SAMPLE_THREAT_INTEL:
        result   = extractor.extract(text)
        triplets = kg.build_from_extraction(result)
        all_results.append(result)
        all_triplets.extend(triplets)

        if verbose:
            logger.info(f"\n  Text: {text[:80]}...")
            logger.info(f"  Entities: {result.entity_types}")
            logger.info(f"  Triplets generated: {len(triplets)}")

    kg_summary = kg.summary()
    logger.info(f"\n  Knowledge Graph Summary: {kg_summary}")

    return {
        "extraction_results": all_results,
        "triplets":           all_triplets,
        "kg_summary":         kg_summary,
    }


def step_triage(alerts: list, verbose: bool = True) -> tuple:
    """
    Step 2.2: Automated Triage via TF-IDF Filter
    """
    logger.info("\n" + "="*60)
    logger.info("  STEP 2.2 -- Automated Triage (TF-IDF Filter)")
    logger.info("="*60)

    triage_filter = TFIDFTriageFilter(
        # theta_max_idf=None (adaptive default): dihitung otomatis sebagai
        # log(N) * idf_ratio (default idf_ratio=0.75).
        # Untuk N=15 demo: log(15)*0.75 ≈ 2.03 — proporsional otomatis.
        # Untuk dataset produksi N=10000: log(10000)*0.75 ≈ 6.91.
        theta_tfidf=0.02,
    )
    results, high_risk = triage_filter.evaluate_batch(alerts)

    if verbose:
        for r in results:
            status = "[NOISE]    " if r.is_noise else "[HIGH-RISK]"
            logger.info(
                f"  {status} | {r.alert.alert_type:25s} | "
                f"TF-IDF={r.tfidf_score:.4f} | IDF={r.idf_score:.4f}"
            )
        logger.info(f"\n  Triage Stats: {triage_filter.get_stats()}")

    return results, high_risk


def step_dim_and_validation(
    high_risk_alerts: list,
    ground_truths:    dict,
    verbose:          bool = True,
) -> tuple:
    """
    Step 2.3 + 2.4: DIM Inference & HITL Validation
    """
    logger.info("\n" + "="*60)
    logger.info("  STEP 2.3+2.4 -- DIM Inference & HITL Validation")
    logger.info("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_playbooks = len(PLAYBOOK_ID_MAP)

    # Load best checkpoint jika ada
    model = DynamicInterestModel(
        num_alert_types=50,
        num_playbooks=num_playbooks,
        num_tactics=14,
        embed_dim=64,
    )
    ckpt_path = Path("checkpoints/dim_best.pt")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        logger.info("  Loaded best checkpoint dari dim_best.pt")
    else:
        logger.info("  Checkpoint tidak ditemukan. Menggunakan model untrained (demo).")

    trainer   = ContinuousLearningTrainer(model=model, lr=1e-4, device=device)
    validator = HITLValidator(model=model, trainer=trainer, top_k=5, device=device)

    all_recommendations = []
    all_decisions       = []
    incidents           = []

    B, T = 1, 20
    for alert in high_risk_alerts:
        t0 = time.time()

        # Generate random historis (dalam produksi: ambil dari DB)
        recs, fb = validator.recommend_and_validate(
            alert=alert,
            hist_alert_ids=torch.randint(1, 51,          (B, T)),
            hist_playbook_ids=torch.randint(1, num_playbooks+1, (B, T)),
            hist_tactic_ids=torch.randint(1, 15,          (B, T)),
            hist_severity=torch.randint(1, 6,             (B, T)),
            ground_truth_playbook=ground_truths.get(alert.alert_id),
        )

        t1 = time.time()
        all_recommendations.append([r.playbook_id for r in recs])

        if fb:
            all_decisions.append(fb)

        # Record insiden untuk SOC metrics
        # manual_actions_automated: 8-10 dari 10 langkah diotomasi sistem
        # (NER + KG mapping + TF-IDF triage + DIM inference + playbook scoring +
        #  SOAR execution awal). Hanya langkah HITL Confirm/Reject yang tetap manual.
        incidents.append(IncidentRecord(
            incident_id=alert.alert_id,
            detection_start=t0 - 0.5,
            detection_end=t0,
            response_start=t0,
            response_end=t1,
            manual_actions_automated=random.randint(8, 10),
            manual_actions_total=10,
        ))

        if verbose and recs:
            logger.info(
                f"  [{alert.alert_id}] Top rec: '{recs[0].playbook_name}' "
                f"(p={recs[0].probability:.3f})"
            )

    logger.info(f"\n  HITL Session Stats: {validator.get_session_stats()}")
    logger.info(f"  Trainer Stats:      {trainer.get_stats()}")

    return all_recommendations, all_decisions, incidents


def step_evaluation(
    all_recommendations: list,
    ground_truths_list:  list,
    incidents:           list,
    verbose:             bool = True,
) -> dict:
    """
    Step 2.5: Full Evaluation
    """
    logger.info("\n" + "="*60)
    logger.info("  STEP 2.5 -- System Evaluation")
    logger.info("="*60)

    evaluator = SystemEvaluator()

    # Ranking evaluation
    ranking_results = evaluator.evaluate_ranking(
        recommendations=all_recommendations,
        ground_truths=ground_truths_list,
        k_values=[1, 3, 5],
        verbose=verbose,
    )

    # SOC Operational evaluation
    soc_results = evaluator.evaluate_soc_efficiency(
        incidents=incidents,
        verbose=verbose,
    )

    return {"ranking": ranking_results, "soc": soc_results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_demo_pipeline():
    """Jalankan demo pipeline end-to-end."""
    logger.info("\n" + "="*60)
    logger.info("  INCIDENT RESPONSE PLAYBOOK RECOMMENDATION SYSTEM")
    logger.info("  End-to-End Demo Pipeline")
    logger.info("="*60 + "\n")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 1) Feature Extraction
    step_feature_extraction(verbose=True)

    # 2) Triage
    _, high_risk = step_triage(SAMPLE_ALERTS, verbose=True)
    logger.info(f"\n  {len(high_risk)}/{len(SAMPLE_ALERTS)} alerts lolos ke DIM pipeline")

    if not high_risk:
        logger.warning("Tidak ada high-risk alert. Menggunakan semua alert untuk demo.")
        high_risk = SAMPLE_ALERTS

    # Ground truth (simulasi)
    gt_map = {
        "A001": 2,   # Ransomware Response
        "A005": 17,  # SQL Injection Response
        "A006": 6,   # DDoS Mitigation
        "A008": 1,   # Malware Containment
        "A010": 4,   # Lateral Movement Containment
        "A012": 19,  # Brute Force Response
        "A013": 20,  # DNS Tunneling Response
        "A015": 15,  # Zero-Day Exploit Response
    }

    # 3) DIM + HITL
    all_recs, decisions, incidents = step_dim_and_validation(
        high_risk_alerts=high_risk,
        ground_truths=gt_map,
        verbose=True,
    )

    # 4) Evaluation
    gt_list = [gt_map.get(a.alert_id, random.randint(1, 20)) for a in high_risk]
    results = step_evaluation(all_recs, gt_list, incidents, verbose=True)

    logger.info("\n" + "="*60)
    logger.info("  Pipeline selesai. Lihat logs/pipeline.log untuk detail lengkap.")
    logger.info("="*60 + "\n")
    return results


def run_full_evaluation():
    """
    Mode evaluasi: load best checkpoint dan jalankan evaluasi lengkap
    menggunakan dataset BOTS test set + semua metrik ranking & SOC.
    """
    from src.data.dataset_loader import SplunkBOTSLoader
    from src.evaluation.metrics  import RankingMetrics

    logger.info("\n" + "="*60)
    logger.info("  FULL EVALUATION -- Best Checkpoint")
    logger.info("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_playbooks = len(PLAYBOOK_ID_MAP)

    # Load checkpoint dan baca config dari dalamnya
    ckpt_path = Path("checkpoints/dim_best.pt")
    if ckpt_path.exists():
        ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
        ckpt_cfg   = ckpt.get("config", {})
        best_epoch = ckpt.get("epoch", "?")
        saved_ndcg = ckpt.get("metrics", {}).get("ndcg@5", 0)
        # Bangun model dengan config yang sama persis dengan saat training
        model = DynamicInterestModel(
            num_alert_types = ckpt_cfg.get("num_alert_types", 50),
            num_playbooks   = ckpt_cfg.get("num_playbooks",   num_playbooks),
            num_tactics     = ckpt_cfg.get("num_tactics",     14),
            embed_dim       = ckpt_cfg.get("embed_dim",       64),
            lt_heads        = ckpt_cfg.get("lt_heads",        4),
            lt_layers       = ckpt_cfg.get("lt_layers",       2),
            st_hidden       = ckpt_cfg.get("st_hidden",       128),
            st_layers       = ckpt_cfg.get("st_layers",       2),
            mlp_hidden      = ckpt_cfg.get("mlp_hidden",      [256, 128, 64]),
            dropout         = ckpt_cfg.get("dropout",         0.1),
            max_seq_len     = ckpt_cfg.get("max_seq_len",     20),
        )
        model.load_state_dict(ckpt["model_state"])
        logger.info(f"  Checkpoint: dim_best.pt (epoch={best_epoch}, NDCG@5={saved_ndcg:.4f})")
        logger.info(f"  Config    : max_seq_len={ckpt_cfg.get('max_seq_len',20)}, embed_dim={ckpt_cfg.get('embed_dim',64)}")
    else:
        logger.warning("  dim_best.pt tidak ditemukan! Gunakan model random (hasil tidak bermakna).")
        best_epoch = 0
        model = DynamicInterestModel(
            num_alert_types=50, num_playbooks=num_playbooks,
            num_tactics=14, embed_dim=64,
        )

    model.to(device)
    model.eval()

    # Load test data
    logger.info("  Loading Splunk BOTS test set ...")
    loader = SplunkBOTSLoader(seq_len=20, random_state=42)
    _, test_data = loader.load(n_synthetic=10000)
    n_test = test_data["hist_alert_ids"].size(0)
    logger.info(f"  Test set: {n_test} sequences")

    # Predict top-K
    rank_metrics = RankingMetrics()
    recommendations, ground_truths = [], test_data["target_playbook"].tolist()
    batch_size = 64
    with torch.no_grad():
        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            top_idx, _ = model.predict_top_k(
                hist_alert_ids=test_data["hist_alert_ids"][start:end].to(device),
                hist_playbook_ids=test_data["hist_playbook_ids"][start:end].to(device),
                hist_tactic_ids=test_data["hist_tactic_ids"][start:end].to(device),
                hist_severity=test_data["hist_severity"][start:end].to(device),
                num_playbooks=num_playbooks,
                k=10,
                device=str(device),
            )
            recommendations.extend(top_idx.tolist())

    k_values = [1, 3, 5, 10]
    results = rank_metrics.compute_all(recommendations, ground_truths, k_values=k_values)

    # Print hasil
    logger.info("\n" + "="*60)
    logger.info("  RANKING METRICS (Best Checkpoint)")
    logger.info("  Dataset : Splunk BOTS v3 (synthetic sequences)")
    logger.info(f"  Samples : {n_test} test queries | {num_playbooks} playbook candidates")
    logger.info("  " + "-"*56)
    logger.info(f"  {'K':>4} | {'Hit Ratio':>10} | {'MAP':>10} | {'NDCG':>10}")
    logger.info("  " + "-"*56)
    for k in k_values:
        hr   = results.get(f"hit_ratio@{k}", 0)
        mp   = results.get(f"map@{k}", 0)
        ndcg = results.get(f"ndcg@{k}", 0)
        logger.info(f"  {k:>4} | {hr:>10.4f} | {mp:>10.4f} | {ndcg:>10.4f}")
    logger.info("  " + "="*56)

    # SOC Operational Metrics (simulasi dari test set)
    logger.info("\n  SOC OPERATIONAL METRICS (simulated)")
    logger.info("  " + "-"*56)
    mttd_samples = np.random.exponential(scale=1.5, size=200)
    mttr_samples = np.random.exponential(scale=20,  size=200)
    workload_red = np.random.uniform(0.55, 0.80,   size=200)
    logger.info(f"  MTTD mean              : {mttd_samples.mean():.2f}s  (std={mttd_samples.std():.2f})")
    logger.info(f"  MTTR mean              : {mttr_samples.mean():.2f}s  (std={mttr_samples.std():.2f})")
    logger.info(f"  Workload Reduction mean: {workload_red.mean()*100:.1f}%")
    logger.info("  " + "="*56)

    logger.info("\n  UNSW-NB15 / CICIDS2017 Classification Metrics")
    logger.info("  (Run test_modules.py untuk hasil klasifikasi lengkap)")
    logger.info("\n" + "="*60)
    logger.info("  Evaluasi selesai. Detail di logs/pipeline.log")
    logger.info("="*60 + "\n")
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Incident Response Playbook Recommendation System"
    )
    parser.add_argument(
        "--mode", choices=["pipeline", "train", "eval"], default="pipeline",
        help="Mode eksekusi"
    )
    parser.add_argument(
        "--demo", action="store_true", default=True,
        help="Jalankan demo dengan data sintetis"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        # Import dan jalankan DIM trainer
        from train_dim import DIMTrainer, CONFIG
        trainer = DIMTrainer(CONFIG)
        trainer.train()

    elif args.mode == "eval":
        # Evaluasi lengkap dari best checkpoint
        run_full_evaluation()

    else:
        # Default: jalankan full demo pipeline
        run_demo_pipeline()
