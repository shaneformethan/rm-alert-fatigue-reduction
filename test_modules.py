"""
test_modules.py — Validasi semua modul dengan dataset NYATA
Jalankan: .\venv\Scripts\python test_modules.py
"""
import sys, time
sys.path.insert(0, '.')

import torch
import numpy as np
from collections import Counter

from src.triage.tfidf_filter import TFIDFTriageFilter, Alert
from src.models.dim import DynamicInterestModel
from src.evaluation.metrics import RankingMetrics, SOCOperationalMetrics, IncidentRecord
from src.data.dataset_loader import (
    UNSWDatasetLoader, CICIDSDatasetLoader, SplunkBOTSLoader, PLAYBOOK_ID_MAP
)

SEP = "=" * 60

# ===========================================================================
# TEST 1 — UNSW-NB15 (Dataset Nyata)
# ===========================================================================
print(f"\n{SEP}")
print("TEST 1 — UNSW-NB15 (Real Dataset)")
print(SEP)
t0 = time.time()

unsw = UNSWDatasetLoader(sample_ratio=0.2)   # 20% ~ 35k train rows
X_tr, X_te, y_tr, y_te, labels = unsw.load(apply_smote=True)

elapsed = time.time() - t0
print(f"[OK] UNSW-NB15 loaded in {elapsed:.1f}s")
print(f"     Train : {X_tr.shape} | Test : {X_te.shape}")
print(f"     Classes ({len(labels)}): {labels}")
print(f"     Train class dist: {dict(sorted(Counter(y_tr.tolist()).items()))}")
assert X_tr.shape[1] > 0 and len(labels) > 1, "UNSW gagal: shape atau label salah"

# ===========================================================================
# TEST 2 — CICIDS2017 (Dataset Nyata, file attack-only)
# ===========================================================================
print(f"\n{SEP}")
print("TEST 2 — CICIDS2017 (Real Dataset, Attack Files)")
print(SEP)
t0 = time.time()

cicids = CICIDSDatasetLoader(max_rows_per_file=30000, sample_ratio=0.5)
X_tr_c, X_te_c, y_tr_c, y_te_c, labels_c = cicids.load(apply_smote=True, max_files=2)

elapsed = time.time() - t0
print(f"[OK] CICIDS2017 loaded in {elapsed:.1f}s")
print(f"     Train : {X_tr_c.shape} | Test : {X_te_c.shape}")
print(f"     Classes ({len(labels_c)}): {labels_c}")
print(f"     Train class dist: {dict(sorted(Counter(y_tr_c.tolist()).items()))}")
assert X_tr_c.shape[1] > 0, "CICIDS gagal: shape salah"
# Pastikan ada kelas attack (bukan cuma BENIGN)
assert len(labels_c) > 1, "CICIDS gagal: hanya satu kelas — pastikan file attack dipakai"

# ===========================================================================
# TEST 3 — TF-IDF Triage dengan alert dari distribusi UNSW
# ===========================================================================
print(f"\n{SEP}")
print("TEST 3 — TF-IDF Triage Filter (Alert dari distribusi UNSW)")
print(SEP)

# Buat alert sintetis berdasarkan kategori attack_cat UNSW yang nyata
unsw_attack_cats = [
    "Port Scan", "Port Scan", "Port Scan", "Port Scan", "Port Scan",
    "Port Scan", "Port Scan", "Port Scan", "Port Scan",          # banyak = sering
    "Ransomware Activity", "Ransomware Activity",
    "DDoS", "DDoS", "DDoS",
    "SQL Injection",
    "Malware",
    "Backdoor",
    "Worm Activity",
    "Zero-Day Exploit",                                            # sangat jarang
]
alerts = [
    Alert(f"A{i:03d}", f"10.0.{i//256}.{i%256}", atype, (i % 5) + 1, f"Detected {atype}")
    for i, atype in enumerate(unsw_attack_cats)
]

triage = TFIDFTriageFilter(theta_max_idf=2.5, theta_tfidf=0.03)
results, high_risk = triage.evaluate_batch(alerts)

noise = [r for r in results if r.is_noise]
print(f"[OK] TF-IDF Triage: {len(alerts)} alerts total")
print(f"     Noise/False Positive  : {len(noise)} dieliminasi")
print(f"     High-Risk (-> DIM)     : {len(high_risk)} diteruskan")
for r in results:
    tag = "[NOISE]" if r.is_noise else "[HIGH] "
    print(f"     {tag} | {r.alert.alert_type:25s} | IDF={r.idf_score:.3f} | TF-IDF={r.tfidf_score:.4f}")

# ===========================================================================
# TEST 4 — Splunk BOTS v3 (DIM Training Sequences)
# ===========================================================================
print(f"\n{SEP}")
print("TEST 4 — Splunk BOTS v3 (DIM Sequence Generation)")
print(SEP)
t0 = time.time()

bots = SplunkBOTSLoader(seq_len=20)
train_data, test_data = bots.load(n_synthetic=2000, test_size=0.2)
elapsed = time.time() - t0

print(f"[OK] BOTS sequences generated in {elapsed:.1f}s")
print(f"     Train : {train_data['hist_alert_ids'].shape} sequences")
print(f"     Test  : {test_data['hist_alert_ids'].shape} sequences")
pb_dist = Counter(train_data['target_playbook'].tolist())
top5 = pb_dist.most_common(5)
print(f"     Top-5 target playbooks: {top5}")
assert train_data['hist_alert_ids'].shape[1] == 20, "BOTS: seq_len salah"

# ===========================================================================
# TEST 5 — DIM Forward Pass + Top-K
# ===========================================================================
print(f"\n{SEP}")
print("TEST 5 — DIM Forward Pass + Top-K Prediction")
print(SEP)

num_playbooks = len(PLAYBOOK_ID_MAP)
model = DynamicInterestModel(
    num_alert_types=50,
    num_playbooks=num_playbooks,
    num_tactics=14,
    embed_dim=64,
    lt_heads=4,
    lt_layers=2,
    st_hidden=128,
    st_layers=2,
)
total_params = sum(p.numel() for p in model.parameters())
print(f"     Model parameters: {total_params:,}")

B, T = 4, 20
p = model(
    torch.randint(1, 51,              (B, T)),
    torch.randint(1, num_playbooks+1, (B, T)),
    torch.randint(1, 15,              (B, T)),
    torch.randint(1, 6,               (B, T)),
    torch.randint(1, num_playbooks+1, (B,)),
)
assert p.shape == (B,) and p.min() >= 0 and p.max() <= 1
print(f"[OK] DIM forward pass: shape={p.shape}, values={p.detach().numpy().round(4)}")

top_idx, top_scores = model.predict_top_k(
    torch.randint(1, 51,              (1, T)),
    torch.randint(1, num_playbooks+1, (1, T)),
    torch.randint(1, 15,             (1, T)),
    torch.randint(1, 6,              (1, T)),
    num_playbooks=num_playbooks, k=5,
)
print(f"[OK] Top-5 playbook IDs   : {top_idx.tolist()}")
print(f"[OK] Top-5 playbook scores: {top_scores.detach().numpy().round(3).tolist()}")

# ===========================================================================
# TEST 6 — Ranking Metrics (dengan prediksi nyata dari DIM)
# ===========================================================================
print(f"\n{SEP}")
print("TEST 6 — Ranking Metrics (HR, MAP, NDCG)")
print(SEP)

# Simulasi n_queries prediksi terhadap test_data dari BOTS
n_q = min(200, test_data['hist_alert_ids'].shape[0])
model.eval()
recommendations = []
ground_truths   = test_data['target_playbook'][:n_q].tolist()

with torch.no_grad():
    for i in range(0, n_q, 32):
        end = min(i + 32, n_q)
        top_idx_b, _ = model.predict_top_k(
            test_data['hist_alert_ids'][i:end],
            test_data['hist_playbook_ids'][i:end],
            test_data['hist_tactic_ids'][i:end],
            test_data['hist_severity'][i:end],
            num_playbooks=num_playbooks, k=10,
        )
        recommendations.extend(top_idx_b.tolist())

rm = RankingMetrics()
r  = rm.compute_all(recommendations, ground_truths, k_values=[1, 3, 5, 10])
print(f"[OK] Ranking Metrics on {n_q} test queries (model untrained — baseline):")
for k in [1, 3, 5, 10]:
    print(f"     K={k:2d} | HR={r[f'hit_ratio@{k}']:.4f} | "
          f"MAP={r[f'map@{k}']:.4f} | NDCG={r[f'ndcg@{k}']:.4f}")

# ===========================================================================
# TEST 7 — SOC Operational Metrics
# ===========================================================================
print(f"\n{SEP}")
print("TEST 7 — SOC Operational Metrics (MTTD, MTTR, Workload Reduction)")
print(SEP)

import random
random.seed(42)
soc = SOCOperationalMetrics()
for i in range(50):
    t_start = time.time()
    soc.record_incident(IncidentRecord(
        incident_id=f"INC-{i:04d}",
        detection_start=t_start,
        detection_end=t_start + random.uniform(0.1, 3.0),      # MTTD: 0.1–3s
        response_start=t_start + random.uniform(3, 8),
        response_end=t_start + random.uniform(10, 45),           # MTTR: 10–45s
        manual_actions_automated=random.randint(4, 9),
        manual_actions_total=10,
    ))

res = soc.compute()
mttd = res['mttd_seconds']['mean']
mttr = res['mttr_seconds']['mean']
wlr  = res['analyst_workload_reduction']['mean_pct']
print(f"[OK] MTTD mean : {mttd:.2f}s")
print(f"[OK] MTTR mean : {mttr:.2f}s")
print(f"[OK] Analyst Workload Reduction mean: {wlr:.1f}%")

# ===========================================================================
# SUMMARY
# ===========================================================================
print(f"\n{'='*60}")
print("  ALL TESTS PASSED -- DATASET NYATA TERVALIDASI")
print(f"{'='*60}")
print(f"  UNSW-NB15  : {X_tr.shape[0]:,} train | {X_te.shape[0]:,} test | {len(labels)} classes")
print(f"  CICIDS2017 : {X_tr_c.shape[0]:,} train | {X_te_c.shape[0]:,} test | {len(labels_c)} classes")
print(f"  BOTS SIEM  : {train_data['hist_alert_ids'].shape[0]:,} train sequences")
print(f"  DIM params : {total_params:,}")
print(f"  HR@5       : {r['hit_ratio@5']:.4f} (baseline untrained)")
print(f"  NDCG@5     : {r['ndcg@5']:.4f} (baseline untrained)")
