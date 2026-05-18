# Walkthrough — Incident Response Playbook Recommendation System

## Apa yang Dibangun

Implementasi lengkap framework multi-modal dari metodologi penelitian Section 2, mencakup seluruh 5 sub-section dalam arsitektur berjenjang.

## Struktur File

```
Code/
├── main.py                          # Orchestration pipeline (entry point)
├── train_dim.py                     # Training script untuk DIM
├── test_modules.py                  # Validasi seluruh modul
├── requirements.txt                 # Dependencies
└── src/
    ├── feature_extraction/
    │   ├── ner_extractor.py         # [2.1] NER + POS Tagging
    │   └── knowledge_graph.py       # [2.1] RDF Triplet + MITRE ATT&CK
    ├── triage/
    │   └── tfidf_filter.py          # [2.2] TF-IDF Automated Triage
    ├── models/
    │   └── dim.py                   # [2.3] Dynamic Interest Model
    ├── validation/
    │   └── hitl_validator.py        # [2.4] Human-in-the-Loop + BCE Loss
    ├── data/
    │   └── dataset_loader.py        # [2.5] BOTS/UNSW/CICIDS Loader + SMOTE
    └── evaluation/
        └── metrics.py               # [2.5] Classification + Ranking + SOC
```

## Hasil Validasi

```
[OK] TF-IDF Triage     : 14/14 alerts dievaluasi
[OK] DIM forward pass  : output shape=[2], values=[0.1147, 0.0676]
[OK] Ranking Metrics   : HR@5=1.000, MAP@5=1.000, NDCG@5=1.000 (dummy data)
[OK] SplunkBOTS Loader : train=[80,10], test=[20,10] sequences
[OK] DIM Top-5 Predict : indices + probability scores dihasilkan
```

## Cara Menjalankan

### 1. Install dependencies (gunakan Python 3.11)
```bash
py -3.11 -m pip install -r requirements.txt
py -3.11 -m spacy download en_core_web_sm
```

### 2. Training DIM dengan Splunk BOTS v3
```bash
py -3.11 train_dim.py
```

### 3. Jalankan full pipeline demo
```bash
py -3.11 main.py --mode pipeline
```

### 4. Evaluasi saja (load checkpoint terbaik)
```bash
py -3.11 main.py --mode eval
```

## Pemetaan ke Metodologi

| Section | Implementasi | File |
|---------|-------------|------|
| 2.1 | NER (spaCy + regex), POS Tagging | `ner_extractor.py` |
| 2.1 | RDF Triplet (S,P,O) + MITRE ATT&CK Ontology | `knowledge_graph.py` |
| 2.2 | TF-IDF filter: IDF > θ_idf AND TF-IDF < θ_tfidf | `tfidf_filter.py` |
| 2.3 | Embedding + Transformer (long-term) + LSTM forget gate (short-term) + MLP | `dim.py` |
| 2.4 | HITL, BCE Loss, backpropagation, replay buffer | `hitl_validator.py` |
| 2.5 | SMOTE, Precision/Recall/F1/ROC-AUC, HR/MAP/NDCG, MTTD/MTTR/Workload | `metrics.py` + `dataset_loader.py` |

## Catatan Teknis

- **Python yang digunakan**: `py -3.11` (Python 3.11 via Windows Python Launcher)
- **PyTorch version**: 2.12.0+cpu
- **MITRE ATT&CK**: File STIX `enterprise-attack.json` dibaca langsung dari `Datasets/MITRE ATT&CK/`
- **Splunk BOTS**: Karena format Splunk lookup tidak memiliki label respons inheren, sekuensi dihasilkan secara sintetis berdasarkan distribusi eventcode
- **UNSW-NB15 & CICIDS2017**: File CSV besar dibaca dengan `nrows` untuk efisiensi; SMOTE diterapkan dengan K-NN auto-adjustment

