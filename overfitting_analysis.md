# Analisis Risiko Overfitting — Diagnosis Jujur

## Ringkasan

> **Bukan overfitting klasik (train loss << test loss), tapi ada tiga masalah yang lebih serius: trivialitas pola, same-distribution test, dan simulation bias. Ini dapat menyebabkan inflasi metrik yang tidak akan bertahan di deployment nyata.**

---

## 1. DIM Training — Masalah Trivialitas Pola (Bukan Overfitting Klasik)

### Apa yang terjadi di `_generate_sequences()`:

```python
# Line 622-634 di dataset_loader.py
dominant_pb = random.choices(weighted_pbs, weights=weighted_vals, k=1)[0]

hist_playbook = [
    dominant_pb if np.random.random() > 0.3   # 70% = dominant_pb
    else random.choice(playbook_ids)           # 30% = random noise
    for _ in range(seq_len)
]

target_playbook = dominant_pb   # <-- TARGET SELALU = DOMINANT_PB
```

### Masalah: Pola terlalu mudah dipelajari

| Apa yang harus diprediksi model | Informasi yang tersedia |
|--------------------------------|-------------------------|
| `target_playbook` | `hist_playbook` (70% berisi target itu sendiri) |

**Artinya:** bahkan tanpa Deep Learning, algoritma sederhana seperti `argmax(Counter(hist_playbook))` akan mencapai HR@5 ~100% pada data ini.

Model DIM tidak perlu belajar "pola respons insiden kompleks" — ia cukup belajar **"lihat hist, pilih yang paling sering muncul"**.

### Bukti: Train/Test dari Distribusi yang Sama

```
Split: split_idx = int(len(all_data) * 0.8)
```

- Training: 8000 sequence dari `SplunkBOTSLoader.random_state=42`
- Test: 2000 sequence dari **generator yang sama, seed yang sama**
- Tidak ada distribusi shift, tidak ada pola baru

Ini **BUKAN overfitting** (model tidak hafal data), tapi **pola yang dipelajari tidak generalize ke kasus nyata** karena pola dunia nyata jauh lebih kompleks.

### Mengapa HR@5 = 88.6% bisa inflasi?

Jika kita uji dengan simple baseline:

```python
# Baseline sederhana: pilih playbook paling sering di history
def simple_majority_baseline(hist_playbook_ids):
    from collections import Counter
    return Counter(hist_playbook_ids).most_common(5)
```

Estimasi HR@5 baseline ini: **~85-90%** — hampir sama dengan model DIM trained!

**Implikasi:** DIM mungkin hanya belajar "majority vote from history" — bukan Deep Learning yang sesungguhnya.

---

## 2. SOC Operational Metrics — Simulation Bias (Bukan Overfitting)

### Masalah: Parameter simulasi disetel untuk menghasilkan hasil yang diinginkan

```python
# test_modules.py dan main.py
manual_actions_automated=random.randint(8, 10),  # kita yang menentukan 8-10
manual_actions_total=10,
```

Ini **bukan overfitting** secara teknis, tapi **circular reasoning**:
1. Kita asumsikan sistem mengotomasi 80-100% langkah
2. Kita hitung MTTR reduction = 97% berdasarkan asumsi itu
3. Kita klaim "sistem melampaui 81% dari [13][14]"

**Ini bukan pengukuran empiris — ini proyeksi asumsi.**

Nilai MTTR reduction 97% HANYA VALID jika:
- Sistem benar-benar mengotomasi 8-10 langkah dari 10
- Baseline MTTR 480s representatif untuk environment target
- Tidak ada latency overhead dari inference (network, I/O, dll.)

---

## 3. TF-IDF Triage — Tidak Bermasalah

TF-IDF adalah algoritma statistik deterministik tanpa learning → tidak bisa overfit.
Hasil triase bergantung sepenuhnya pada distribusi alert di batch.

---

## 4. Diagnosa Per Komponen

| Komponen | Overfit? | Masalah Aktual | Severity |
|----------|----------|----------------|----------|
| DIM HR@5 = 88.6% | Tidak overfit teknis | Pola trivial (70% signal) → inflasi metrik | **Tinggi** |
| DIM NDCG@5 = 0.8646 | Tidak overfit teknis | Same-distribution test | **Sedang** |
| MTTR 97% | Bukan overfitting | Simulation bias (parameter sengaja diset) | **Tinggi** |
| Workload 91% | Bukan overfitting | Simulation bias | **Sedang** |
| TF-IDF | Tidak berlaku | Tidak ada learning | **Tidak ada** |

---

## 5. Cara Memverifikasi — Uji yang Perlu Dilakukan

### Uji 1: Simple Majority Baseline (kritis)
Jika `simple_majority_vote` mendapat HR@5 ≥ 80%, maka DIM tidak menambah nilai signifikan.

```python
def majority_vote_baseline(hist_playbook_ids, k=5):
    from collections import Counter
    # Top-K playbook paling sering di history
    return [pb for pb, _ in Counter(hist_playbook_ids.tolist()).most_common(k)]
```

### Uji 2: Cross-Distribution Test
Latih pada distribusi A (uniform), test pada distribusi B (BOTS-weighted). Jika HR@5 turun drastis, model tidak generalize.

### Uji 3: Pola Acak Total
Test dengan history yang di-shuffle secara acak. Jika HR@5 tidak turun banyak, model bergantung pada statistik sederhana, bukan pola urutan (sequential patterns).

### Uji 4: Ubah Persentase Noise
Test dengan `70%` → `50%` dominant_pb. Jika HR@5 turun drastis, model hanya belajar majority vote.

---

## 6. Apa yang Seharusnya Diklaim di Paper

### Klaim yang Aman:
- "HR@5 = 88.6% pada dataset sintetis berbasis distribusi BOTS — menunjukkan kemampuan sistem mempelajari pola rekomendasi playbook dari histori sekuensial"
- "MTTR reduction ~97% sebagai proyeksi operasional berdasarkan asumsi otomasi 80-100% langkah manual"
- "Evaluasi pada data sintetis ini bersifat proof-of-concept; generalisasi ke deployment nyata memerlukan validasi dengan log SOC historis aktual"

### Klaim yang Perlu Dikecualikan atau Diberi Caveat:
- Jangan klaim "HR@5 88.6% melampaui 84% dari [10]" tanpa catatan bahwa keduanya diuji pada domain yang berbeda
- Jangan klaim "MTTR reduction 97% > 81% [13][14]" sebagai hasil empiris — ini simulasi
- Tambahkan limitation section yang menyebutkan: *"hasil DIM diukur pada data sintetis dengan pola yang sengaja disederhanakan (70% dominant signal), sehingga performa aktual di lingkungan produksi yang lebih noisy belum dapat dikonfirmasi"*

---

## 7. Mitigasi yang Bisa Dilakukan

### Prioritas Tinggi (sebelum submit paper):
1. **Jalankan majority vote baseline** dan laporkan gap DIM vs baseline
2. **Ubah komposisi sequence**: turunkan dari 70%/30% ke 50%/50% dan lihat apakah HR@5 tetap tinggi
3. **Frame dengan jujur**: gunakan kata "sintetis", "proyeksi", "proof-of-concept"

### Prioritas Sedang:
4. **Uji cross-distribution**: latih pada 60% BOTS-weighted, test pada 40% uniform (atau sebaliknya)
5. **Tambahkan Transformer-only vs LSTM-only ablation**: buktikan komponen Transformer+LSTM memberi keunggulan vs model lebih sederhana

### Justifikasi yang Tetap Valid:
- **Arsitektur novelty**: kombinasi TF-IDF + DIM + HITL belum ada → ini research gap yang valid
- **NDCG@5 = 0.8646**: sebagai ranking quality indicator tetap informatif
- **Computational efficiency**: TF-IDF sebelum DL → valid dan terverifikasi
- **HITL feedback loop**: mekanismenya sudah benar secara arsitektural

---

## 8. Kesimpulan

**Hasil bukan overfit dalam arti teknis** (train loss tidak jauh lebih rendah dari test loss, model tidak hafal training data). Tapi ada **dua risiko penelitian yang lebih serius**:

1. **Inflasi metrik DIM** akibat pola sintetis yang terlalu mudah (70% dominant signal). Solusi: jalankan majority baseline dan laporkan gap-nya.

2. **Simulation bias pada SOC metrics** — parameter simulasi disetel untuk menghasilkan angka yang mengalahkan [13][14]. Solusi: frame sebagai proyeksi/asumsi, bukan hasil empiris.

Riset ini tetap valid sebagai **proof-of-concept arsitektur multi-modal** — kontribusi utamanya adalah integrasi novelty (NER+KG+TF-IDF+DIM+HITL), bukan klaim performa absolut.
