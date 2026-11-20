# SIMULASI LKS DIKMEN — KLASIFIKASI PENGADUAN PENDIDIKAN (NLP)

**Status:** latihan simulasi mandiri; **BUKAN soal atau dataset resmi LKS**.

## Acuan desain
- Kisi-kisi AI LKS Dikmen Provinsi Jawa Timur 2025: EDA 20%, preprocessing 20%, machine learning 40%, evaluasi 20%. Dokumen asli menggunakan data numerik dan KNN; ini adaptasi khusus teks, bukan reproduksi soal.
  https://id.scribd.com/document/866366688/4-Artificial-Intelligence-Kisi2-Deskripsi-Teknis-LKS-Dikmen-Tk-Jatim-2025
- Panduan Jatim 2021 menyebut data preparation, preprocessing, training classification, testing classification, dan presentation sebagai modul.
  https://id.scribd.com/document/561806665/Panduan-Teknis-Dan-Kisi-Kisi-LKS-JATIM-2021

## Kasus bisnis
Tim layanan pendidikan menerima pesan pengaduan tidak berstruktur. Bangun model **multiclass single-label** yang mengklasifikasikan **keluhan utama yang harus ditindaklanjuti**. Suatu pesan boleh menyebut beberapa masalah, tetapi label menunjukkan tindakan utama yang diminta, bukan semua kata yang muncul.

## Data & struktur
Total 4.000 pesan **fiktif/sintetis**, dibuat untuk latihan. Tidak berisi pengaduan atau data pribadi nyata.
- Train: 2,800 berlabel
- Validation: 601 berlabel
- Test: 599 tanpa label (kunci tersedia terpisah dan jangan dibuka sebelum selesai)
- File CSV UTF-8 BOM agar nyaman dibuka Excel/Pandas
- Kolom: `id_laporan`, `pesan`, `tanggal`, `kanal`, `wilayah`, `peran_pelapor`, `case_group`, `kategori` (hanya train+validation)

## Label
- `AKUN_AKSES`: 640 contoh total
- `SISTEM_JARINGAN`: 570 contoh total
- `DOKUMEN_ADMIN`: 520 contoh total
- `AKADEMIK_PEMBELAJARAN`: 480 contoh total
- `BEASISWA_BANTUAN`: 450 contoh total
- `KEUANGAN_PEMBAYARAN`: 410 contoh total
- `SARANA_PRASARANA`: 380 contoh total
- `KEAMANAN_KETERTIBAN`: 320 contoh total
- `PENDAFTARAN_MUTASI`: 230 contoh total

## Design challenge: serius tetapi proporsional
- Distribusi kelas tidak seimbang (rasio terbesar/terkecil ≈ 2.8 : 1)
- Keluhan bahasa Indonesia baku, percakapan, singkatan, salah ketik, HTML sporadis, panjang bervariasi
- Sering ada isu kedua sebagai konteks; beberapa kalimat eksplisit menyatakan *bukan* isu tersebut
- Ada duplikat persis dalam train; juga sejumlah pesan kosong pada train
- Sebagian keluarga skenario di validation dan test **tidak terdapat** dalam train (sekitar sepertiga topik keluarga per kelas); sisanya topik serupa tetapi pesan dan incident group berbeda. Ini menguji sebagian generalisasi sekaligus menjaga kesulitan tetap seimbang.
- `case_group` adalah penanda kelompok laporan terkait: **jangan gunakan sebagai fitur prediktor**; digunakan untuk audit duplikasi/splitting bila melakukan CV
- `tanggal`, `kanal`, `wilayah`, `peran_pelapor` adalah metadata tidak dirancang sebagai sumber kelas yang stabil; latih model teks utama dengan `pesan`
- Semua kategori ditetapkan pada saat membuat insiden sintetik sehingga merupakan label internal skenario. Tidak sama dengan label hasil anotasi manusia di dunia nyata.

## Rules
1. Hanya **classical machine learning** (tanpa Transformer/BERT/CNN/LSTM/LLM untuk inferensi).
2. Boleh pakai scikit-learn, pandas, numpy, matplotlib, seaborn, Gradio.
3. Dilarang fit vectorizer, feature selector, ataupun tuning model menggunakan validation/test.
4. Data validation boleh dipakai untuk memilih model, tetapi jangan menganggap skor validation sebagai test final.
5. Jangan melihat file kunci test sebelum model final dan file prediksi dibuat.
6. **Wajib** menyimpan label prediksi di `submission.csv` yang kolomnya tepat `id_laporan,kategori`.
7. Jangan memasukkan `case_group` atau `id_laporan` sebagai fitur — keduanya identifier. Jelaskan bila menggunakan fitur metadata lainnya.

## Tugas berurutan
### A. EDA (20 poin)
- Audit tipe data, missing/blank, panjang pesan, duplikasi keseluruhan dan per `pesan`, jumlah kelas, distribusi kanal/wilayah
- Analisis panjang teks per kelas, istilah yang sama antara kelas berbeda, contoh sulit/ambigu
- Jelaskan potensi shortcut learning dan kebocoran antar split
### B. Preprocessing (20 poin)
- Buat baseline minimal (lowercase + TF-IDF) sebelum preprocessing agresif
- Uji minimal dua pilihan tokenisasi/normalisasi; ablation stopword, unigram/bigram, word vs char n-gram
- Terapkan `Pipeline` agar transformasi belajar hanya dari train; tangani NaN, HTML ringan, slang secara sadar
- Audit identifikasi duplikat dan hubungannya dengan `case_group`
### C. Modeling (40 poin)
- DummyClassifier sebagai baseline
- Bandingkan sedikitnya MultinomialNB, LogisticRegression, dan LinearSVC
- Gunakan stratified atau group-aware CV sesuai hipotesis; tunjukkan tradeoff dan waktu pelatihan
- Hyperparameter tuning yang terarah; jangan buru-buru memilih model dari accuracy saja
### D. Evaluation (20 poin)
- Laporkan accuracy, macro-F1, weighted-F1, recall tiap kelas, confusion matrix
- Analisis 20 false predictions, termasuk apakah isu sekunder lebih kuat daripada isu utama
- Putuskan model akhir menggunakan validation; lalu buat test submission satu kali
- Bonus opsional: deploy dengan Gradio, tunjukkan hasil untuk input yang belum pernah dilatih

## Kriteria kelulusan (target latihan, BUKAN ambang resmi LKS)
- Nilai utama: kualitas metodologi dan validitas evaluasi, **bukan** sekadar angka skor
- Macro-F1 pada test dianjurkan sebagai metrik utama untuk memperhatikan kelas minoritas
- Jika validation jauh lebih tinggi dari test, jelaskan generalization gap dan scenario shift
- Jangan menyatakan dataset ini adalah data riil atau soal resmi

## Ukur hasil test (setelah selesai)
Simpan prediksi test ke `submission.csv`, lalu gunakan `evaluasi_submission.py` bersama file kunci test yang disimpan **terpisah dari arsip utama**. Template yang harus diisi ada pada `submission_template.csv`.
