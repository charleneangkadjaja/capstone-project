# 🎓 Learning Analytics & Gaya Belajar VAK - Capstone Project

## 📁 Struktur Berkas Utama
* `capstone.py`: Skrip utama untuk memuat 3 dataset, prapemrosesan, integrasi data, *clustering* ritme belajar (K-Means), dan ekspor data latih model AI.
* `capstone_EDA.py`: Skrip analisis data untuk menghasilkan grafik distribusi, visualisasi indikator VAK, dan analisis korelasi fitur.
* `.gitignore`: Menyaring folder dataset agar penyimpanan repositori tetap optimal.

## Sumber Dataset
1. Student Education Dataset
2. Student Performance Dataset
3. VAK Text Dataset

## Data Handoff (Output proyek)
Hasil pengeksekusian kode otomatis akan tersimpan di dalam folder `output/` yang terdiri dari:
* **Untuk AI Engineer**: File `X_train_scaled.csv`, `y_train.csv`, `X_test_scaled.csv`, dan `y_test.csv` (siap digunakan untuk tahap pemodelan klasifikasi).
* **Untuk Web Developer**: File `final_dataset_cleaned_with_pace.csv` (master data lengkap dengan cluster kecepatan belajar untuk basis data dashboard).
* **Untuk NLP Modul**: File `master_vak_nlp.csv` (data teks bersih untuk rekomendasi modul).

*Catatan: Dikarenakan ukuran file output (*.csv) cukup besar, folder `output/` disembunyikan lewat `.gitignore` dan berkas fisik data hasil pemrosesan dibagikan antar-tim*

## 📊 Dashboard, A/B Testing & Deployment
Bagian visualisasi data dan pembuatan dashboard interaktif untuk proyek ini dikembangkan menggunakan **Streamlit** oleh rekan tim saya, [Raissa Nadia](https://eduprofile-ai-capstone.streamlit.app/).

* **Repository Dashboard:** [RaissaNadia/EduProfile](https://github.com/RaissaNadia/EduProfile-AI)
