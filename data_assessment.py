"""
DATA ASSESSMENT SCRIPT — EduProfile AI
Jalankan script ini SEBELUM preprocessing untuk mendokumentasikan
kualitas data mentah dari masing-masing dataset.

Cara pakai:
    python data_assessment.py

Pastikan folder 'sourcedataset/' berisi:
    - student_education_dataset.csv
    - student_performance.csv
    - dataset.csv
"""

import pandas as pd
import sys
import os
import io
import numpy as np

SEP = "=" * 65

def assess_dataset(df, nama):
    print(f"\n{SEP}")
    print(f"  ASSESSMENT: {nama}")
    print(SEP)

    # ── 2.2.1 Dimensi ──────────────────────────────────────────────
    print(f"\n[1] DIMENSI DATA")
    print(f"    Jumlah baris   : {df.shape[0]:,}")
    print(f"    Jumlah kolom   : {df.shape[1]}")
    print(f"    Nama kolom     : {list(df.columns)}")

    # ── 2.2.2 Tipe Data ────────────────────────────────────────────
    print(f"\n[2] TIPE DATA PER KOLOM")
    for col, dtype in df.dtypes.items():
        print(f"    {col:<35} : {str(dtype):<12}", end="")
        # Flag tipe tidak sesuai (object di kolom yang harusnya numerik)
        if dtype == 'object':
            unique_sample = df[col].dropna().unique()[:3]
            print(f"  ← object | contoh: {unique_sample}", end="")
        print()

    # ── 2.2.3 Missing Values ───────────────────────────────────────
    print(f"\n[3] MISSING VALUES")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'Jumlah NaN': missing, 'Persentase (%)': missing_pct})
    missing_df = missing_df[missing_df['Jumlah NaN'] > 0]
    if missing_df.empty:
        print("    ✅ Tidak ada missing values.")
    else:
        print(f"    ⚠️  Ditemukan missing values:")
        print(missing_df.to_string(index=True))
    print(f"    Total sel kosong: {df.isnull().sum().sum()}")

    # ── 2.2.4 Duplikat ─────────────────────────────────────────────
    print(f"\n[4] DATA DUPLIKAT")
    n_dup = df.duplicated().sum()
    if n_dup == 0:
        print("    ✅ Tidak ada baris duplikat.")
    else:
        print(f"    ⚠️  Ditemukan {n_dup:,} baris duplikat ({n_dup/len(df)*100:.2f}%)")
        print("    Contoh duplikat:")
        print(df[df.duplicated(keep=False)].head(4).to_string())

    # ── 2.2.5 Statistik Deskriptif ─────────────────────────────────
    print(f"\n[5] STATISTIK DESKRIPTIF (Kolom Numerik)")
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        print(df[num_cols].describe().round(3).to_string())

    # ── 2.2.6 Outlier Detection (IQR Method) ──────────────────────
    print(f"\n[6] DETEKSI OUTLIER (Metode IQR)")
    outlier_summary = []
    for col in num_cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_out = df[(df[col] < lower) | (df[col] > upper)][col].count()
        if n_out > 0:
            outlier_summary.append({
                'Kolom': col,
                'Q1': round(Q1, 3),
                'Q3': round(Q3, 3),
                'IQR': round(IQR, 3),
                'Batas Bawah': round(lower, 3),
                'Batas Atas': round(upper, 3),
                'Jumlah Outlier': n_out,
                'Persen (%)': round(n_out / len(df) * 100, 2)
            })
    if outlier_summary:
        print(pd.DataFrame(outlier_summary).to_string(index=False))
    else:
        print("    ✅ Tidak ada outlier signifikan terdeteksi.")

    # ── 2.2.7 Nilai Unik Kolom Kategorik ──────────────────────────
    cat_cols = df.select_dtypes(include='object').columns
    if len(cat_cols) > 0:
        print(f"\n[7] NILAI UNIK KOLOM KATEGORIK")
        for col in cat_cols:
            uniq = df[col].unique()
            print(f"    {col} ({df[col].nunique()} nilai unik): {list(uniq[:10])}", end="")
            if df[col].nunique() > 10:
                print(f" ... (+{df[col].nunique()-10} lagi)", end="")
            print()

    # ── Ringkasan Masalah ──────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  RINGKASAN MASALAH — {nama}")
    print(f"{'─'*65}")
    masalah = []
    if missing_df.shape[0] > 0:
        masalah.append(f"Missing values pada {list(missing_df.index)}")
    if n_dup > 0:
        masalah.append(f"Duplikat: {n_dup} baris")
    if outlier_summary:
        kolom_out = [o['Kolom'] for o in outlier_summary]
        masalah.append(f"Outlier pada kolom: {kolom_out}")
    wrong_type = [c for c in df.columns if df[c].dtype == 'object'
                  and c.lower() not in ['learningstyle','type','sentence','gender','major','school']]
    if wrong_type:
        masalah.append(f"Kolom object yang mungkin perlu dicek: {wrong_type}")

    if masalah:
        for i, m in enumerate(masalah, 1):
            print(f"  {i}. ⚠️  {m}")
    else:
        print("  ✅ Tidak ada masalah kualitas data yang ditemukan.")

    return {
        'nama': nama,
        'n_baris': df.shape[0],
        'n_kolom': df.shape[1],
        'missing_total': int(df.isnull().sum().sum()),
        'missing_kolom': list(missing_df.index),
        'duplikat': int(n_dup),
        'outlier_kolom': [o['Kolom'] for o in outlier_summary],
        'outlier_detail': outlier_summary,
    }
# ── Auto-save ke output/ ───────────────────────────────────────────────────────
def run_and_save():
    OUTPUT_DIR  = "output"
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "assessment_results.txt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Tangkap semua print ke buffer
    buffer = io.StringIO()
    sys.stdout = buffer

    # ── Jalankan assessment ──────────────────────────────────────────────────
    print(SEP)
    print("  DATA ASSESSMENT — EduProfile AI CC26-PSU099")
    print("  Hasil otomatis disimpan ke: output/assessment_results.txt")
    print(SEP)

    datasets = {
        "Dataset 1: student_education_dataset.csv": "sourcedataset/student_education_dataset.csv",
        "Dataset 2: student_performance.csv":       "sourcedataset/student_performance.csv",
        "Dataset NLP VAK: dataset.csv":             "sourcedataset/dataset.csv",
    }

    results = []
    for nama, path in datasets.items():
        try:
            df = pd.read_csv(path)
            r  = assess_dataset(df, nama)
            results.append(r)
        except FileNotFoundError:
            print(f"\n[!] File tidak ditemukan: {path}")
        except Exception as e:
            print(f"\n[!] Error saat memuat {path}: {e}")

    # Tabel Ringkasan
    print(f"\n{SEP}")
    print("  TABEL RINGKASAN AKHIR (untuk Laporan Seksi 2.2)")
    print(SEP)
    print(f"\n{'Dataset':<40} {'Baris':>6} {'Kolom':>6} {'Missing':>8} {'Duplikat':>9} {'Outlier Kolom'}")
    print("-" * 90)
    for r in results:
        out_str      = ", ".join(r['outlier_kolom']) if r['outlier_kolom'] else "Tidak ada"
        miss_str     = str(r['missing_total'])
        nama_singkat = r['nama'].split(':')[1].strip() if ':' in r['nama'] else r['nama']
        print(f"  {nama_singkat:<38} {r['n_baris']:>6,} {r['n_kolom']:>6} {miss_str:>8}  {r['duplikat']:>8}  {out_str}")

    print(f"\n[OK] Assessment selesai. File disimpan di: {OUTPUT_FILE}")

    # ── Kembalikan stdout & tulis ke file ──────────────────────────────────
    output_text = buffer.getvalue()
    sys.stdout  = sys.__stdout__

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(output_text)

    # Tampilkan juga ke terminal
    print(output_text)
    print(f"[OK] File tersimpan di: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_and_save()