# -*- coding: utf-8 -*-
"""DS_CAPSTONE_INTEGRATED
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

"""### Load Datasets"""

try:
    df_education = pd.read_csv('sourcedataset/student_education_dataset.csv')
    df_vak = pd.read_csv('sourcedataset/dataset.csv')
    df_performance = pd.read_csv('sourcedataset/student_performance.csv')
    print("✅ Berhasil memuat semua dataset.")
except FileNotFoundError:
    print("❌ Error: File CSV tidak ditemukan. Pastikan path 'sourcedataset/' sudah benar.")

"""### Data Pre-Integration & Feature Engineering Per Dataset
Agar index data tidak rusak dan tidak menghasilkan nilai NaN, kita hitung Indikator VAK kustom pada masing-masing dataset asal sebelum digabungkan.
"""

# === PREPROCESSING DATASET 1: Education Dataset ===
df_edu_proc = df_education.copy()

# Normalisasi skala fitur ke range 1-5
df_edu_proc['DeviceUsage_Norm'] = (df_edu_proc['DeviceUsage'] / df_edu_proc['DeviceUsage'].max()) * 4 + 1
df_edu_proc['Edutech_Norm'] = (df_edu_proc['AcademicScore'] / df_edu_proc['AcademicScore'].max()) * 4 + 1
df_edu_proc['Resources_Norm'] = (df_edu_proc['AttendanceRate'] * 4) + 1

df_edu_proc['CourseParticipation_Norm'] = (df_edu_proc['CourseParticipation'] / df_edu_proc['CourseParticipation'].max()) * 4 + 1
df_edu_proc['EmotionEngagement_Norm'] = (df_edu_proc['EmotionEngagement'] / df_edu_proc['EmotionEngagement'].max()) * 4 + 1
df_edu_proc['Discussion_Norm'] = (df_edu_proc['CourseParticipation_Norm'] * 0.8) + (df_edu_proc['EmotionEngagement_Norm'] * 0.2)

df_edu_proc['PhysicalActivity_Norm'] = (df_edu_proc['PhysicalActivity'] / df_edu_proc['PhysicalActivity'].max()) * 4 + 1
# Dataset 1 tidak punya Extracurricular langsung, kita beri nilai tengah default
df_edu_proc['Extracurricular_Norm'] = 3.0
df_edu_proc['StudentPerformance'] = df_edu_proc['Edutech_Norm']

# Kalkulasi Indikator Berbobot Utama (Skala disetarakan 1-5 dengan kurung pembagi yang benar)
df_edu_proc['Indikator_Visual'] = ((df_edu_proc['Edutech_Norm'] * 3) + (df_edu_proc['DeviceUsage_Norm'] * 1) + (df_edu_proc['Resources_Norm'] * 3)) / 7
df_edu_proc['Indikator_Auditory'] = ((df_edu_proc['Discussion_Norm'] * 3) + (df_edu_proc['CourseParticipation_Norm'] * 1) + (df_edu_proc['EmotionEngagement_Norm'] * 1)) / 5
df_edu_proc['Indikator_Kinestetik'] = ((df_edu_proc['PhysicalActivity_Norm'] * 1) + (df_edu_proc['Extracurricular_Norm'] * 3)) / 4

# === PREPROCESSING DATASET 2: Performance Dataset ===
df_perf_proc = df_performance.copy()

# Penyesuaian Skala dasar
df_perf_proc['AttendanceRate'] = df_perf_proc['Attendance'] / 100
df_perf_proc = df_perf_proc.rename(columns={
    'ExamScore': 'AcademicScore',
    'FinalGrade': 'StudentPerformance'
})
df_perf_proc['DeviceUsage_Norm'] = 3.0
df_perf_proc['DeviceUsage'] = 3.0 # cadangan agar kolom dasar tetap terbentuk

# Mapping gaya belajar dari angka ke label
mapping_styles = {0: 'Auditory', 1: 'Kinesthetic', 2: 'Visual'}
df_perf_proc['LearningStyle'] = df_perf_proc['LearningStyle'].map(mapping_styles)

# Normalisasi skala fitur ke range 1-5
# df_perf_proc['DeviceUsage_Norm'] = (df_perf_proc['DeviceUsage'] / df_perf_proc['DeviceUsage'].max()) * 4 + 1
df_perf_proc['StudentPerformance'] = (df_perf_proc['StudentPerformance'] / 3) * 4 + 1
df_perf_proc['Edutech_Norm'] = (df_perf_proc['EduTech'] / df_perf_proc['EduTech'].max() * 4 + 1) if 'EduTech' in df_perf_proc.columns else (df_perf_proc['AcademicScore'] / df_perf_proc['AcademicScore'].max() * 4 + 1)
df_perf_proc['Resources_Norm'] = (df_perf_proc['Resources'] / df_perf_proc['Resources'].max() * 4 + 1) if 'Resources' in df_perf_proc.columns else (df_perf_proc['AttendanceRate'] * 4 + 1)

# Fitur spesifik yang absen di dataset 2 diisi dengan nilai tengah (3.0)
df_perf_proc['CourseParticipation_Norm'] = (df_perf_proc['OnlineCourses'] / df_perf_proc['OnlineCourses'].max() * 4 + 1) if 'OnlineCourses' in df_perf_proc.columns else 3.0
df_perf_proc['EmotionEngagement_Norm'] = 3.0
df_perf_proc['Discussion_Norm'] = (df_perf_proc['Discussions'] / df_perf_proc['Discussions'].max() * 4 + 1) if 'Discussions' in df_perf_proc.columns else 3.0

df_perf_proc['PhysicalActivity_Norm'] = 3.0
df_perf_proc['Extracurricular_Norm'] = (df_perf_proc['Extracurricular'] * 0.9) + 1

""" Dokumentasi Imputasi Nilai Default 3.0"""
# Fitur yang tidak ada di dataset asal diisi dengan nilai tengah 3.0 untuk menjaga keseimbangan skala 1-5 dan mencegah bias ekstrem pada model AI klasifikasi nanti.
fitur_imputasi = {
    'DeviceUsage_Norm': 'Tidak ada kolom DeviceUsage di dataset 2, jadi diisi dengan nilai tengah 3.0',
    'EmotionEngagement_Norm': 'Tidak ada kolom EmotionEngagement di dataset 2, jadi diisi dengan nilai tengah 3.0',
    'PhysicalActivity_Norm': 'Tidak ada kolom PhysicalActivity di dataset 2, jadi diisi dengan nilai tengah 3.0',
}
n_imputasi = len(df_perf_proc)
print(f"\n Catatan Imputasi Dataset 2 (student performance): ")
print(f" Jumlah baris terdampak : {n_imputasi} baris")
print(f" Fitur yang diimputasi : {list(fitur_imputasi.keys())}")
print(f" Nilai default : 3.0 (nilai tengah skala 1-5)")
for fitur, alasan in fitur_imputasi.items():
    print(f"   • {fitur:<30} → {alasan}")

# Kalkulasi Indikator Berbobot Utama (Skala disetarakan 1-5 dengan kurung pembagi yang benar)
df_perf_proc['Indikator_Visual'] = ((df_perf_proc['Edutech_Norm'] * 3) + (df_perf_proc['DeviceUsage_Norm'] * 1) + (df_perf_proc['Resources_Norm'] * 3)) / 7
df_perf_proc['Indikator_Auditory'] = ((df_perf_proc['Discussion_Norm'] * 3) + (df_perf_proc['CourseParticipation_Norm'] * 1) + (df_perf_proc['EmotionEngagement_Norm'] * 1)) / 5
df_perf_proc['Indikator_Kinestetik'] = ((df_perf_proc['PhysicalActivity_Norm'] * 1) + (df_perf_proc['Extracurricular_Norm'] * 3)) / 4


"""### Data Integration"""
# Filter kolom seragam yang akan digabungkan ke df_combined
cols_to_combine = [
    'AcademicScore', 'AttendanceRate', 'StudentPerformance', 'DeviceUsage', 'LearningStyle',
    'Indikator_Visual', 'Indikator_Auditory', 'Indikator_Kinestetik'
]
# Ganti fitur mentah pada penumpukan dengan fitur ternormalisasi skala 1-5
df_edu_ready = df_edu_proc.copy()
df_edu_ready['AcademicScore'] = df_edu_ready['Edutech_Norm']
df_edu_ready['AttendanceRate'] = df_edu_ready['Resources_Norm']
df_edu_ready['DeviceUsage'] = df_edu_ready['DeviceUsage_Norm']
df_edu_ready['StudentPerformance'] = df_edu_ready['StudentPerformance']

df_perf_ready = df_perf_proc.copy()
df_perf_ready['AcademicScore'] = df_perf_ready['Edutech_Norm']
df_perf_ready['AttendanceRate'] = df_perf_ready['Resources_Norm']
df_perf_ready['DeviceUsage'] = df_perf_ready['DeviceUsage_Norm']
df_perf_ready['StudentPerformance'] = df_perf_ready['StudentPerformance']

# Menggabungkan data secara vertikal dengan skala yang seragam
df_combined = pd.concat([df_edu_ready[cols_to_combine], df_perf_ready[cols_to_combine]], axis=0, ignore_index=True)

# Imputation & Pembersihan Akhir Data Gabungan
df_combined.dropna(subset=['LearningStyle'], inplace=True)
df_combined.fillna(df_combined.median(numeric_only=True), inplace=True)
df_combined['AttendanceRate'] = df_combined['AttendanceRate'].round(2)
df_combined.drop_duplicates(inplace=True)
df_combined['LearningStyle'] = df_combined['LearningStyle'].str.strip().str.capitalize()
df_combined = df_combined[df_combined['AcademicScore'] >= 0]

print(f"✅ Total data gabungan bersih: {len(df_combined)} baris.")

"""### IMPLEMENTASI FITUR BARU: Learning Pace (Unsupervised K-Means)"""
# Memilih fitur esensial penentu ritme belajar
pace_features = ['AcademicScore', 'AttendanceRate', 'DeviceUsage']
scaler_pace = StandardScaler()
df_pace_scaled = scaler_pace.fit_transform(df_combined[pace_features])

# === VALIDASI JUMLAH CLUSTER OPTIMAL (Elbow + Silhouette) ===
print("\n📊 Validasi K-Means — Mencari jumlah cluster optimal...")
inertias    = []
sil_scores  = []
k_range     = range(2, 7)

for k in k_range:
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(df_pace_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(df_pace_scaled, labels))

print(f"{'K':>4} | {'Inertia':>10} | {'Silhouette':>10}")
print("-" * 30)
for k, inertia, sil in zip(k_range, inertias, sil_scores):
    marker = " ← optimal" if sil == max(sil_scores) else ""
    print(f"{k:>4} | {inertia:>10.2f} | {sil:>10.4f}{marker}")

best_k = list(k_range)[sil_scores.index(max(sil_scores))]
print(f"\n✅ Silhouette Score tertinggi pada K={best_k} (nilai: {max(sil_scores):.4f})")
if best_k == 3:
    print("✅ Konfirmasi: K=3 yang digunakan sudah optimal berdasarkan Silhouette Score.")
else:
    print(f"Catatan: K optimal adalah {best_k}, namun proyek ini tetap menggunakan K=3")
    print("karena sesuai dengan konsep Slow/Medium/Fast yang bermakna secara domain.")

# Latih K-Means untuk membentuk 3 kelompok kecepatan belajar
kmeans_pace = KMeans(n_clusters=3, random_state=42, n_init=10)
df_combined['LearningPace_Cluster'] = kmeans_pace.fit_predict(df_pace_scaled)

print("✅ Fitur 'LearningPace_Cluster' berhasil dibuat menggunakan K-Means!")


"""### Cleaning Dataset NLP VAK (`dataset.csv`)"""
df_vak.columns = df_vak.columns.str.strip()
df_vak = df_vak.rename(columns={'Type': 'LearningStyle'})
df_vak['Sentence'] = df_vak['Sentence'].str.replace('"', '').str.strip()
df_vak['LearningStyle'] = df_vak['LearningStyle'].str.strip().str.capitalize()
df_vak = df_vak[~df_vak['Sentence'].str.contains('Show More', case=False, na=False)]
df_vak.dropna(subset=['Sentence'], inplace=True)
df_vak.drop_duplicates(subset=['Sentence'], inplace=True)
df_vak['Sentence'] = df_vak['Sentence'].str.lower().str.strip()

print(f"✅ Proses cleaning selesai. Data siap: {len(df_combined)} data siswa & {len(df_vak)} data teks kalimat.")


"""### Preprocessing untuk Pemodelan AI Klasifikasi VAK"""
# Encoding Target Gaya Belajar
le = LabelEncoder()
df_combined['LearningStyle_Encoded'] = le.fit_transform(df_combined['LearningStyle'])
mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("\nHandoff info - Label Mapping Gaya Belajar:", mapping)

# Feature Selection untuk Model AI Klasifikasi
selected_features = [
    'AcademicScore',
    'AttendanceRate',
    'Indikator_Visual',
    'Indikator_Auditory',
    'Indikator_Kinestetik'
]

X = df_combined[selected_features]
y = df_combined['LearningStyle_Encoded']

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling Akhir Data Numerik untuk Model Klasifikasi
scaler_final = StandardScaler()
X_train_scaled = pd.DataFrame(scaler_final.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler_final.transform(X_test), columns=X_test.columns)
print("✅ Pemisahan dan scaling data klasifikasi AI selesai.")

"""### Export Data untuk Web Developer & AI Engineer"""
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# 1. Simpan data porsi training & testing (Untuk AI Engineer membuat Model Klasifikasi VAK)
X_train_scaled.to_csv(os.path.join(output_dir, 'X_train_scaled.csv'), index=False)
y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
X_test_scaled.to_csv(os.path.join(output_dir, 'X_test_scaled.csv'), index=False)
y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

# 2. Simpan master dataset hasil penggabungan (LENGKAP dengan label Gaya Belajar dan Cluster Learning Pace untuk Web Developer)
df_combined.to_csv(os.path.join(output_dir, 'final_dataset_cleaned_with_pace.csv'), index=False)

# 3. Simpan dataset kualitatif teks kalimat (Untuk Fitur NLP Rekomendasi Modul Belajar)
df_vak.to_csv(os.path.join(output_dir, 'master_vak_nlp.csv'), index=False)

print("\n🚀 ALL PROCESSES COMPLETED SUCCESSFULLY! SEMUA FILE SIAP DIUNDUH.")
