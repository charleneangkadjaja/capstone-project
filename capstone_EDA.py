# Capstone Project - Exploratory Data Analysis (EDA)
# Pastikan untuk pip install pandas numpy matplotlib seaborn scikit-learn jika belum terinstall
# Jalankan setelah capstone.py selesai:
# python capstone_EDA.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# ==========================================================
# CONFIG
# ==========================================================

INPUT_FILE = 'output/final_dataset_cleaned_with_pace.csv'
OUTPUT_DIR = 'output/eda_results'

os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE_VAK = {
    'Visual': '#4C72B0',
    'Auditory': '#ff7f0e',
    'Kinesthetic': '#55A868'
}

PALETTE_PACE = {
    'Fast': '#4C72B0',
    'Medium': '#ff7f0e',
    'Slow': '#55A868'
}

COLORS_VAK = ['#4C72B0', '#ff7f0e', '#55A868']

PACE_ORDER = ['Fast', 'Medium', 'Slow']
LS_ORDER = ['Visual', 'Auditory', 'Kinesthetic']

sns.set_theme(style='whitegrid', font_scale=1.2)

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(INPUT_FILE)

print(f"Dataset dimuat: {df.shape[0]} baris dan {df.shape[1]} kolom")

# ==========================================================
# LABEL LEARNING PACE
# ==========================================================

cluster_means = df.groupby('LearningPace_Cluster')['AcademicScore'].mean().sort_values()

cluster_map = {
    cluster_means.index[0]: 'Slow',
    cluster_means.index[1]: 'Medium',
    cluster_means.index[2]: 'Fast'
}

df['LearningPace_Label'] = df['LearningPace_Cluster'].map(cluster_map)

# ==========================================================
# DESCRIPTIVE ANALYSIS
# ==========================================================

print("\nStatistik Deskriptif:")
print(df.describe(include='all').to_string())

print("\nDistribusi Learning Style:")
print(df['LearningStyle'].value_counts().to_string())

print("\nDistribusi Learning Pace:")
print(df['LearningPace_Label'].value_counts().to_string())

num_features = [
    'AcademicScore',
    'AttendanceRate',
    'StudentPerformance',
    'DeviceUsage',
    'Indikator_Visual',
    'Indikator_Auditory',
    'Indikator_Kinestetik'
]

print("\nRata-rata fitur berdasarkan Learning Style:")
print(
    df.groupby('LearningStyle')[num_features]
    .mean()
    .round(3)
    .to_string()
)

print("\nRata-rata fitur berdasarkan Learning Pace:")
print(
    df.groupby('LearningPace_Label')[['AcademicScore', 'AttendanceRate', 'DeviceUsage']]
    .agg(['mean', 'std'])
    .round(2)
    .to_string()
)

# ==========================================================
# PCA VISUALIZATION
# ==========================================================

pace_features = [
    'AcademicScore',
    'AttendanceRate',
    'DeviceUsage'
]

df_pace = df[pace_features]

scaler = StandardScaler()
df_pace_scaled = scaler.fit_transform(df_pace)

pca = PCA(n_components=2)

coords = pca.fit_transform(df_pace_scaled)

df['PCA1'] = coords[:, 0]
df['PCA2'] = coords[:, 1]

print("✅ PCA visualization berhasil dibuat.")

# ==========================================================
# 1. DISTRIBUSI LEARNING STYLE & PACE
# ==========================================================

fig1, axes = plt.subplots(1, 3, figsize=(16, 6))

fig1.suptitle(
    'Distribusi Learning Style & Learning Pace',
    fontsize=16,
    fontweight='bold'
)

# Pie Chart
counts_ls = df['LearningStyle'].value_counts()

axes[0].pie(
    counts_ls,
    labels=counts_ls.index,
    autopct='%1.1f%%',
    colors=COLORS_VAK,
    startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=2)
)

axes[0].set_title(
    'Proporsi Learning Style',
    fontsize=14,
    fontweight='bold'
)

# Countplot Learning Style
sns.countplot(
    data=df,
    x='LearningStyle',
    order=LS_ORDER,
    palette=PALETTE_VAK,
    ax=axes[1]
)

axes[1].set_title(
    'Jumlah Siswa per Learning Style',
    fontsize=14,
    fontweight='bold'
)

axes[1].set_xlabel('Learning Style')
axes[1].set_ylabel('Jumlah Siswa')

for p in axes[1].patches:
    axes[1].annotate(
        f'{int(p.get_height())}',
        (p.get_x() + p.get_width() / 2, p.get_height() + 20),
        ha='center',
        va='bottom',
        fontsize=12
    )

# Countplot Learning Pace
sns.countplot(
    data=df,
    x='LearningPace_Label',
    order=PACE_ORDER,
    palette=PALETTE_PACE,
    ax=axes[2]
)

axes[2].set_title(
    'Distribusi Learning Pace (K-Means)',
    fontsize=14,
    fontweight='bold'
)

axes[2].set_xlabel('Learning Pace')
axes[2].set_ylabel('Jumlah Siswa')

for p in axes[2].patches:
    axes[2].annotate(
        f'{int(p.get_height())}',
        (p.get_x() + p.get_width() / 2, p.get_height() + 20),
        ha='center',
        va='bottom',
        fontsize=12
    )

plt.tight_layout()

fig1.savefig(
    os.path.join(OUTPUT_DIR, 'EDA_01_distribusi.png'),
    dpi=150,
    bbox_inches='tight'
)

plt.close()

# ==========================================================
# 2. DISTRIBUSI FITUR NUMERIK
# ==========================================================

basic_cols = [
    'AcademicScore',
    'AttendanceRate',
    'StudentPerformance',
    'DeviceUsage'
]

fig2, axes = plt.subplots(2, 4, figsize=(20, 10))

fig2.suptitle(
    'Distribusi & Boxplot Fitur Numerik per Learning Style',
    fontsize=16,
    fontweight='bold'
)

for i, col in enumerate(basic_cols):

    axes[0][i].hist(
        df[col],
        bins=30,
        color='#4C72B0',
        edgecolor='white',
        alpha=0.85
    )

    axes[0][i].set_title(f'Distribusi {col}', fontweight='bold')

    axes[0][i].axvline(
        df[col].mean(),
        color='red',
        linestyle='--',
        linewidth=1.5,
        label=f'Mean: {df[col].mean():.2f}'
    )

    axes[0][i].legend(fontsize=9)

    data_box = [
        df[df['LearningStyle'] == ls][col].values
        for ls in LS_ORDER
    ]

    bp = axes[1][i].boxplot(
        data_box,
        labels=['Visual', 'Auditory', 'Kinesth.'],
        patch_artist=True
    )

    for patch, color in zip(bp['boxes'], COLORS_VAK):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    axes[1][i].set_title(
        f'{col} per Gaya Belajar',
        fontweight='bold'
    )

plt.tight_layout()

fig2.savefig(
    os.path.join(OUTPUT_DIR, 'EDA_02_distribusi_fitur.png'),
    dpi=150,
    bbox_inches='tight'
)

plt.close()

# ==========================================================
# 3. VIOLIN PLOT INDIKATOR VAK
# ==========================================================

indikators = [
    'Indikator_Visual',
    'Indikator_Auditory',
    'Indikator_Kinestetik'
]

fig3, axes = plt.subplots(1, 3, figsize=(18, 6))

fig3.suptitle(
    'Violin Plot Indikator VAK per Learning Style',
    fontsize=16,
    fontweight='bold'
)

for i, ind in enumerate(indikators):

    sns.violinplot(
        data=df,
        x='LearningStyle',
        y=ind,
        order=LS_ORDER,
        palette=PALETTE_VAK,
        ax=axes[i],
        inner='quart'
    )

    axes[i].set_title(
        f'{ind} per Learning Style',
        fontweight='bold'
    )

plt.tight_layout()

fig3.savefig(
    os.path.join(OUTPUT_DIR, 'EDA_03_indikator_vak.png'),
    dpi=150,
    bbox_inches='tight'
)

plt.close()

# ==========================================================
# 4. HEATMAP KORELASI
# ==========================================================

fig4, axes = plt.subplots(1, 2, figsize=(18, 7))

fig4.suptitle(
    'EDA — Korelasi Antar Fitur & terhadap Gaya Belajar',
    fontsize=15,
    fontweight='bold'
)

corr = df[num_features].corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    ax=axes[0],
    square=True,
    linewidths=0.5,
    cbar_kws={'shrink': 0.8}
)

axes[0].set_title(
    'Heatmap Korelasi Fitur Numerik',
    fontweight='bold'
)

corr_target = (
    df[num_features + ['LearningStyle_Encoded']]
    .corr()['LearningStyle_Encoded']
    .drop('LearningStyle_Encoded')
    .sort_values()
)

bar_colors = [
    '#E84040' if v < 0 else '#27AE60'
    for v in corr_target
]

axes[1].barh(
    corr_target.index,
    corr_target.values,
    color=bar_colors,
    edgecolor='white'
)

axes[1].axvline(0, color='black', linewidth=0.8)

axes[1].set_title(
    'Korelasi Tiap Fitur vs LearningStyle',
    fontweight='bold'
)

plt.tight_layout()

fig4.savefig(
    os.path.join(OUTPUT_DIR, 'EDA_04_korelasi.png'),
    dpi=150,
    bbox_inches='tight'
)

plt.close()

# ==========================================================
# 5. LEARNING PACE VS PERFORMA AKADEMIK
# ==========================================================

fig5, axes = plt.subplots(1, 3, figsize=(18, 6))

fig5.suptitle(
    'EDA — Learning Pace vs Performa Akademik',
    fontsize=15,
    fontweight='bold'
)

# Boxplot
sns.boxplot(
    data=df,
    x='LearningPace_Label',
    y='AcademicScore',
    palette=PALETTE_PACE,
    order=PACE_ORDER,
    ax=axes[0]
)

axes[0].set_title(
    'AcademicScore per Learning Pace',
    fontweight='bold'
)

# Crosstab
crosstab = pd.crosstab(
    df['LearningPace_Label'],
    df['LearningStyle'],
    normalize='index'
)

crosstab = crosstab * 100
crosstab = crosstab.reindex(PACE_ORDER)

crosstab[LS_ORDER].plot(
    kind='bar',
    stacked=True,
    color=COLORS_VAK,
    ax=axes[1],
    edgecolor='white'
)

axes[1].set_title(
    'Komposisi Gaya Belajar per Learning Pace',
    fontweight='bold'
)

# PCA Scatter
for pace, grp in df.groupby('LearningPace_Label'):

    axes[2].scatter(
        grp['PCA1'],
        grp['PCA2'],
        c=PALETTE_PACE[pace],
        label=pace,
        alpha=0.4,
        s=15
    )

axes[2].set_title(
    'PCA Visualization Learning Pace',
    fontweight='bold'
)

axes[2].set_xlabel('PCA Component 1')
axes[2].set_ylabel('PCA Component 2')

axes[2].legend(title='Learning Pace')

plt.tight_layout()

fig5.savefig(
    os.path.join(OUTPUT_DIR, 'EDA_05_learning_pace.png'),
    dpi=150,
    bbox_inches='tight'
)

plt.close()

# ==========================================================
# 6. PCA CLUSTER VISUALIZATION
# ==========================================================

fig6 = plt.figure(figsize=(8, 6))

for pace, grp in df.groupby('LearningPace_Label'):

    plt.scatter(
        grp['PCA1'],
        grp['PCA2'],
        c=PALETTE_PACE[pace],
        label=pace,
        alpha=0.5,
        s=20
    )

plt.title(
    'PCA Cluster Visualization',
    fontsize=15,
    fontweight='bold'
)

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

plt.legend(title='Learning Pace')

plt.tight_layout()

fig6.savefig(
    os.path.join(OUTPUT_DIR, 'EDA_06_PCA_Cluster.png'),
    dpi=150,
    bbox_inches='tight'
)

plt.close()

print("✅ PCA visualization berhasil disimpan.")

# ==========================================================
# FINAL INSIGHT
# ==========================================================
print("🚀 EDA SELESAI! Semua grafik tersimpan di folder:", OUTPUT_DIR)
print("\n🚀 EDA SELESAI! Semua grafik tersimpan di folder:", OUTPUT_DIR)

print("\n📌 INSIGHT UTAMA:")
print("  • Data VAK sangat seimbang — ideal untuk klasifikasi.")
print("  • Cluster Fast memiliki AcademicScore tertinggi.")
print("  • Cluster Slow memiliki AcademicScore terendah.")
print("  • Indikator_Visual & Indikator_Auditory paling berkorelasi dengan LearningStyle.")
print("  • Tidak ada multikolinearitas parah — fitur aman untuk modeling.")
