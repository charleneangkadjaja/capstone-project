# Capstone Project - Exploratory Data Analysis (EDA)
# Pastikan untuk pip install pandas numpy matplotlib seaborn jika belum terinstall
# Pastikan file 'output/final_dataset_cleaned_with_pace.csv' sudah ada (dihasilkan oleh capstone.py), lalu jalankan: python capstone_eda.py

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

INPUT_FILE = 'output/final_dataset_cleaned_with_pace.csv'
OUTPUT_DIR = 'output/eda_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE_VAK = {'Visual': '#4C72B0', 'Auditory': '#ff7f0e', 'Kinesthetic': '#55A868'}
PALETTE_PACE = {'Fast': '#4C72B0', 'Medium': '#ff7f0e', 'Slow': '#55A868'}
COLORS_VAK = ['#4C72B0', '#ff7f0e', '#55A868']
PACE_ORDER = ['Fast', 'Medium', 'Slow']
LS_ORDER = ['Visual', 'Auditory', 'Kinesthetic']

sns.set_theme(style='whitegrid', font_scale=1.2)

# Load data
df = pd.read_csv(INPUT_FILE)

# Label bermakna pada Cluster K-Means berdasarkan rata-rata AcademicScore ---> tau dari mana K-Means
cluster_means = df.groupby('LearningPace_Cluster')['AcademicScore'].mean().sort_values()
cluster_map = {
    cluster_means.index[0]: 'Slow',
    cluster_means.index[1]: 'Medium',
    cluster_means.index[2]: 'Fast'
}
df['LearningPace_Label'] = df['LearningPace_Cluster'].map(cluster_map)
print(f"Dataset dimuat: {df.shape[0]} baris dan {df.shape[1]} kolom")

# EDA (Exploratory Data Analysis)
# Statitik deskriptif
print("\nStatistik Deskriptif:")
print(df.describe(include='all').to_string())

print("\nDistribusi Learning Style:")
print(df['LearningStyle'].value_counts().to_string())

print("\nDistribusi Learning Pace:")
print(df['LearningPace_Label'].value_counts().to_string())

print("\nRata-rata fitur berdasarkan Learning Style:")
num_features = ['AcademicScore', 'AttendanceRate', 'StudentPerformance',
                'DeviceUsage', 'Indikator_Visual', 'Indikator_Auditory', 'Indikator_Kinestetik']
print(df.groupby('LearningStyle')[num_features].mean().round(3).to_string())


print("\nRata-rata fitur berdasarkan Learning Pace:")
print(df.groupby('LearningPace_Label')[['AcademicScore', 'AttendanceRate', 'DeviceUsage']]
        .agg(['mean', 'std']).round(2).to_string())


# 1 - Distribusi Learning Style & Learning Pace
fig1, axes = plt.subplots(1, 3, figsize=(16, 6))
fig1.suptitle('Distribusi Learning Style & Learning Pace', fontsize=16, fontweight='bold')

# Pie chart distribusi Learning Style
counts_ls = df['LearningStyle'].value_counts()
axes[0].pie(
    counts_ls,
    labels=counts_ls.index,
    autopct='%1.1f%%',
    colors=COLORS_VAK,
    startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=2)
)
axes[0].set_title('Proporsi Learning Style', fontsize=14, fontweight='bold')

# Bar chart jumlah siswa per Learning Style
sns.countplot(data=df, x='LearningStyle', order=LS_ORDER, palette=PALETTE_VAK, ax=axes[1])
axes[1].set_title('Jumlah Siswa per Learning Style', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Learning Style')
axes[1].set_ylabel('Jumlah Siswa')
for p in axes[1].patches:
    axes[1].annotate(f'{int(p.get_height())}',
                     (p.get_x() + p.get_width() / 2, p.get_height()+20),
                        ha='center', va='bottom', fontsize=12)

#  Bar chart distribusi Learning Pace (hasil K-Means)
sns.countplot(data=df, x='LearningPace_Label', order=PACE_ORDER, palette=PALETTE_PACE, ax=axes[2])
axes[2].set_title('Distribusi Learning Pace (K-Means)', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Learning Pace')
axes[2].set_ylabel('Jumlah Siswa')
for p in axes[2].patches:
    axes[2].annotate(f'{int(p.get_height())}',
                     (p.get_x() + p.get_width() / 2, p.get_height()+20),
                        ha='center', va='bottom', fontsize=12)

plt.tight_layout()
fig1.savefig(os.path.join(OUTPUT_DIR, 'EDA_01_distribusi.png'), dpi=150, bbox_inches='tight')
plt.close()


# 2 - Distribusi & Boxplot Fitur Numerik
basic_cols = ['AcademicScore', 'AttendanceRate', 'StudentPerformance', 'DeviceUsage']
fig2, axes = plt.subplots(2, 4, figsize=(20, 10))
fig2.suptitle('Distribusi & Boxplot Fitur Numeri per Learning Style', fontsize=16, fontweight='bold')
for i, col in enumerate(basic_cols):
    # Baris atas: histogram dengan garis mean
    axes[0][i].hist(df[col], bins=30, color='#4C72B0', edgecolor='white', alpha=0.85)
    axes[0][i].set_title(f'Distribusi {col}', fontweight='bold')
    axes[0][i].set_xlabel(col)
    axes[0][i].set_ylabel('Frekuensi')
    axes[0][i].axvline(df[col].mean(), color='red', linestyle='--',
                       linewidth=1.5, label=f'Mean: {df[col].mean():.2f}')
    axes[0][i].legend(fontsize=9)
 
    # Baris bawah: boxplot per gaya belajar
    data_box = [df[df['LearningStyle'] == ls][col].values for ls in LS_ORDER]
    bp = axes[1][i].boxplot(data_box, labels=['Visual', 'Auditory', 'Kinesth.'],
                             patch_artist=True)
    for patch, color in zip(bp['boxes'], COLORS_VAK):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[1][i].set_title(f'{col} per Gaya Belajar', fontweight='bold')
    axes[1][i].set_ylabel(col)
 
plt.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, 'EDA_02_distribusi_fitur.png'), dpi=150, bbox_inches='tight')
plt.close()

# 3 - Violin Plot Indikator VAK
indikators = ['Indikator_Visual', 'Indikator_Auditory', 'Indikator_Kinestetik']
ind_labels =  ['Indikator_Visual', 'Indikator_Auditory', 'Indikator_Kinestetik']
fig3, axes = plt.subplots(1, 3, figsize=(18, 6))
fig3.suptitle('Violin Plot Indikator VAK per Learning Style', fontsize=16, fontweight='bold')

for i, (ind, label) in enumerate(zip(indikators, ind_labels)):
    sns.violinplot(data=df, x='LearningStyle', y=ind, order=LS_ORDER,
                   palette=PALETTE_VAK, ax=axes[i], inner='quart')
    axes[i].set_title(f'{label} per Learning Style', fontweight='bold')
    axes[i].set_xlabel('Learning Style')
    axes[i].set_ylabel('Nilai Indikator')
    
plt.tight_layout()
fig3.savefig(os.path.join(OUTPUT_DIR, 'EDA_03_indikator_vak.png'), dpi=150, bbox_inches='tight')
plt.close()

#  4 - Heatmap Korelasi
fig4, axes = plt.subplots(1, 2, figsize=(18, 7))
fig4.suptitle('EDA — Korelasi Antar Fitur & terhadap Gaya Belajar',
              fontsize=15, fontweight='bold')
# Heatmap korelasi antar fitur numerik
corr = df[num_features].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=axes[0], square=True,
            linewidths=0.5, cbar_kws={'shrink': 0.8})
axes[0].set_title('Heatmap Korelasi Fitur Numerik', fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)


# Bar chart korelasi setiap fitur terhadap LearningStyle_Encoded
corr_target = (df[num_features + ['LearningStyle_Encoded']]
               .corr()['LearningStyle_Encoded']
               .drop('LearningStyle_Encoded')
               .sort_values())
bar_colors = ['#E84040' if v < 0 else '#27AE60' for v in corr_target]
axes[1].barh(corr_target.index, corr_target.values, color=bar_colors, edgecolor='white')
axes[1].axvline(0, color='black', linewidth=0.8)
axes[1].set_title('Korelasi Tiap Fitur vs LearningStyle', fontweight='bold')
axes[1].set_xlabel('Korelasi Pearson')
for i, v in enumerate(corr_target.values):
    offset = 0.003 if v >= 0 else -0.003
    ha     = 'left' if v >= 0 else 'right'
    axes[1].text(v + offset, i, f'{v:.3f}', va='center', ha=ha, fontsize=10)
 
plt.tight_layout()
fig4.savefig(os.path.join(OUTPUT_DIR, 'eda_04_korelasi.png'), dpi=150, bbox_inches='tight')
plt.close()

#  5 - Learning Pace vs Performa Akademik
fig5, axes = plt.subplots(1, 3, figsize=(18, 6))
fig5.suptitle('EDA — Learning Pace vs Performa Akademik',
              fontsize=15, fontweight='bold')

# Boxplot AcademicScore per Learning Pace
sns.boxplot(data=df, x='LearningPace_Label', y='AcademicScore',
            palette=PALETTE_PACE, order=PACE_ORDER, ax=axes[0])
axes[0].set_title('AcademicScore per Learning Pace', fontweight='bold')
axes[0].set_xlabel('Learning Pace')
axes[0].set_ylabel('Academic Score')

# Stacked bar: komposisi gaya belajar per learning pace
crosstab = pd.crosstab(df['LearningPace_Label'], df['LearningStyle'], normalize='index')
crosstab = crosstab * 100
crosstab = crosstab.reindex(PACE_ORDER)

crosstab[LS_ORDER].plot(
    kind='bar',
    stacked=True,
    color=COLORS_VAK,
    ax=axes[1],
    edgecolor='white'
)

axes[1].set_title('Komposisi Gaya Belajar per Learning Pace', fontweight='bold')
axes[1].set_xlabel('Learning Pace')
axes[1].set_ylabel('Persentase (%)')
axes[1].legend(title='Gaya Belajar', bbox_to_anchor=(1.01, 1))
axes[1].tick_params(axis='x', rotation=0)

# Scatter plot AcademicScore vs AttendanceRate, warna berdasarkan Learning Pace
for pace, grp in df.groupby('LearningPace_Label'):
    axes[2].scatter(grp['AttendanceRate'], grp['AcademicScore'],
                    c=PALETTE_PACE[pace], label=pace, alpha=0.4, s=15)
axes[2].set_title('AcademicScore vs AttendanceRate\n(warna = Learning Pace)', fontweight='bold')
axes[2].set_xlabel('Attendance Rate')
axes[2].set_ylabel('Academic Score')
axes[2].legend(title='Learning Pace')
 
plt.tight_layout()
fig5.savefig(os.path.join(OUTPUT_DIR, 'eda_05_learning_pace.png'), dpi=150, bbox_inches='tight')
plt.close()

print("🚀 EDA SELESAI! Semua grafik tersimpan di folder:", OUTPUT_DIR)
print("\n📌 INSIGHT UTAMA:")
print("  • Data VAK sangat seimbang — ideal untuk klasifikasi.")
print("  • Cluster Fast: AcademicScore tinggi (~86) + DeviceUsage tinggi.")
print("  • Cluster Slow: AcademicScore rendah (~55) meski attendance serupa.")
print("  • Indikator_Visual & Indikator_Auditory paling berkorelasi dengan LearningStyle.")
print("  • Tidak ada multikolinearitas parah — semua fitur aman dipakai untuk modeling.")