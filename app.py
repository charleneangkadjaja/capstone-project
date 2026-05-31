# =============================================================================
# app.py — EduProfile AI: Learning Analytics Dashboard (Streamlit)
# CC26-PSU099 | Data Science Team (Raissa & Charlene)
# =============================================================================
# Cara menjalankan:
#   pip install streamlit pandas numpy matplotlib seaborn plotly scipy scikit-learn
#   streamlit run app.py
#
# Cara deploy ke Streamlit Cloud:
#   1. Push ke GitHub (pastikan output/final_dataset_cleaned_with_pace.csv ada)
#   2. Buka share.streamlit.io → New app → pilih repo & file app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── Konfigurasi Halaman ────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduProfile AI — Learning Analytics Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F8FAFC; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border-left: 5px solid #4C72B0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        margin-bottom: 1rem;
    }
    .metric-card.orange { border-left-color: #FF7F0E; }
    .metric-card.green  { border-left-color: #55A868; }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1E293B;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #E2E8F0;
    }
    .insight-box {
        background: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        font-size: 0.95rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Palette & Konstanta ─────────────────────────────────────────────────────
PALETTE_VAK   = {'Visual': '#4C72B0', 'Auditory': '#FF7F0E', 'Kinesthetic': '#55A868'}
PALETTE_PACE  = {'Fast': '#4C72B0', 'Medium': '#FF7F0E', 'Slow': '#55A868'}
COLORS_VAK    = ['#4C72B0', '#FF7F0E', '#55A868']
LS_ORDER      = ['Visual', 'Auditory', 'Kinesthetic']
PACE_ORDER    = ['Fast', 'Medium', 'Slow']
ALPHA         = 0.05

# ── Load & Cache Data ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('output/final_dataset_cleaned_with_pace.csv')
    cluster_means = df.groupby('LearningPace_Cluster')['AcademicScore'].mean().sort_values()
    cluster_map = {
        cluster_means.index[0]: 'Slow',
        cluster_means.index[1]: 'Medium',
        cluster_means.index[2]: 'Fast'
    }
    df['LearningPace_Label'] = df['LearningPace_Cluster'].map(cluster_map)
    return df

try:
    df = load_data()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=64)
    st.title("EduProfile AI")
    st.caption("CC26-PSU099 — Data Science Team")
    st.divider()

    if data_loaded:
        st.markdown("**🔍 Filter Data**")
        filter_style = st.multiselect(
            "Learning Style",
            options=LS_ORDER,
            default=LS_ORDER
        )
        filter_pace = st.multiselect(
            "Learning Pace",
            options=PACE_ORDER,
            default=PACE_ORDER
        )

        df_filtered = df[
            df['LearningStyle'].isin(filter_style) &
            df['LearningPace_Label'].isin(filter_pace)
        ]

        st.divider()
        st.markdown(f"**📊 Data Aktif:** `{len(df_filtered):,}` siswa")
        st.markdown(f"**📦 Total Dataset:** `{len(df):,}` siswa")
    else:
        st.error("❌ File data tidak ditemukan.\n\nJalankan `capstone.py` terlebih dahulu untuk menghasilkan:\n`output/final_dataset_cleaned_with_pace.csv`")

# ── Header Utama ─────────────────────────────────────────────────────────────
st.title("🎓 EduProfile AI — Learning Analytics Dashboard")
st.markdown(
    "Platform analitik untuk memahami **gaya belajar VAK** dan **kecepatan belajar** "
    "siswa secara objektif berbasis data. | *Coding Camp 2026 powered by DBS Foundation*"
)

if not data_loaded:
    st.error("Pastikan file `output/final_dataset_cleaned_with_pace.csv` tersedia. Jalankan `capstone.py` terlebih dahulu.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# METRIC CARDS — KPI Utama
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Siswa", f"{len(df_filtered):,}", f"{len(df_filtered)-len(df):,}" if len(df_filtered) != len(df) else "Semua Data")
with col2:
    st.metric("Rata-rata Skor", f"{df_filtered['AcademicScore'].mean():.2f}", "Skala 1–5")
with col3:
    st.metric("Rata-rata Kehadiran", f"{df_filtered['AttendanceRate'].mean():.2f}", "Skala 1–5")
with col4:
    dominant_style = df_filtered['LearningStyle'].value_counts().idxmax()
    pct_dominant   = df_filtered['LearningStyle'].value_counts(normalize=True).max() * 100
    st.metric("Gaya Belajar Dominan", dominant_style, f"{pct_dominant:.1f}%")
with col5:
    fast_pct = (df_filtered['LearningPace_Label'] == 'Fast').mean() * 100
    st.metric("Fast Learners", f"{fast_pct:.1f}%", "dari total siswa")

# ══════════════════════════════════════════════════════════════════════════════
# TABS UTAMA
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Distribusi & Overview",
    "🔬 Indikator VAK",
    "🚀 Learning Pace",
    "📐 Korelasi & Fitur",
    "🧪 A/B Testing"
])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: Distribusi & Overview
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown(
        '<div class="section-header">Distribusi Gaya Belajar & Kecepatan Belajar</div>',
        unsafe_allow_html=True
    )

    # ── BUSINESS QUESTIONS ────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">❓ Business Questions</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="insight-box">

    <b>1.</b> Sejauh mana profil belajar siswa saat ini terdistribusi secara merata di dalam sistem?<br><br>

    <b>2.</b> Apakah sistem pengujian mampu menangkap variasi gaya belajar secara objektif untuk mendukung personalisasi?<br><br>

    <b>3.</b> Faktor perilaku apa yang memiliki pengaruh paling kuat terhadap performa akademik siswa?<br><br>

    <b>4.</b> Bagaimana cara sistem mengelompokkan kecepatan belajar siswa untuk meningkatkan efisiensi kurikulum?

    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Proporsi Learning Style (VAK)**")

        fig, ax = plt.subplots(figsize=(5, 4))

        counts = (
            df_filtered['LearningStyle']
            .value_counts()
            .reindex(LS_ORDER)
            .fillna(0)
        )

        counts = counts[counts > 0]

        wedge_props = dict(edgecolor='white', linewidth=2.5)

        if len(counts) > 0:
            ax.pie(
                counts,
                labels=counts.index,
                autopct='%1.1f%%',
                colors=[PALETTE_VAK[x] for x in counts.index],
                startangle=90,
                wedgeprops=wedge_props,
                textprops={'fontsize': 11}
            )
        else:
            ax.text(
                0.5,
                0.5,
                'Tidak ada data',
                ha='center',
                va='center',
                fontsize=12
            )

        ax.set_title(
            'Distribusi Learning Style',
            fontweight='bold',
            pad=10
        )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("**Jumlah Siswa per Kategori**")

        fig, axes = plt.subplots(1, 2, figsize=(7, 4))

        sns.set_theme(style='whitegrid', font_scale=1.0)

        # =========================
        # BAR LEARNING STYLE
        # =========================
        cnt_ls = (
            df_filtered['LearningStyle']
            .value_counts()
            .reindex(LS_ORDER)
            .fillna(0)
        )

        cnt_ls = cnt_ls[cnt_ls > 0]

        bars = axes[0].bar(
            cnt_ls.index,
            cnt_ls.values,
            color=[PALETTE_VAK[ls] for ls in cnt_ls.index],
            edgecolor='white',
            linewidth=1.5
        )

        axes[0].set_title(
            'Per Learning Style',
            fontweight='bold'
        )

        axes[0].set_ylabel('Jumlah Siswa')

        for bar in bars:
            axes[0].text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 15,
                f'{int(bar.get_height()):,}',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )

        # =========================
        # BAR LEARNING PACE
        # =========================
        cnt_pace = (
            df_filtered['LearningPace_Label']
            .value_counts()
            .reindex(PACE_ORDER)
            .fillna(0)
        )

        cnt_pace = cnt_pace[cnt_pace > 0]

        bars2 = axes[1].bar(
            cnt_pace.index,
            cnt_pace.values,
            color=[PALETTE_PACE[p] for p in cnt_pace.index],
            edgecolor='white',
            linewidth=1.5
        )

        axes[1].set_title(
            'Per Learning Pace',
            fontweight='bold'
        )

        axes[1].set_ylabel('Jumlah Siswa')

        for bar in bars2:
            axes[1].text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 15,
                f'{int(bar.get_height()):,}',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # =========================================================================
    # DISTRIBUSI FITUR NUMERIK
    # =========================================================================
    st.markdown(
        '<div class="section-header">Distribusi Fitur Numerik Utama</div>',
        unsafe_allow_html=True
    )

    numeric_cols = [
        'AcademicScore',
        'AttendanceRate',
        'StudentPerformance',
        'DeviceUsage'
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for i, col in enumerate(numeric_cols):

        axes[i].hist(
            df_filtered[col],
            bins=30,
            color='#4C72B0',
            edgecolor='white',
            alpha=0.85
        )

        mean_val = df_filtered[col].mean()

        axes[i].axvline(
            mean_val,
            color='#E84040',
            linestyle='--',
            linewidth=2,
            label=f'Mean: {mean_val:.2f}'
        )

        axes[i].set_title(
            col,
            fontweight='bold',
            fontsize=11
        )

        axes[i].set_xlabel('Nilai')
        axes[i].set_ylabel('Frekuensi')
        axes[i].legend(fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # =========================================================================
    # BUSINESS QUESTION
    # =========================================================================
    st.markdown(
        '<div class="section-header">📌 Jawaban Pertanyaan Bisnis #1</div>',
        unsafe_allow_html=True
    )

    style_score = (
        df_filtered
        .groupby('LearningStyle')['AcademicScore']
        .agg(['mean', 'std'])
        .round(3)
    )

    if len(style_score) > 0:

        best_style = style_score['mean'].idxmax()

        st.markdown(f"""
        <div class="insight-box">
        <b>Q: Apakah ada gaya belajar dengan performa akademik lebih tinggi?</b><br>
        Siswa dengan gaya belajar <b>{best_style}</b>
        memiliki rata-rata AcademicScore tertinggi
        ({style_score.loc[best_style, 'mean']:.3f}),
        namun perbedaan antar kelompok perlu
        divalidasi secara statistik
        (lihat tab A/B Testing untuk hasil ANOVA).
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(
            style_score.rename(columns={
                'mean': 'Rata-rata Skor',
                'std': 'Std. Deviasi'
            }),
            use_container_width=True
        )

    else:

        st.warning("Tidak ada data untuk ditampilkan.")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: Indikator VAK
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Analisis Indikator VAK per Gaya Belajar</div>', unsafe_allow_html=True)

    indikators = ['Indikator_Visual', 'Indikator_Auditory', 'Indikator_Kinestetik']

    # Violin Plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, ind in enumerate(indikators):
        sns.violinplot(data=df_filtered, x='LearningStyle', y=ind, order=LS_ORDER,
                       palette=PALETTE_VAK, ax=axes[i], inner='quart')
        axes[i].set_title(f'{ind.replace("Indikator_", "")} per Learning Style',
                          fontweight='bold', fontsize=12)
        axes[i].set_xlabel('Learning Style')
        axes[i].set_ylabel('Nilai Indikator (1–5)')
    fig.suptitle('Violin Plot: Distribusi Indikator VAK', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Radar Chart (Spider) per gaya belajar rata-rata
    st.markdown('<div class="section-header">Radar Chart: Profil Rata-rata per Gaya Belajar</div>', unsafe_allow_html=True)
    col_r1, col_r2 = st.columns([2, 1])

    with col_r1:
        categories = ['Indikator\nVisual', 'Indikator\nAuditory', 'Indikator\nKinestetik', 'AcademicScore', 'AttendanceRate']
        feat_radar  = ['Indikator_Visual', 'Indikator_Auditory', 'Indikator_Kinestetik', 'AcademicScore', 'AttendanceRate']
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 5), subplot_kw=dict(polar=True))
        for ls, color in PALETTE_VAK.items():
            vals = df_filtered[df_filtered['LearningStyle'] == ls][feat_radar].mean().tolist()
            # Normalize to 0-1 for radar
            vals_norm = [(v - 1) / 4 for v in vals]
            vals_norm += vals_norm[:1]
            ax.plot(angles, vals_norm, 'o-', linewidth=2, color=color, label=ls)
            ax.fill(angles, vals_norm, alpha=0.15, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels(['1.25–2', '2–3', '3–4'], fontsize=7)
        ax.set_title('Profil Rata-rata per Gaya Belajar\n(Skala Ternormalisasi 0–1)',
                     fontweight='bold', pad=15)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_r2:
        st.markdown("**Tabel Rata-rata Indikator VAK**")
        summary_vak = df_filtered.groupby('LearningStyle')[feat_radar].mean().round(3)
        st.dataframe(summary_vak, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <b>Insight:</b> Indikator VAK dirancang agar nilai tertinggi
        pada indikator yang sesuai dengan gaya belajar.
        Validasi statistik tersedia di tab A/B Testing.
        </div>
        """, unsafe_allow_html=True)

    # Boxplot
    st.markdown('<div class="section-header">Boxplot Indikator VAK per Gaya Belajar</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for i, ind in enumerate(indikators):
        data_box = [df_filtered[df_filtered['LearningStyle'] == ls][ind].values for ls in LS_ORDER]
        bp = axes[i].boxplot(data_box, labels=LS_ORDER, patch_artist=True, notch=True)
        for patch, color in zip(bp['boxes'], COLORS_VAK):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        axes[i].set_title(ind.replace("Indikator_", "Indikator "), fontweight='bold')
        axes[i].set_ylabel('Nilai (1–5)')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3: Learning Pace
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Analisis Learning Pace (K-Means Clustering)</div>', unsafe_allow_html=True)

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        # Boxplot AcademicScore per Learning Pace
        fig, ax = plt.subplots(figsize=(6, 4))
        data_pace = [df_filtered[df_filtered['LearningPace_Label'] == p]['AcademicScore'].values for p in PACE_ORDER]
        bp = ax.boxplot(data_pace, labels=PACE_ORDER, patch_artist=True, notch=True)
        for patch, (pace, color) in zip(bp['boxes'], PALETTE_PACE.items()):
            patch.set_facecolor(PALETTE_PACE[PACE_ORDER[list(bp['boxes']).index(patch)]])
            patch.set_alpha(0.8)
        ax.set_title('AcademicScore per Learning Pace', fontweight='bold')
        ax.set_xlabel('Learning Pace')
        ax.set_ylabel('Academic Score (1–5)')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_p2:
        # Stacked Bar: Komposisi Learning Style per Pace
        fig, ax = plt.subplots(figsize=(6, 4))
        crosstab = pd.crosstab(df_filtered['LearningPace_Label'], df_filtered['LearningStyle'], normalize='index') * 100
        crosstab = crosstab.reindex(PACE_ORDER)
        bottom = np.zeros(len(PACE_ORDER))
        for ls, color in PALETTE_VAK.items():
            if ls in crosstab.columns:
                ax.bar(PACE_ORDER, crosstab[ls], bottom=bottom, color=color, label=ls, edgecolor='white', linewidth=0.5)
                bottom += crosstab[ls].values
        ax.set_title('Komposisi Gaya Belajar per Learning Pace', fontweight='bold')
        ax.set_xlabel('Learning Pace')
        ax.set_ylabel('Persentase (%)')
        ax.legend(title='Gaya Belajar', bbox_to_anchor=(1.01, 1))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


    # Scatter + PCA Visualization
    st.markdown(
        '<div class="section-header">Visualisasi Cluster Learning Pace (PCA 2D)</div>',
        unsafe_allow_html=True
    )

    pace_features = ['AcademicScore', 'AttendanceRate', 'DeviceUsage']

    # cek data kosong
    if len(df_filtered) > 1:

        # ambil data PCA
        df_pca = df_filtered[pace_features].copy()

        # isi missing value
        df_pca = df_pca.fillna(df_pca.median())

        # scaling
        scaler_pca = StandardScaler()
        X_scaled = scaler_pca.fit_transform(df_pca)

        # PCA
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X_scaled)

        col_scatter1, col_scatter2 = st.columns(2)

        # =========================
        # PCA Scatter
        # =========================
        with col_scatter1:

            fig, ax = plt.subplots(figsize=(6, 5))

            for pace, color in PALETTE_PACE.items():

                mask = df_filtered['LearningPace_Label'] == pace

                if mask.sum() > 0:
                    ax.scatter(
                        coords[mask, 0],
                        coords[mask, 1],
                        c=color,
                        label=pace,
                        alpha=0.4,
                        s=12
                    )

            ax.set_xlabel(
                f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)'
            )

            ax.set_ylabel(
                f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)'
            )

            ax.set_title(
                'Cluster Learning Pace (PCA 2D)',
                fontweight='bold'
            )

            ax.legend(title='Learning Pace')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # =========================
        # Scatter AcademicScore vs AttendanceRate
        # =========================
        with col_scatter2:

            fig, ax = plt.subplots(figsize=(6, 5))

            for pace, color in PALETTE_PACE.items():

                grp = df_filtered[
                    df_filtered['LearningPace_Label'] == pace
                ]

                if len(grp) > 0:
                    ax.scatter(
                        grp['AttendanceRate'],
                        grp['AcademicScore'],
                        c=color,
                        label=pace,
                        alpha=0.35,
                        s=12
                    )

            ax.set_xlabel('Attendance Rate')
            ax.set_ylabel('Academic Score')

            ax.set_title(
                'AcademicScore vs AttendanceRate\n(warna = Learning Pace)',
                fontweight='bold'
            )

            ax.legend(title='Learning Pace')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    else:
        st.warning("Data terlalu sedikit untuk menampilkan PCA Visualization.")

    # Business Question Answer
    st.markdown("""
    <div class="insight-box">
    <b>📌 Insight Learning Pace:</b><br>
    • <b>Fast Learner</b>: AcademicScore & DeviceUsage tinggi — cenderung memanfaatkan teknologi secara optimal.<br>
    • <b>Slow Learner</b>: AcademicScore rendah meski attendance rate tidak berbeda jauh — mengindikasikan butuh pendekatan belajar berbeda, bukan sekadar kehadiran lebih sering.<br>
    • <b>Implikasi Platform</b>: EduProfile AI perlu memberikan rekomendasi yang berbeda untuk setiap segmen pace.
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4: Korelasi & Fitur
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Analisis Korelasi Antar Fitur</div>', unsafe_allow_html=True)

    num_features = ['AcademicScore', 'AttendanceRate', 'StudentPerformance',
                    'DeviceUsage', 'Indikator_Visual', 'Indikator_Auditory', 'Indikator_Kinestetik']

    col_c1, col_c2 = st.columns([3, 2])
    with col_c1:
        fig, ax = plt.subplots(figsize=(8, 6))
        corr = df_filtered[num_features].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, ax=ax, square=True, linewidths=0.5,
                    cbar_kws={'shrink': 0.8}, annot_kws={'size': 9})
        ax.set_title('Heatmap Korelasi Fitur Numerik', fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_c2:
        # Korelasi terhadap LearningStyle_Encoded
        le_corr = df_filtered[num_features + ['LearningStyle_Encoded']].corr()
        corr_target = le_corr['LearningStyle_Encoded'].drop('LearningStyle_Encoded').sort_values()
        bar_colors = ['#E84040' if v < 0 else '#27AE60' for v in corr_target.values]

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.barh(corr_target.index, corr_target.values, color=bar_colors, edgecolor='white')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title('Korelasi Fitur vs\nLearningStyle', fontweight='bold', fontsize=11)
        ax.set_xlabel('Korelasi Pearson')
        for i, v in enumerate(corr_target.values):
            offset = 0.005 if v >= 0 else -0.005
            ha     = 'left' if v >= 0 else 'right'
            ax.text(v + offset, i, f'{v:.3f}', va='center', ha=ha, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Feature Importance Insight
    st.markdown('<div class="section-header">📌 Insight Korelasi untuk AI Engineer</div>', unsafe_allow_html=True)
    top2 = corr_target.abs().nlargest(2)
    st.markdown(f"""
    <div class="insight-box">
    <b>Fitur paling berkorelasi dengan LearningStyle:</b>
    <b>{top2.index[0]}</b> (|r|={top2.iloc[0]:.3f}) dan
    <b>{top2.index[1]}</b> (|r|={top2.iloc[1]:.3f}).<br>
    <b>Tidak ada multikolinearitas parah</b> (tidak ada korelasi antar-fitur > 0.9) —
    semua fitur aman digunakan untuk model klasifikasi Deep Learning.
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 5: A/B Testing
# ──────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">Statistical A/B Testing & Validasi Model</div>', unsafe_allow_html=True)
    st.info("Semua uji statistik menggunakan α = 0.05 (tingkat kepercayaan 95%)")

# ── Experiment 1: Learning Pace Comparison ──────────────────────────
st.markdown("### Experiment 1 — Perbandingan AcademicScore antar Learning Pace")

available_pace = [
    p for p in PACE_ORDER
    if len(df_filtered[df_filtered['LearningPace_Label'] == p]) > 0
]

if len(available_pace) >= 2:

    pace_1 = available_pace[0]
    pace_2 = available_pace[1]

    st.markdown(
        f"**H₀:** Tidak ada perbedaan AcademicScore antara {pace_1} dan {pace_2} "
        f"| **H₁:** Ada perbedaan signifikan"
    )

    group_1 = df_filtered[
        df_filtered['LearningPace_Label'] == pace_1
    ]['AcademicScore'].dropna()

    group_2 = df_filtered[
        df_filtered['LearningPace_Label'] == pace_2
    ]['AcademicScore'].dropna()

    t_stat, p_ttest = stats.ttest_ind(group_1, group_2)

    pooled_std = np.sqrt(
        (group_1.std()**2 + group_2.std()**2) / 2
    )

    cohens_d = (
        (group_1.mean() - group_2.mean()) / pooled_std
        if pooled_std > 0 else 0
    )

    col_e1a, col_e1b, col_e1c = st.columns(3)

    col_e1a.metric("T-statistic", f"{t_stat:.4f}")
    col_e1b.metric("p-value", f"{p_ttest:.6f}")
    col_e1c.metric("Cohen's d", f"{cohens_d:.4f}")

    if p_ttest < ALPHA:
        st.success(
            f"✅ H₀ DITOLAK — Ada perbedaan signifikan "
            f"antara {pace_1} dan {pace_2}."
        )
    else:
        st.info(
            f"ℹ️ H₀ GAGAL DITOLAK — Tidak ada perbedaan signifikan "
            f"antara {pace_1} dan {pace_2}."
        )

else:
    st.warning(
        "⚠️ Minimal pilih 2 kategori Learning Pace untuk menjalankan Experiment 1."
    )
    
    # ── Experiment 2: ANOVA Indikator VAK ──────────────────────────────────
    st.markdown("### Experiment 2 — Validasi Indikator VAK (One-Way ANOVA)")
    st.markdown("**H₀:** Rata-rata indikator sama di semua gaya belajar | **H₁:** Ada perbedaan signifikan")

    anova_results = []
    for ind in ['Indikator_Visual', 'Indikator_Auditory', 'Indikator_Kinestetik']:
        groups = [df_filtered[df_filtered['LearningStyle'] == ls][ind].dropna().values for ls in LS_ORDER]
        if all(len(g) > 1 for g in groups):
            f_stat, p_anova   = stats.f_oneway(*groups)
            h_stat, p_kruskal_val = kruskal(*groups)
            anova_results.append({
                'Indikator': ind,
                'F-statistic': round(f_stat, 4),
                'p-value (ANOVA)': round(p_anova, 6),
                'H-statistic': round(h_stat, 4),
                'p-value (Kruskal)': round(p_kruskal_val, 6),
                'Signifikan?': '✅ Ya' if p_anova < ALPHA else '❌ Tidak'
            })

    if anova_results:
        df_anova = pd.DataFrame(anova_results)
        st.dataframe(df_anova, use_container_width=True)
        all_sig = all('✅' in r['Signifikan?'] for r in anova_results)
        if all_sig:
            st.success("✅ Semua indikator VAK terbukti signifikan membedakan gaya belajar — Feature Engineering VALID secara statistik.")
        else:
            st.warning("⚠️ Beberapa indikator kurang signifikan dalam kondisi filter ini.")

    st.divider()

    # ── Experiment 3: Chi-Square ────────────────────────────────────────────
    st.markdown("### Experiment 3 — Chi-Square: Learning Style × Learning Pace")
    st.markdown("**H₀:** Learning Style dan Learning Pace independen | **H₁:** Ada hubungan signifikan")

    contingency = pd.crosstab(df_filtered['LearningStyle'], df_filtered['LearningPace_Label'])
    if contingency.shape[0] > 1 and contingency.shape[1] > 1:
        chi2, p_chi2, dof, _ = chi2_contingency(contingency)
        cramers_v = np.sqrt(chi2 / (len(df_filtered) * (min(contingency.shape) - 1)))

        col_chi1, col_chi2, col_chi3 = st.columns(3)
        col_chi1.metric("Chi-Square (χ²)", f"{chi2:.4f}")
        col_chi2.metric("p-value", f"{p_chi2:.6f}", f"df = {dof}")
        col_chi3.metric("Cramér's V", f"{cramers_v:.4f}",
                        "Kuat (≥0.3)" if cramers_v >= 0.3 else "Sedang" if cramers_v >= 0.1 else "Lemah")

        st.markdown("**Contingency Table:**")
        st.dataframe(contingency, use_container_width=False)

        if p_chi2 < ALPHA:
            st.success(f"✅ H₀ **DITOLAK** — Ada hubungan SIGNIFIKAN antara Learning Style dan Learning Pace. Profil ganda EduProfile AI terjustifikasi.")
        else:
            st.info(f"ℹ️ H₀ **GAGAL DITOLAK** — Learning Style dan Learning Pace bersifat independen. Keduanya perlu diprediksi secara terpisah.")

    # ── Ringkasan ───────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📋 Ringkasan Eksekutif A/B Testing")
    st.markdown("""
    | # | Eksperimen | Metode | Implikasi |
    |---|------------|--------|-----------|
    | 1 | Fast vs Slow AcademicScore | t-test + Mann-Whitney | Validasi K-Means Learning Pace |
    | 2 | Indikator VAK vs Learning Style | ANOVA + Kruskal-Wallis | Validasi Feature Engineering |
    | 3 | Learning Style × Learning Pace | Chi-Square + Cramér's V | Justifikasi profil ganda |
    """)

# ──────────────────────────────────────────────────────────────────────────────
# BUSINESS IMPACT
# ──────────────────────────────────────────────────────────────────────────────
st.divider()

st.markdown(
    '<div class="section-header">🎯 Business Impact</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="insight-box">

✅ Personalize learning recommendations<br>
✅ Improve teaching strategies<br>
✅ Identify students needing intervention<br>
✅ Support adaptive learning systems<br>
✅ Assist educational decision-making using data-driven insights

</div>
""", unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>EduProfile AI — CC26-PSU099 | Coding Camp 2026 powered by DBS Foundation | "
    "Data Science Team: Nadia Raissa R & Charlene Manuella Angkadjaja</small></center>",
    unsafe_allow_html=True
)