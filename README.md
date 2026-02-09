
# 📚 **DOKUMENTASI PROYEK - CLUSTERING PELANGGAN K-MEANS**
**Framework Utama**: Streamlit, Pandas, NumPy, Matplotlib, Seaborn

---

## 📋 DAFTAR ISI

1. [Gambaran Umum Proyek](#gambaran-umum-proyek)
2. [Struktur Proyek](#struktur-proyek)
3. [Teknologi & Dependencies](#teknologi--dependencies)
4. [Panduan Instalasi](#panduan-instalasi)
5. [Alur Kerja Aplikasi](#alur-kerja-aplikasi)
6. [Dokumentasi Modul Fungsi](#dokumentasi-modul-fungsi)
7. [Dataset & Fitur](#dataset--fitur)
8. [Metodologi K-Means](#metodologi-k-means)
9. [Segmentasi Pelanggan](#segmentasi-pelanggan)
10. [Cara Menggunakan Aplikasi](#cara-menggunakan-aplikasi)
11. [Troubleshooting](#troubleshooting)

---

## 🎯 GAMBARAN UMUM PROYEK
Proyek ini dirancang untuk memahami perilaku pelanggan melalui pendekatan unsupervised learning menggunakan algoritma **K-Means Clustering**. Dengan segmentasi pelanggan yang tepat, perusahaan dapat:

- ✅ Mengidentifikasi pelanggan bernilai tinggi
- ✅ Mengoptimalkan strategi pemasaran
- ✅ Meningkatkan loyalitas pelanggan
- ✅ Mengalokasikan sumber daya secara efisien
- ✅ Membuat campaign yang lebih personalized

### Hasil Akhir
Segmentasi pelanggan ke dalam **7 kategori strategis**:
- **RETAIN** - Pelanggan setia yang perlu dipertahankan
- **RE-ENGAGE** - Pelanggan yang jarang bertransaksi
- **NURTURE** - Pelanggan dengan potensi tinggi
- **REWARD** - Pelanggan loyal dengan reward
- **PAMPER** - Pelanggan dengan monetary tinggi (outlier)
- **UPSELL** - Pelanggan frequent tapi monetary rendah (outlier)
- **DELIGHT** - Pelanggan premium dengan semua aspek terbaik (outlier)

---

## 📁 STRUKTUR PROYEK

```
streamlit-pelanggan/
├── aiproyek.py                    # File utama aplikasi Streamlit
├── cus_data_bersih.csv            # Dataset pelanggan (6.7 MB)
├── requirements.txt               # Daftar dependency Python
├── README.md                       # Overview proyek (format lama)
├── .devcontainer/                 # Konfigurasi development container
└── DOKUMENTASI.md                 # Dokumentasi ini
```

### Penjelasan File Utama

| File | Ukuran | Fungsi |
|------|--------|--------|
| **aiproyek.py** | 20.5 KB | File aplikasi utama dengan semua fungsi dan logic |
| **cus_data_bersih.csv** | 6.7 MB | Dataset pelanggan bersih siap diproses |
| **requirements.txt** | 50 bytes | Daftar pustaka Python yang diperlukan |

---

## ⚙️ TEKNOLOGI & DEPENDENCIES

### Technology Stack

```python
# Core Data Processing
numpy          # Komputasi numerik dan array operations
pandas         # Data manipulation dan aggregation

# Visualization
matplotlib     # Static plotting
seaborn        # Statistical data visualization

# Web Framework
streamlit      # Interactive web application framework

# Package Management
pip            # Python package installer
```

### Versi Minimum yang Disarankan
- Python 3.8+
- NumPy 1.20+
- Pandas 1.3+
- Matplotlib 3.4+
- Seaborn 0.11+
- Streamlit 1.0+

---

## 🚀 PANDUAN INSTALASI

### Prasyarat
- Python 3.8+ sudah terinstall
- pip (Python package manager) sudah tersedia
- Terminal/Command Prompt yang dapat diakses

### Langkah Instalasi

#### 1️⃣ Clone Repository
```bash
git clone https://github.com/PanefiDwi/streamlit-pelanggan.git
cd streamlit-pelanggan
```

#### 2️⃣ Buat Virtual Environment (Opsional tapi Disarankan)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

Atau install manual:
```bash
pip install numpy pandas streamlit matplotlib seaborn
```

#### 4️⃣ Verifikasi Instalasi
```bash
python -c "import streamlit; print(f'Streamlit {streamlit.__version__} installed!')"
python -c "import pandas; import numpy; print('Pandas & NumPy OK')"
```

---

## 📊 ALUR KERJA APLIKASI

### Diagram Alur Proses

```
START
  ↓
[1. Upload CSV] → Validasi file & load data
  ↓
[2. TF-IDF Processing] → Ekstrak fitur text
  ↓
[3. Korelasi Matriks] → Tampilkan heatmap
  ↓
[4. Pilih Fitur] → User memilih kolom untuk aggregation
  ↓
[5. Aggregation] → Groupby fitur utama, hitung Mean/Nunique
  ↓
[6. Outlier Detection] → Gunakan IQR method
  ↓
[7. Data Split] → Pisahkan outlier & non-outlier
  ↓
[8. Standardisasi] → Z-score normalization
  ↓
[9. Elbow Method] → Tentukan optimal k (2-12)
  ↓
[10. K-Means Clustering] → Clustering dengan k=4
  ↓
[11. Labeling Cluster] → Assign label untuk non-outlier
  ↓
[12. Outlier Labeling] → Assign label untuk outlier (-1, -2, -3)
  ↓
[13. Merge & Visualisasi] → Gabungkan data & tampilkan hasil
  ↓
END
```

### 7 Tahapan Utama Proses

| Tahap | Fungsi | Input | Output |
|-------|--------|-------|--------|
| **1. EDA & TF-IDF** | Eksplorasi data & ekstraksi fitur | CSV mentah | Data dengan TF-IDF scores |
| **2. Korelasi** | Analisis hubungan antar fitur | Data TF-IDF | Heatmap korelasi |
| **3. Aggregation** | Agregasi per pelanggan | Data asli | Monetary, Frequency, Satisfaction |
| **4. Outlier Detection** | Identifikasi outlier dengan IQR | Agregat data | Split outlier/non-outlier |
| **5. Standardisasi** | Normalisasi dengan Z-score | Non-outlier | Data terstandarisasi (0 mean, 1 std) |
| **6. K-Means** | Clustering custom implementation | Normalized | Cluster assignments & centroids |
| **7. Labeling** | Assign label bisnis ke cluster | Clusters | Segmentasi final dengan label |

---

## 📖 DOKUMENTASI MODUL FUNGSI

### 1️⃣ MODUL UPLOAD & LOADING DATA

#### `load_file(file_path)`
**Fungsi**: Memuat file CSV dengan caching

```python
@st.cache_data
def load_file(file_path):
    with open(file_path, "rb") as f:
        return f.read()
```

| Parameter | Tipe | Deskripsi |
|-----------|------|-----------|
| `file_path` | str | Path ke file CSV |
| **Return** | bytes | Content file dalam format bytes |

**Kegunaan**: Download button untuk sample dataset

---

#### `upload_csv_file()`
**Fungsi**: Widget upload CSV dari user

**Output UI**:
- File uploader widget
- Dataframe preview
- Statistik deskriptif (describe())

**Return**: `pd.DataFrame` atau `None`

**Pengecekan**:
- ✓ File harus bertipe CSV
- ✓ Data ditampilkan dengan `st.dataframe()`
- ✓ Statistik otomatis ditampilkan

---

### 2️⃣ MODUL TF-IDF & TEXT PROCESSING

#### `tf_idf(df, text_columns)`
**Fungsi**: Menghitung TF-IDF score untuk kolom text

**Proses**:

```python
# Step 1: Combine & Tokenize
df['combined_text'] = combine_columns(text_columns)
df['tokenized'] = lowercase_and_split()

# Step 2: Hitung Term Frequency (TF)
tf = count_terms / total_words

# Step 3: Hitung Document Frequency (DF)
df_count = berapa dokumen mengandung term

# Step 4: Hitung Inverse Document Frequency (IDF)
idf = log(total_docs / (1 + df_count[word]))

# Step 5: Hitung TF-IDF
tfidf = tf * idf
```

**Parameter**:
- `df`: DataFrame dengan kolom text
- `text_columns`: List kolom untuk ekstraksi

**Return**: 
```python
{
    'original_data': DataFrame asli,
    'tfidf_features': DataFrame dengan fitur TF-IDF,
    'combined_data': Gabungan original + TF-IDF
}
```

**Kolom Text Default**:
- `gender`, `education`, `region`
- `loyalty_status`, `purchase_frequency`, `product_category`

---

#### `tampilan_korelasi_matriks(tfidf_df)`
**Fungsi**: Visualisasi korelasi antar fitur numerik

**Output**:
- Heatmap 20x12 inches
- Annotation dengan nilai korelasi
- Colorbar untuk scale

**Visualisasi Parameter**:
```python
sns.heatmap(
    correlation_matrix,
    annot=True,      # Tampilkan nilai
    cmap='coolwarm', # Color gradient (blue=neg, red=pos)
    fmt='.2f',       # Format 2 decimal places
    cbar=True        # Show colorbar
)
```

---

### 3️⃣ MODUL FEATURE ENGINEERING

#### `pilih_fitur(df)`
**Fungsi**: Widget untuk user memilih kolom untuk aggregation

**Input UI** (3 selectbox):
1. **Fitur Utama**: Kolom untuk groupby (e.g., customer_id)
2. **Fitur 1 (Monetary)**: Kolom untuk mean (e.g., purchase_amount)
3. **Fitur 2 (Frequency)**: Kolom untuk nunique (e.g., transaction_id)
4. **Fitur 3 (Satisfaction)**: Kolom untuk mean (e.g., satisfaction_score)

**Button**: "Selanjutnya" untuk melanjutkan proses

**Return**: Tuple `(fitur_main, fitur1, fitur2, fitur3)` atau `(None, None, None, None)`

---

#### `membuat_kriteria_pengelompokan(df, fitur_main, fitur1, fitur2, fitur3)`
**Fungsi**: Agregasi data berdasarkan fitur yang dipilih

**Proses**:
```python
aggregated_df = df.groupby(by=fitur_main).agg({
    fitur1: "mean"        → MonetaryValue
    fitur2: "nunique"     → Frequency
    fitur3: "mean"        → Satisfaction
})
```

**Output DataFrame**:
```
| Fitur_Utama | MonetaryValue | Frequency | Satisfaction |
|-------------|---------------|-----------|--------------|
| Customer_1  | 500.5         | 12        | 8.5          |
| Customer_2  | 1200.3        | 25        | 9.2          |
| ...         | ...           | ...       | ...          |
```

**Return**: `pd.DataFrame` dengan 3 fitur numerik

---

### 4️⃣ MODUL OUTLIER DETECTION

#### `minimalisir_outlier(df, fitur_list)`
**Fungsi**: Deteksi outlier menggunakan Interquartile Range (IQR)

**Formula IQR**:
```
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
Lower Bound = Q1 - 1.5 * IQR
Upper Bound = Q3 + 1.5 * IQR

Outlier: nilai < Lower Bound ATAU nilai > Upper Bound
```

**Proses**:
```python
for fitur in ['MonetaryValue', 'Frequency', 'Satisfaction']:
    # Hitung bounds
    Q1, Q3 = df[fitur].quantile([0.25, 0.75])
    # Identifikasi outlier
    outliers = df[(df[fitur] < lower) | (df[fitur] > upper)]
    # Kumpulkan
    outliers_list.append(outliers)
```

**Return**: 
- `outliers_df`: Data outlier
- `non_outliers_df`: Data normal

---

#### `menampilkan_hasil_outlier(outliers_df, non_outliers_df)`
**Fungsi**: Tampilkan statistik perbandingan

**Output**:
```
RINGKASAN DATA OUTLIER:
count    mean    std     min     25%     50%     75%     max
...

RINGKASAN DATA NON-OUTLIER:
count    mean    std     min     25%     50%     75%     max
...
```

---

### 5️⃣ MODUL VISUALISASI DATA

#### `tampilan_scatterplot_data(non_outliers_df, fitur_list)`
**Fungsi**: Scatter plot 3D dari data non-outlier

**Aksis**:
- X-axis: MonetaryValue
- Y-axis: Frequency
- Z-axis: Satisfaction
- Color: MonetaryValue gradient

**Fitur**:
- Rotatable 3D plot
- Color gradient scale
- Legend untuk interpretasi

---

### 6️⃣ MODUL STANDARDISASI DATA

#### `standarisasi(non_outliers_df)`
**Fungsi**: Normalisasi menggunakan Z-score

**Formula Z-Score**:
```
X_scaled = (X - mean) / std_dev

Hasil: 
- Mean = 0
- Std Dev = 1
- Range ≈ [-3, 3]
```

**Input**: Non-outlier data dengan kolom:
- MonetaryValue
- Frequency
- Satisfaction

**Return**: DataFrame terstandarisasi (2x return yang sama)

**NaN Handling**: Jika ada NaN hasil normalisasi → ganti dengan mean

---

### 7️⃣ MODUL K-MEANS CLUSTERING

#### `KMeans(k, data, max_iter=100)`
**Fungsi**: Custom implementation K-Means clustering algorithm

**Parameter**:
- `k`: Jumlah cluster (default 4)
- `data`: NumPy array data normalized
- `max_iter`: Iterasi maksimal (default 100)

**Algoritma**:

```
Step 1: INISIALISASI CENTROID
  - Pilih k sample random dari data
  - Centroid pertama

Step 2: ASSIGN CLUSTER (repeat)
  - Hitung jarak Euclidean: ||data_point - centroid||
  - Assign ke cluster terdekat

Step 3: UPDATE CENTROID
  - Hitung mean setiap cluster
  - Centroid baru = mean dari points dalam cluster

Step 4: CHECK KONVERGENSI
  - If centroid_baru ≈ centroid_lama (tolerance 1e-6)
  - STOP iterasi (CONVERGED)
  - Else lanjutkan iterasi

RETURN:
- cluster_labels: Array cluster assignment
- centroids: Final centroid positions
- konvergen: Iterasi ke berapa konvergen
```

**Return**:
```python
(
    clusters: np.array [0,1,2,3,1,0,...],  # Label cluster setiap point
    centroids: np.array [[x,y,z],...],    # Posisi centroid
    konvergen: int                         # Iterasi konvergen
)
```

---

#### `silhouette_score(data, cluster_labels)`
**Fungsi**: Hitung Silhouette Score untuk evaluasi clustering

**Formula**:
```
For each point i:
  a = mean distance ke semua points dalam cluster yang sama
  b = min(mean distance ke points dalam cluster lain)
  s(i) = (b - a) / max(a, b)

Silhouette Score = mean dari semua s(i)

Range: [-1, 1]
- 1 = Clustering sempurna
- 0 = Overlap cluster
- -1 = Misclassified
```

**Return**: Float [-1.0 to 1.0]

---

#### `calculate_inertia(data, cluster_labels, centroids)`
**Fungsi**: Hitung Inertia (sum of squared distances)

**Formula**:
```
Inertia = Σ(||point - centroid_terdekat||²)
          untuk semua points dalam cluster

Semakin kecil = semakin tight clusters
```

**Return**: Float (inertia value)

---

#### `menampilkan_diagram_elbow(scaled_data_df)`
**Fungsi**: Elbow Method & Silhouette Analysis

**Proses**:
```python
for k in range(2, 13):
    clusters, centroids, konvergen = KMeans(k, data)
    inertia = calculate_inertia(...)
    sil_score = silhouette_score(...)
    # Tampilkan hasil
```

**Output**:
- **Elbow Chart**: Inertia vs k
- **Silhouette Chart**: Silhouette Score vs k
- **Console Output**: Detail setiap k dengan konvergen iteration

**Interpretasi Elbow**:
```
Inertia
  |     
  |  \
  |   \___
  |       \___
  |____________
  2  4  6  8  10 12 k
  
  "Elbow" = Titik dimana kurva mulai flat
  (diminishing return)
```

---

### 8️⃣ MODUL CLUSTER LABELING

#### `labeling_klaster(KMeans, scaled_data_df, non_outliers_df, k=4)`
**Fungsi**: Assign cluster label ke data

**Proses**:
```python
# Run K-Means dengan k=4
cluster_labels, centroids, _ = KMeans(k, scaled_data_df)

# Assign label ke original data
non_outliers_df['Cluster'] = cluster_labels
# Output: Cluster values [0, 1, 2, 3]
```

**Validasi**:
- ✓ Jumlah label == jumlah data
- ✗ Jika tidak match → error

**Return**: 
- Updated DataFrame dengan kolom 'Cluster'
- Cluster labels array
- Centroids

---

#### `scatter_plot_KMeans(non_outliers_df)`
**Fungsi**: Visualisasi 3D clustering hasil

**Warna Cluster**:
```python
{
    0: '#1f77b4',  # Blue - RETAIN
    1: '#ff7f0e',  # Orange - RE-ENGAGE
    2: '#2ca02c',  # Green - NURTURE
    3: '#d62728'   # Red - REWARD
}
```

**Plot**:
- X: Monetary Value
- Y: Frequency
- Z: Satisfaction
- Color: Cluster assignment

---

### 9️⃣ MODUL OUTLIER LABELING

#### `mengatasi_overlap_klaster(outliers_df)`
**Fungsi**: Assign label khusus untuk data outlier

**Logika Kategorisasi**:

```python
# Deteksi outlier dimension:
# 1. Monetary only (Ada pada monetary, tidak pada lain)
#    → Cluster = -1 (PAMPER)

# 2. Frequency only (Ada pada frequency, tidak pada lain)
#    → Cluster = -2 (UPSELL)

# 3. Overlap (Ada di 2+ dimension)
#    → Cluster = -3 (DELIGHT)
```

**Output**:
```
| Index | MonetaryValue | Frequency | Satisfaction | Cluster |
|-------|---------------|-----------|--------------|---------|
| 5     | 5000          | NaN       | NaN          | -1      |
| 12    | NaN           | 100       | NaN          | -2      |
| 20    | 8000          | 80        | 9.5          | -3      |
```

**Return**: DataFrame dengan kolom Cluster (-1, -2, -3)

---

#### `violin_diagram_outlier(outlier_clusters_df)`
**Fungsi**: Visualisasi distribusi outlier dengan Violin Plot

**Subplots** (3 baris):
1. **Monetary Value by Cluster**: Distribusi nilai monetary per cluster outlier
2. **Frequency by Cluster**: Distribusi frekuensi per cluster outlier
3. **Satisfaction by Cluster**: Distribusi kepuasan per cluster outlier

**Warna**:
```python
{
    -1: '#9467bd',  # Purple - PAMPER
    -2: '#8c564b',  # Brown - UPSELL
    -3: '#e377c2'   # Pink - DELIGHT
}
```

---

### 🔟 MODUL FINAL SEGMENTATION

#### `labeling_seluruh_klaster(non_outliers_df, outlier_clusters_df)`
**Fungsi**: Merge data & assign business labels

**Cluster → Business Label Mapping**:
```python
{
    0: "RETAIN",       # Normal cluster 0
    1: "RE-ENGAGE",    # Normal cluster 1
    2: "NURTURE",      # Normal cluster 2
    3: "REWARD",       # Normal cluster 3
    -1: "PAMPER",      # Outlier monetary high
    -2: "UPSELL",      # Outlier frequency high
    -3: "DELIGHT"      # Outlier premium
}
```

**Proses**:
```python
# Merge non-outlier & outlier
full_df = concat([non_outliers_df, outlier_clusters_df])

# Map cluster number ke label
full_df['ClusterLabel'] = full_df['Cluster'].map(cluster_labels)
```

**Output**: DataFrame dengan kolom baru 'ClusterLabel'

---

#### `distribusi_klaster(full_clustering_df)`
**Fungsi**: Analisis distribusi dan statistik per cluster

**Output 1 - Bar Chart**:
```
Sumbu X: Cluster Label (RETAIN, RE-ENGAGE, dst)
Sumbu Y: Jumlah Customer
Warna: Viridis palette
```

**Output 2 - Line Chart** (Twin Axis):
```
Sumbu Y (kanan): Rata-rata fitur
3 line plot:
  - Satisfaction
  - Frequency
  - MonetaryValue (per 100 pounds)
```

**Output 3 - Statistics Table**:
```
                  Satisfaction  Frequency  MonetaryValue per 100
ClusterLabel                                          
RETAIN            7.2           15.3       3.5
RE-ENGAGE         5.8           4.2        2.1
NURTURE           8.1           20.1       5.2
...
```

---

### 1️⃣1️⃣ MODUL ORCHESTRATION

#### `tampilan_widget()`
**Fungsi**: Main orchestrator - menjalankan seluruh pipeline

**Sequence**:
```python
1. Upload CSV → df
2. TF-IDF Processing → tfidf_result
3. Display Korelasi Matriks
4. Pilih Fitur → fitur_main, fitur1, fitur2, fitur3
5. Aggregation → aggregated_df
6. Outlier Detection → outliers_df, non_outliers_df
7. Display Outlier Summary
8. Standardisasi → scaled_data
9. 3D Scatter Plot
10. Elbow Method & Silhouette
11. K-Means Clustering (k=4)
12. 3D Cluster Visualization
13. Outlier Labeling
14. Violin Plot
15. Merge & Label
16. Distribusi Chart & Stats
```

**Kondisi**:
- Setiap step tergantung output sebelumnya
- Jika ada error → stop dan display error message

---

## 📊 DATASET & FITUR

### Struktur Dataset

```csv
gender, education, region, loyalty_status, purchase_frequency, 
product_category, purchase_amount, satisfaction_score, ...
```

### Kolom Utama

**Elbow Method**:
- Test k = 2 hingga 12
- Hitung inertia untuk setiap k
- Cari "elbow point" = diminishing return

**Silhouette Score**:
- Measure cohesion (kechatan dalam cluster)
- Range [-1, 1], semakin tinggi semakin baik
- Optimal k = highest silhouette score

### Tahap 3: K-Means Clustering

**Custom Implementation** (bukan sklearn):
1. Random initialization: pilih k centroid
2. Repeat:
   - Assign: point → centroid terdekat
   - Update: centroid = mean cluster
3. Until: centroid tidak berubah (converged)

### Tahap 4: Cluster Interpretation

**Strategi Labeling**:
- Analisis karakteristik setiap cluster
- Non-outlier → label berdasarkan cluster number (0,1,2,3)
- Outlier → label berdasarkan dimensi outlier (-1,-2,-3)
- Map ke business labels (RETAIN, DELIGHT, dst)

---

## 👥 SEGMENTASI PELANGGAN

### 7 Segmen Pelanggan

#### 1. **RETAIN** (Cluster 0)
- **Karakteristik**: Pelanggan normal dengan performa cukup
- **Monetary Value**: Medium
- **Frequency**: Medium
- **Satisfaction**: Medium
- **Strategi**: Maintain relationship, regular communication

#### 2. **RE-ENGAGE** (Cluster 1)
- **Karakteristik**: Pelanggan dormant/jarang transaksi
- **Monetary Value**: Low
- **Frequency**: **Low** ← Key indicator
- **Satisfaction**: Varies
- **Strategi**: Win-back campaign, special offers, reminder

#### 3. **NURTURE** (Cluster 2)
- **Karakteristik**: Pelanggan berkembang dengan potensi tinggi
- **Monetary Value**: Medium-High
- **Frequency**: Medium-High
- **Satisfaction**: High
- **Strategi**: Growth programs, upsell, exclusive benefits

#### 4. **REWARD** (Cluster 3)
- **Karakteristik**: Pelanggan loyal bernilai tinggi
- **Monetary Value**: High
- **Frequency**: Medium-High
- **Satisfaction**: High
- **Strategi**: VIP treatment, exclusive products, loyalty rewards

#### 5. **PAMPER** (Cluster -1)
- **Karakteristik**: Nilai monetary SANGAT TINGGI (outlier)
- **Monetary Value**: **VERY HIGH** ← Outlier
- **Frequency**: Varies
- **Satisfaction**: Varies
- **Strategi**: Premium service, personalized attention, concierge

#### 6. **UPSELL** (Cluster -2)
- **Karakteristik**: Frekuensi TINGGI tapi monetary RENDAH (outlier)
- **Monetary Value**: **LOW** ← Outlier
- **Frequency**: **VERY HIGH** ← Outlier
- **Satisfaction**: Varies
- **Strategi**: Upsell higher-value products, bundle offers, premium upgrade

#### 7. **DELIGHT** (Cluster -3)
- **Karakteristik**: PREMIUM di semua dimensi (overlap outlier)
- **Monetary Value**: **VERY HIGH** ← Outlier
- **Frequency**: **VERY HIGH** ← Outlier
- **Satisfaction**: **HIGH**
- **Strategi**: Exclusive VIP program, product co-creation, ambassador program


## 🎮 CARA MENGGUNAKAN APLIKASI
### **Langkah Penggunaan**

#### Step 1: Start Application
```bash
streamlit run aiproyek.py
```
#### Step 2: Upload Dataset
- Click "Unggah file CSV pelanggan:"
- Select file CSV
- Preview data + statistics ditampilkan otomatis

#### Step 3: Download Sample
- Jika ingin coba, click "Download cus_data_bersih.csv"
- File akan didownload sebagai reference
#### Step 6: Interpret Results
- Review segmentasi di table akhir
- Lihat distribusi cluster
- Analisis karakteristik per segmen
- Export insights untuk business decision

### Tips Penggunaan

| Tip | Kegunaan |
|-----|----------|
| Scroll down | Lihat semua visualisasi & tabel |
| Interact 3D | Rotate, zoom plot dengan mouse |
| Check console | Lihat iterasi konvergen K-Means |
| Note k value | Elbow method membantu tentukan optimal k |
| Compare scores | Silhouette vs Inertia untuk validasi |

## Deskripsi Output


1. RETAIN → Focus: Maintain relationships
   - Regular communication
   - Personalized offers
2. RE-ENGAGE → Focus: Win-back
   - Incentive campaigns
   - Re-activation 
3. NURTURE → Focus: Growth
   - Upsell opportunities
   - Premium 
4. REWARD → Focus: Retention
   - VIP benefits
   - Loyalty 
5. PAMPER → Focus: Premium experience
   - Concierge service
   - Exclusive 
6. UPSELL → Focus: Monetization
   - Higher-value products
   - Premium 
7. DELIGHT → Focus: Lifetime value
   - Ambassador program
   - Exclusive events

---
## ✨ KESIMPULAN

Aplikasi ini menyediakan **comprehensive solution** untuk:

✅ **Memahami** perilaku pelanggan melalui data-driven approach
✅ **Mensegmentasi** pelanggan ke 7 kategori strategis
✅ **Mengidentifikasi** high-value customers & at-risk segments
✅ **Mendukung** business decisions dengan statistical rigor

Dengan metodologi **K-Means Clustering yang robust** dan **interactive visualization**, stakeholder dapat dengan mudah:
- Melihat customer segments
- Memahami karakteristik setiap segment
- Membuat targeted strategies
- Optimize marketing budget allocation
