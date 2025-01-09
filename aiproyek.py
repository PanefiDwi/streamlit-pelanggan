import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title("Clustering Pelanggan Menggunakan K-Means")
#DOWNLOAD FILE CSV
file_path = "cus_data_bersih.csv"

@st.cache_data
def load_file(file_path):
    with open(file_path, "rb") as f:
        return f.read()

csv = load_file(file_path)

st.download_button(
    label="Download cus_data_bersih.csv",
    data=csv,
    file_name="cus_data_bersih.csv",
    mime="text/csv")

#INPUT FILE CSV
def upload_csv_file():
    uploaded_file = st.file_uploader("Unggah file CSV pelanggan:", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data CSV pelanggan:")
        st.dataframe(df) 
        st.dataframe(df.describe())
        return df
    else:
        st.warning("Silakan unggah file CSV terlebih dahulu.")
        return None
#FUNGSI TF IDF REV (ALL SHOW)
def tf_idf(df, text_columns):
    missing_columns = [col for col in text_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Kolom berikut tidak ada dalam dataset: {missing_columns}. Ganti dengan nama kolom yang benar.")
        return None

    st.write(f"Menghitung TF-IDF untuk kolom: {text_columns}")
    df['combined_text'] = df[text_columns].fillna("").apply(lambda x: " ".join(x.astype(str)), axis=1)

    df['tokenized'] = df['combined_text'].apply(lambda x: x.lower().split())

    vocabulary = set(word for tokens in df['tokenized'] for word in tokens)
    vocabulary = sorted(vocabulary)
    # st.write(f"Jumlah kata unik (vocabulary): {len(vocabulary)}")

    def compute_tf(tokens, vocabulary):
        tf = dict.fromkeys(vocabulary, 0)
        total_words = len(tokens)
        if total_words == 0:
            return {word: 0 for word in tf}
        for word in tokens:
            if word in tf:
                tf[word] += 1
        return {word: count / total_words for word, count in tf.items()}

    df['tf'] = df['tokenized'].apply(lambda x: compute_tf(x, vocabulary))

    # Menghitung DF
    def compute_df(df, vocabulary):
        df_count = dict.fromkeys(vocabulary, 0)
        for tokens in df['tokenized']:
            unique_tokens = set(tokens)
            for word in unique_tokens:
                if word in df_count:
                    df_count[word] += 1
        return df_count

    df_count = compute_df(df, vocabulary)
    total_docs = len(df)

    # Menghitung IDF
    idf = {word: np.log(total_docs / (1 + df_count[word])) for word in vocabulary}
    def compute_tfidf(tf, idf):
        return {word: tf[word] * idf[word] for word in tf}

    df['tfidf'] = df['tf'].apply(lambda x: compute_tfidf(x, idf))
    tfidf_df = pd.DataFrame(df['tfidf'].tolist()).fillna(0)

    # Gabungkan data asli dengan TF-IDF
    df_combined = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)

    st.write("Data dengan TF-IDF:")
    st.dataframe(df_combined.head())

    return df_combined
    
#FUNGSI MENAMPILKAN KORELASI MATRIKS
def tampilan_korelasi_matriks(tfidf_df):
    try:
        # Memilih hanya kolom numerik
        tfidf_numeric = tfidf_df.select_dtypes(include=['float64', 'int64'])
        if tfidf_numeric.empty:
            st.error("Tidak ada kolom numerik untuk menghitung korelasi.")
            return
        st.write("Data numerik (TF-IDF):")
        st.dataframe(tfidf_numeric.head())
 
        correlation_matrix = tfidf_numeric.corr()
        
        # Membuat heatmap
        st.write("Heatmap Korelasi:")
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
        plt.title('Korelasi Matriks TF-IDF')
        
        # Menampilkan plot di Streamlit
        st.pyplot(plt)
        plt.close()
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

#FUNGSI MEMILIH FITUR
def pilih_fitur(df):
    if df is not None:
        st.write("Pilih fitur untuk pengelompokan:")
        fitur_main = st.selectbox("Pilih fitur utama:", df.columns)
        fitur1 = st.selectbox("Pilih fitur pertama (untuk rata-rata) sebagai monetary value:", df.columns)
        fitur2 = st.selectbox("Pilih fitur kedua (untuk jumlah unik) sebagai frekuensi:", df.columns)
        fitur3 = st.selectbox("Pilih fitur ketiga (untuk rata-rata) sebagai kepuasan pelanggan (satisfaction):", df.columns)
        
        st.write(f"Fitur yang dipilih:\n- Fitur utama: {fitur_main}\n- Fitur 1: {fitur1}\n- Fitur 2: {fitur2}\n- Fitur 3: {fitur3}")
        st.warning("Pastikan fitur input sudah sesuai")
        return fitur_main, fitur1, fitur2, fitur3
    else:
        st.warning("Tidak ada data untuk memilih fitur.")
        return None, None, None, None

#FUNGSI OPERASI FITUR
def membuat_kriteria_pengelompokan(df, fitur_main, fitur1, fitur2, fitur3):
    st.write(f"Memulai pengelompokan dengan fitur:\n- {fitur_main}, {fitur1}, {fitur2}, {fitur3}")
    
    aggregated_df = df.groupby(by=fitur_main, as_index=False).agg(
        MonetaryValue=(fitur1, "mean"),
        Frequency=(fitur2, "nunique"),
        Satisfaction=(fitur3, "mean")
    )
    
    st.write("Hasil Pengelompokan:")
    st.dataframe(aggregated_df)  # Menampilkan 10 baris pertama hasil pengelompokan
    
    return aggregated_df

# FUNGSI MEMISAHKAN OUTLIER
def minimalisir_outlier(df, fitur_list):

    outliers_df_list = []
    non_outliers_df = df.copy()  # Salin data asli untuk memisahkan non-outlier nanti

    for fitur in fitur_list:
        Q1 = df[fitur].quantile(0.25)
        Q3 = df[fitur].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Mendapatkan outlier
        outliers = df[(df[fitur] < lower_bound) | (df[fitur] > upper_bound)]
        outliers_df_list.append(outliers)

        # Menghapus outlier dari non-outliers
        non_outliers_df = non_outliers_df[~non_outliers_df.index.isin(outliers.index)]

    # Menggabungkan semua outlier untuk analisis
    outliers_df = pd.concat(outliers_df_list).drop_duplicates()

    return outliers_df, non_outliers_df

# FUNGSI UNTUK MENAMPILKAN HASIL OUTLIER
def menampilkan_hasil_outlier(outliers_df, non_outliers_df):

    st.write("**Ringkasan Data Outlier:**")
    st.dataframe(outliers_df.describe())
    
    st.write("**Ringkasan Data Non-Outlier:**")
    st.dataframe(non_outliers_df.describe())

# MENAMPILKAN SCATTER PLOT DATA SET
def tampilan_scatterplot_data(non_outliers_df, fitur_list):
    try:
        # Membuat scatter plot 3D
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection="3d")

        # Pastikan kita memanggil elemen dari fitur_list dengan benar
        scatter = ax.scatter(
            non_outliers_df[fitur_list[0]],  
            non_outliers_df[fitur_list[1]],  
            non_outliers_df[fitur_list[2]],  
        )

        # Menambahkan label dan judul
        ax.set_xlabel(fitur_list[0])  
        ax.set_ylabel(fitur_list[1])  
        ax.set_zlabel(fitur_list[2])  
        ax.set_title('3D Scatter Plot of Customer Data')

        fig.colorbar(scatter, ax=ax, label=fitur_list[0])

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat membuat scatter plot: {e}")

#STANDARISASI DATA
def standarisasi(non_outliers_df):
    fitur_relevan = ["MonetaryValue", "Frequency", "Satisfaction"]
    scaled_data = non_outliers_df[fitur_relevan].copy()

    for col in scaled_data.columns:
        mean = scaled_data[col].mean()
        std = scaled_data[col].std()
        scaled_data[col] = (scaled_data[col] - mean) / std

        if scaled_data[col].isna().any():
            scaled_data[col] = scaled_data[col].fillna(mean)
            st.warning(f"Kolom '{col}' menghasilkan NaN dan digantikan dengan nilai mean.")
    
    st.write("Data setelah standarisasi:")
    st.dataframe(scaled_data)

    return scaled_data, scaled_data

#ALGORITMA K MEANS KLASTERING
def KMeans(k, data, max_iter=100):
    n_samples, n_features = data.shape
    centroids = data[np.random.choice(n_samples, k, replace=False)]
    clusters = np.zeros(n_samples, dtype=int)
    konvergen = None  # Menambahkan variabel konvergen untuk menyimpan iterasi konvergensi

    for iteration in range(max_iter):
        # Step 2: Assign data points ke cluster terdekat
        for i in range(n_samples):
            distances = np.linalg.norm(data[i] - centroids, axis=1)
            clusters[i] = np.argmin(distances)

        # Step 3: Update centroid
        new_centroids = np.zeros_like(centroids)
        for cluster_id in range(k):
            points_in_cluster = data[clusters == cluster_id]
            if len(points_in_cluster) > 0:
                new_centroids[cluster_id] = np.mean(points_in_cluster, axis=0)
            else:
                new_centroids[cluster_id] = centroids[cluster_id]

        if np.allclose(centroids, new_centroids, atol=1e-6):
            konvergen = iteration + 1  # Menyimpan iterasi konvergensi
            break

        centroids = new_centroids

    return clusters, centroids, konvergen  # Mengembalikan konvergensi

def silhouette_score(data, cluster_labels):
    n_samples = data.shape[0]
    silhouette_scores = []

    for i in range(n_samples):
        same_cluster = data[cluster_labels == cluster_labels[i]]
        other_clusters = data[cluster_labels != cluster_labels[i]]

        a = np.mean(np.linalg.norm(data[i] - same_cluster, axis=1))
        b_values = []
        for label in np.unique(cluster_labels):
            if label != cluster_labels[i]:
                other_cluster_points = data[cluster_labels == label]
                b_values.append(np.mean(np.linalg.norm(data[i] - other_cluster_points, axis=1)))

        b = np.min(b_values) if b_values else 0

        silhouette_scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0)

    return np.mean(silhouette_scores)

def calculate_inertia(data, cluster_labels, centroids):
    inertia = 0
    for i, centroid in enumerate(centroids):
        cluster_points = data[cluster_labels == i]
        inertia += np.sum((cluster_points - centroid) ** 2)
    return inertia

def menampilkan_diagram_elbow(scaled_data_df):
    max_k = 12
    inertia_values = []
    silhouette_scores_list = []
    k_values = range(2, max_k + 1)

    st.subheader("Inertia and Silhouette Scores Calculation")

    # Loop untuk menghitung inertia dan silhouette scores
    for k in k_values:
        clusters, centroids, konvergen = KMeans(k, scaled_data_df.values)  # Gunakan data dari scaled_data_df
        inertia = calculate_inertia(scaled_data_df.values, clusters, centroids)
        sil_score = silhouette_score(scaled_data_df.values, clusters)

        inertia_values.append(inertia)
        silhouette_scores_list.append(sil_score)

        # Menampilkan hasil inertia, silhouette score dan iterasi konvergensi
        st.write(f"k = {k} ---> Inertia = {inertia:.4f} ---> Silhouette Score = {sil_score:.4f} --> konvergen pada iterasi ke-{konvergen}")

    st.subheader('Elbow Method and Silhouette Analysis')

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].plot(k_values, inertia_values, marker='o', label='Inertia')
    ax[0].set_title('KMeans Inertia for Different Values of k')
    ax[0].set_xlabel('Number of Clusters (k)')
    ax[0].set_ylabel('Inertia')
    ax[0].grid(True)

    ax[1].plot(k_values, silhouette_scores_list, marker='o', color='orange', label='Silhouette Score')
    ax[1].set_title('Silhouette Scores for Different Values of k')
    ax[1].set_xlabel('Number of Clusters (k)')
    ax[1].set_ylabel('Silhouette Score')
    ax[1].grid(True)

    plt.tight_layout()
    st.pyplot(fig)

# LABELING UNTUK SETIAP KLASTER
def labeling_klaster(KMeans, scaled_data_df, non_outliers_df, k=4):
    st.subheader('Labeling Klaster')
    cluster_labels, centroids, _ = KMeans(k, scaled_data_df)

    if len(cluster_labels) != len(non_outliers_df):
        st.error("Jumlah label klaster tidak sesuai dengan jumlah data.")
        return None, None, None

    non_outliers_df["Cluster"] = cluster_labels
    st.write("Data setelah penambahan label klaster:")
    st.dataframe(non_outliers_df)
    return non_outliers_df, cluster_labels, centroids

#VISUALISASI KMEANS DENGAN PLOT 3D
def scatter_plot_KMeans(non_outliers_df):
    cluster_colors = {0: '#1f77b4',  # Blue
                  1: '#ff7f0e',  # Orange
                  2: '#2ca02c',  # Green
                  3: '#d62728'}  # Red

    colors = non_outliers_df['Cluster'].map(cluster_colors)
    # Membuat scatter plot 3D
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    scatter = ax.scatter(non_outliers_df['MonetaryValue'],
                         non_outliers_df['Frequency'],
                         non_outliers_df['Satisfaction'],
                         c=colors,
                         marker='o')
    # Menambahkan label dan judul
    ax.set_xlabel('Monetary Value')
    ax.set_ylabel('Frequency')
    ax.set_zlabel('Satisfaction')
    ax.set_title('3D Scatter Plot of Customer Data by Cluster')
    st.pyplot(fig)
    return cluster_colors, scatter

#LABELING DATA OUTLIER
def mengatasi_overlap_klaster(outliers_df):
    st.subheader('Labeling Klaster Outlier')
    if outliers_df.empty:
        st.warning("Data outliers kosong. Tidak ada yang perlu diproses.")
        return pd.DataFrame()

    required_columns = ["MonetaryValue", "Frequency", "Satisfaction"]
    missing_columns = [col for col in required_columns if col not in outliers_df.columns]
    if missing_columns:
        st.error(f"Kolom yang hilang: {', '.join(missing_columns)}")
        return pd.DataFrame()
    
    overlap_indices = outliers_df.index[
        outliers_df["MonetaryValue"].notna() &
        outliers_df["Frequency"].notna() &
        outliers_df["Satisfaction"].notna()
    ]

    monetary_only_outliers = outliers_df.loc[outliers_df["MonetaryValue"].notna() & ~outliers_df.index.isin(overlap_indices)]
    frequency_only_outliers = outliers_df.loc[outliers_df["Frequency"].notna() & ~outliers_df.index.isin(overlap_indices)]
    monetary_and_frequency_outliers = outliers_df.loc[overlap_indices]

    monetary_only_outliers = monetary_only_outliers.assign(Cluster=-1)
    frequency_only_outliers = frequency_only_outliers.assign(Cluster=-2)
    monetary_and_frequency_outliers = monetary_and_frequency_outliers.assign(Cluster=-3)

    outlier_clusters_df = pd.concat([monetary_only_outliers, frequency_only_outliers, monetary_and_frequency_outliers],
                                     ignore_index=True)

    st.write("Hasil Labeling Outliers:")
    if not outlier_clusters_df.empty:
        st.dataframe(outlier_clusters_df)
    else:
        st.warning("Tidak ada outliers yang ditemukan.")

    return outlier_clusters_df

#DIAGRAM VIOLIN UNTUK KLASTER OUTLIER
def violin_diagram_outlier(outlier_clusters_df):
    st.subheader('Visualisasi Diagram Violin untuk Klaster Outlier')
    outlier_clusters_df['Cluster'] = outlier_clusters_df['Cluster'].astype(int)
    cluster_colors = {-1: '#9467bd', -2: '#8c564b', -3: '#e377c2'}
    unique_clusters = outlier_clusters_df['Cluster'].unique()
    for cluster in unique_clusters:
        if cluster not in cluster_colors:
            cluster_colors[cluster] = '#d3d3d3'  

    plt.figure(figsize=(12, 18))
    plt.subplot(3, 1, 1)
    sns.violinplot(x=outlier_clusters_df['Cluster'],y=outlier_clusters_df['MonetaryValue'],palette=cluster_colors,hue=outlier_clusters_df["Cluster"],split=True )
    sns.violinplot(y=outlier_clusters_df['MonetaryValue'],color='gray', linewidth=1.0)
    plt.title('Monetary Value by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Monetary Value')

    plt.subplot(3, 1, 2)
    sns.violinplot(x=outlier_clusters_df['Cluster'],y=outlier_clusters_df['Frequency'],palette=cluster_colors,hue=outlier_clusters_df["Cluster"],split=True)
    sns.violinplot(y=outlier_clusters_df['Frequency'],color='gray', linewidth=1.0)
    plt.title('Frequency by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Frequency')

    plt.subplot(3, 1, 3)
    sns.violinplot(x=outlier_clusters_df['Cluster'],y=outlier_clusters_df['Satisfaction'],palette=cluster_colors,hue=outlier_clusters_df["Cluster"],split=True)
    sns.violinplot(y=outlier_clusters_df['Satisfaction'],color='gray', linewidth=1.0)
    plt.title('Satisfaction by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Satisfaction')

    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

#SEGMENTASI SELURUH KLASTER
def labeling_seluruh_klaster(non_outliers_df, outlier_clusters_df):
    st.subheader('Segmentasi Seluruh Klaster')
    cluster_labels = {
        0: "RETAIN",
        1: "RE-ENGAGE",
        2: "NURTURE",
        3: "REWARD",
        -1: "PAMPER",
        -2: "UPSELL",
        -3: "DELIGHT"
    }
    full_clustering_df = pd.concat([non_outliers_df, outlier_clusters_df])
    full_clustering_df["ClusterLabel"] = full_clustering_df["Cluster"].map(cluster_labels)
    st.dataframe(full_clustering_df)
    return full_clustering_df

#DISTRIBUSI KLASTER DENGAN RATA-RATA FITUR VALUE
def distribusi_klaster(full_clustering_df):
    st.subheader('Distribusi Klaster dengan Rata-rata Fitur')
    cluster_counts = full_clustering_df['ClusterLabel'].value_counts()
    full_clustering_df["MonetaryValue per 100 pounds"] = full_clustering_df["MonetaryValue"] / 100.0
    feature_means = full_clustering_df.groupby('ClusterLabel')[['Satisfaction', 'Frequency', 'MonetaryValue per 100 pounds']].mean()
    
    fig1, ax1 = plt.subplots(figsize=(12, 8))

    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax1, palette='viridis', hue=cluster_counts.index)
    ax1.set_ylabel('Number of Customers', color='b')
    ax1.set_title('Cluster Distribution with Average Feature Values')

    ax2 = ax1.twinx()

    sns.lineplot(data=feature_means, ax=ax2, palette='Set2', marker='o')
    ax2.set_ylabel('Average Value', color='g')

    st.write('Rata-rata nilai fitur per klaster:')
    st.dataframe(feature_means)
    st.pyplot(fig1)

# TAMPILAN WIDGETS
def tampilan_widget():
    df = upload_csv_file()
    if df is not None:
        text_column = ['gender', 'education', 'region', 'loyalty_status', 'purchase_frequency', 'product_category']
        tfidf_result = tf_idf(df, text_column)
        if tfidf_result is not None:
            tampilan_korelasi_matriks(tfidf_result)

            fitur_main, fitur1, fitur2, fitur3 = pilih_fitur(df)
            if fitur_main and fitur1 and fitur2 and fitur3:
                aggregated_df = membuat_kriteria_pengelompokan(df, fitur_main, fitur1, fitur2, fitur3)

                if aggregated_df is not None:
                    fitur_list = ['MonetaryValue', 'Frequency', 'Satisfaction']
                    outliers_df, non_outliers_df = minimalisir_outlier(aggregated_df, fitur_list)

                    menampilkan_hasil_outlier(outliers_df, non_outliers_df)
                    scaled_data, scaled_data_df = standarisasi(non_outliers_df[fitur_list])
                    tampilan_scatterplot_data(scaled_data_df, fitur_list)

                    menampilkan_diagram_elbow(scaled_data_df)
                    non_outliers_df, cluster_labels, centroids = labeling_klaster(
                        KMeans, scaled_data_df.values, non_outliers_df, k=4)
                    scatter_plot_KMeans(non_outliers_df)
                    outlier_clusters_df = mengatasi_overlap_klaster(outliers_df)
                    violin_diagram_outlier(outlier_clusters_df)
                    full_clustering_df = labeling_seluruh_klaster(non_outliers_df, outlier_clusters_df)
                    distribusi_klaster(full_clustering_df)
tampilan_widget()
