Understanding Customer's Behavior with Using K-Means Clustering Algorithm
📌 Overview

Project ini bertujuan untuk memahami perilaku pelanggan (customer behavior) melalui pendekatan unsupervised learning menggunakan algoritma K-Means Clustering.

Analisis dilakukan terhadap dataset cus_data_bersih.csv untuk mengidentifikasi pola berdasarkan:

Monetary Value (nilai rata-rata pembelian)

Frequency (frekuensi transaksi unik)

Satisfaction (tingkat kepuasan pelanggan)

Hasil clustering digunakan untuk melakukan segmentasi pelanggan ke dalam beberapa kategori strategis seperti:

RETAIN

RE-ENGAGE

NURTURE

REWARD

PAMPER

UPSELL

DELIGHT

Project ini di-deploy menggunakan Streamlit sebagai aplikasi interaktif berbasis web.

📂 Dataset

Dataset yang digunakan:

cus_data_bersih.csv


Dataset ini berisi data pelanggan seperti:

gender

education

region

loyalty_status

purchase_frequency

product_category

purchase_amount

satisfaction_score

dan fitur numerik lainnya

⚙️ Tech Stack

Python

Streamlit

Pandas

NumPy

Matplotlib

Seaborn

🧠 Methodology

Project ini mengikuti tahapan berikut:

1️⃣ Exploratory Data Analysis (EDA)

Menampilkan statistik deskriptif

Visualisasi distribusi data

Deteksi outlier menggunakan metode IQR

Korelasi fitur menggunakan TF-IDF dan heatmap

2️⃣ Feature Engineering

Aggregasi data berdasarkan fitur utama

Perhitungan:

MonetaryValue (mean)

Frequency (nunique)

Satisfaction (mean)

3️⃣ Outlier Handling

Menggunakan metode IQR:

Q1 - 1.5 * IQR
Q3 + 1.5 * IQR


Outlier dipisahkan menjadi:

Monetary only

Frequency only

Overlap

4️⃣ Standardization

Normalisasi menggunakan Z-score:

(x - mean) / std

5️⃣ K-Means Clustering (Custom Implementation)

Algoritma K-Means diimplementasikan manual:

Inisialisasi centroid secara acak

Hitung jarak Euclidean

Assign cluster

Update centroid

Iterasi hingga konvergen

6️⃣ Model Evaluation

Elbow Method (Inertia)

Silhouette Score

Rentang k yang diuji: k = 2 sampai 12

7️⃣ Cluster Labeling

Cluster in-layer:

0 → RETAIN

1 → RE-ENGAGE

2 → NURTURE

3 → REWARD

Cluster out-layer:

-1 → PAMPER

-2 → UPSELL

-3 → DELIGHT

📊 Visualizations

Aplikasi menghasilkan:

Heatmap korelasi TF-IDF

3D Scatter Plot (Customer Data)

Elbow & Silhouette Chart

3D Cluster Visualization

Violin Plot untuk Outlier

Distribusi Cluster dengan rata-rata fitur

🚀 How to Run
1️⃣ Install dependencies
pip install streamlit pandas numpy matplotlib seaborn

2️⃣ Jalankan Streamlit App
streamlit run nama_file.py


Ganti nama_file.py dengan nama file Python Anda.

📈 Business Insights

Segmentasi pelanggan memungkinkan perusahaan untuk:

Mengidentifikasi pelanggan bernilai tinggi

Mengoptimalkan strategi pemasaran

Meningkatkan loyalitas pelanggan

Mengalokasikan sumber daya secara efisien

Membuat campaign yang lebih personalized

Contoh insight:

Pelanggan dengan MonetaryValue tinggi dan Frequency tinggi → DELIGHT

Pelanggan dengan Frequency tinggi namun Monetary rendah → UPSELL

Pelanggan jarang transaksi → RE-ENGAGE

🎯 Project Goals

Mengidentifikasi pola perilaku pelanggan

Mengelompokkan pelanggan berdasarkan karakteristik serupa

Memberikan rekomendasi strategis berbasis data

Mengembangkan model clustering berbasis implementasi manual

📌 Deployment

Model di-deploy menggunakan Streamlit dan masih dalam tahap pengembangan lanjutan.

👥 Use Case

Project ini dapat digunakan untuk:

Retail Business

E-commerce

Banking

CRM Optimization

Customer Segmentation Strategy

📜 Conclusion

Dengan pendekatan K-Means Clustering, perusahaan dapat memahami karakteristik pelanggan secara lebih sistematis dan berbasis data, sehingga strategi bisnis dapat lebih terarah dan efisien.
