import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
from math import isnan
from matplotlib import colors
from matplotlib.colors import ListedColormap

data =pd.read_csv('incident_event_log.csv')

st.title("Menampilkan 5 Data Teratas")
st.dataframe(data.head())

st.title("Menampilkan 5 Data Terbawah")
st.dataframe(data.tail())

st.title("Visualisasi Categorical Data")
cat_cols = ['incident_state', 'priority', 'impact', 'urgency']

# Membuat layout 2x2 dengan Streamlit columns
cols = st.columns(2)

# Membuat plot untuk setiap kolom kategorikal dan menampilkan dalam layout 2x2
for i, col in enumerate(cat_cols):
    # Menggunakan kolom yang tepat berdasarkan indeks
    with cols[i % 2]:
        # Membuat plot dengan Matplotlib dan Seaborn
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=data, x=col, ax=ax)
        ax.tick_params(axis='x', rotation=45)
        ax.set_title(f"Distribution of {col}")
        
        # Menampilkan plot di kolom Streamlit
        st.pyplot(fig)

# Data yang preprocessing
data_preprocessing=pd.read_csv('closed_data.csv')

st.title('EDA Untuk Categorical Column Setelah Preprocessing')
cat_cols = ['reassignment_count', 'reopen_count', 'contact_type', 'category',
            'impact', 'urgency', 'priority', 'knowledge', 'u_priority_confirmation',
            'notify', 'closed_code']

cols_per_row = 2  # Jumlah kolom per baris

# Membuat visualisasi dalam layout 2 kolom menggunakan Streamlit columns
for i in range(0, len(cat_cols), cols_per_row):
    cols = st.columns(cols_per_row)  # Membuat 2 kolom per baris

    for j, col in enumerate(cat_cols[i:i + cols_per_row]):
        with cols[j]:  # Menampilkan setiap plot di kolom yang sesuai
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.countplot(data=data_preprocessing, x=col, ax=ax, palette='Set3')
            ax.set_title(f"Distribusi Kolom {col}")
            ax.tick_params(axis='x', rotation=45)

            # Menambahkan label jumlah pada setiap bar
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2, p.get_height()),
                            ha='center', va='center', xytext=(0, 10), textcoords='offset points')

            st.pyplot(fig)  # Menampilkan plot di Streamlit

data_PCA=pd.read_csv('PCA_ds.csv')
st.title('Visualisasi PCA')
# Menentukan nilai x, y, dan z dari kolom DataFrame
x = data_PCA["col1"]
y = data_PCA["col2"]
z = data_PCA["col3"]

# Membuat visualisasi 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x, y, z, c="maroon", marker="o")
ax.set_title("A 3D Projection Of Data In The Reduced Dimension")
ax.set_xlabel("col1")
ax.set_ylabel("col2")
ax.set_zlabel("col3")

# Menampilkan plot di Streamlit
st.pyplot(fig)

# modelling k-means
inertia = []
k_values = range(2, 10)
for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data_PCA)
    inertia.append(kmeans.inertia_)

# Membuat plot Elbow
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)

# Menampilkan plot di Streamlit
st.title("Elbow Visualization")
st.pyplot(plt)

data_baru=pd.read_csv('closed_data_fix.csv')
real_data=pd.read_csv('real_data.csv')
kmeans = KMeans(n_clusters=4, random_state=42)
label = kmeans.fit_predict(data_PCA)
data_PCA['cluster'] = label
data_baru['cluster'] = label
real_data['cluster'] = label

# Evaluate using Silhouette Score
silhouette_avg = silhouette_score(data_baru, data_baru['cluster'])
st.title("Silhouette Score")
st.write(silhouette_avg)

fig = plt.figure(figsize=(10,8))
ax = plt.subplot(111, projection='3d', label="bla")
cmap = colors.ListedColormap(["#73F5A7","#5AD6F5", "#6A70F7","#F5584E", "#9F8A78", "#F3AB60"])
ax.scatter(x, y, z, s=40, c=data_PCA["cluster"], marker='o', cmap = cmap )
ax.set_title("The Plot Of The cluster")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")

st.title("PCA 3D Visualization")
st.pyplot(fig)

real_data_with_cluster=pd.read_csv('real_data_cluster.csv')
st.title("Distribusi Cluster")
plt.figure(figsize=(8, 5))
pl = sns.countplot(x=real_data["cluster"], palette='Set3')
pl.set_title("Distribution Of The Cluster")
plt.xlabel("Cluster")
plt.ylabel("Count")

# Menampilkan plot di Streamlit
st.title("Cluster Distribution Visualization")
st.pyplot(plt)

# Kolom kategorikal
cat_cols = ['reassignment_count', 'reopen_count', 'contact_type',
            'impact', 'urgency', 'priority', 'knowledge', 'u_priority_confirmation',
            'notify', 'closed_code', 'made_sla']

# Jumlah kolom yang akan ditampilkan per baris dalam grid
cols_per_row = 2

# Menghitung jumlah baris yang dibutuhkan untuk subplot
rows = (len(cat_cols) + cols_per_row - 1) // cols_per_row

# Membuat figure dan axes untuk subplot
fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 5 * rows))

# Flatten axes untuk memudahkan iterasi
axes = axes.flatten()

# Membuat barplot untuk setiap kolom kategorikal dengan hue cluster
for i, col in enumerate(cat_cols):
    sns.countplot(data=real_data, x=col, hue='cluster', ax=axes[i], palette='Set3')
    axes[i].set_title(f"Barplot of {col} with Cluster")
    axes[i].set_ylabel('Count')
    axes[i].set_xlabel(col)
    axes[i].tick_params(axis='x', rotation=45)  # Memutar label x-axis jika diperlukan

# Menghapus axes kosong jika ada
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Menampilkan plot dengan layout yang lebih rapat
plt.tight_layout()

# Menampilkan plot di Streamlit
st.title("Categorical Barplots with Cluster")
st.pyplot(fig)

closed_data_with_cluster=pd.read_csv('closed_data_cluster.csv')

# Set style untuk seaborn
sns.set(style="whitegrid")

# Membuat jointplot
joint_plot = sns.jointplot(
    data=closed_data_with_cluster,
    x="made_sla",
    y="duration_hours",
    hue='cluster',
    palette="viridis",
    kind="kde",
    height=8
)

# Menampilkan plot di Streamlit
st.title("Joint Plot of Made SLA vs Duration Hours")
st.pyplot(joint_plot.fig)  # Menampilkan plot yang dihasilkan

cluster_means = closed_data_with_cluster.groupby('cluster').mean()

# Mendapatkan nama-nama fitur dan cluster
features = cluster_means.columns.tolist()
clusters = cluster_means.index.tolist()

# Inisialisasi spider plot
num_vars = len(features)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Membuat lingkaran penuh

# Membuat subplot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Membuat plot untuk setiap cluster
for cluster in clusters:
    values = cluster_means.loc[cluster].tolist()
    values += values[:1]  # Menutup lingkaran

    ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Cluster {cluster}')
    ax.fill(angles, values, alpha=0.25)

# Menyesuaikan label dan legenda
ax.set_xticks(angles[:-1])
ax.set_xticklabels(features)
ax.set_title('Spider Plot for Cluster Analysis')
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

# Menampilkan plot di Streamlit
st.title("Spider Plot for Cluster Analysis")
st.pyplot(fig)