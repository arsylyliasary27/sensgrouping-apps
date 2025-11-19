import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np

st.title("Sample Similarity Analyzer")

st.write("""
Aplikasi ini menghitung kemiripan antar sampel berdasarkan pengelompokan panelis,
lalu menampilkan heatmap dan dendrogram.
""")

# --- Input sampel ---
st.header("1. Input Sample & Panelist Info")

num_samples = st.number_input("Jumlah sampel", min_value=2, value=7)
sample_names = st.text_area(
    "Nama sampel (pisahkan dengan baris baru)",
    value="SP3\nSP6\nSP9\nSP12\nR25\nR27\nR07"
)

num_panel = st.number_input("Jumlah panelis", min_value=1, value=6)
panel_names = st.text_area(
    "Label panelis (pisahkan baris baru)",
    value="PanA\nPanB\nPanC\nPanD\nPanE\nPanF"
)

# Convert to lists
samples = sample_names.strip().split("\n")
panels = panel_names.strip().split("\n")

# --- Data input table ---
st.header("2. Input Tabel Pengelompokan Panelis")

default_df = pd.DataFrame({
    "Sample": samples,
    **{p: "" for p in panels}
})

df = st.data_editor(default_df, num_rows="dynamic")

if st.button("Proses Data"):
    df = df.set_index("Sample")

    # ===== Similarity Matrix =====
    similarity = pd.DataFrame(0, index=samples, columns=samples)

    for i in samples:
        for j in samples:
            if i != j:
                similarity.loc[i, j] = sum(df.loc[i] == df.loc[j])
            else:
                similarity.loc[i, j] = df.shape[1]

    st.subheader("Similarity Matrix")
    st.dataframe(similarity)

    # ===== Heatmap =====
    st.subheader("Heatmap")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(similarity, annot=True, cmap="Blues", fmt="d", ax=ax)
    st.pyplot(fig)

    # ===== Dendrogram =====
    st.subheader("Dendrogram")

    max_similarity = df.shape[1]
    distance = max_similarity - similarity
    np.fill_diagonal(distance.values, 0)

    linked = linkage(squareform(distance), method="average")

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    dendrogram(linked, labels=similarity.index.tolist(), ax=ax2)
    st.pyplot(fig2)
