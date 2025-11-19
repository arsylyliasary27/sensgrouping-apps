import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np

st.title("Sensory Evaluation Result Analyzer (Grouping Method)")

st.write("""
This app helps you to analyze sensory grouping results by giving views of the sample characteristics closeness in the form of heatmap and dendrogram .
""")

# --- Input sampel ---
st.header("Input Sample & Panelist Information")

num_samples = st.number_input("Number of samples", min_value=2, value=7)
sample_names = st.text_area(
    "Sample Names (separate with a new line, use coded names)",
    value="1\2\3\4\5\6\7"
)

num_panel = st.number_input("Number of panelists", min_value=1, value=6)
panel_names = st.text_area(
    "Panelist Names (separate with a new line)",
    value="A\B\C\D\E\F"
)

# Convert to lists
samples = sample_names.strip().split("\n")
panels = panel_names.strip().split("\n")

# --- Data input table ---
st.header("Input Grouping Result")

default_df = pd.DataFrame({
    "Sample": samples,
    **{p: "" for p in panels}
})

df = st.data_editor(default_df, num_rows="dynamic")

if st.button("Proceed"):
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
