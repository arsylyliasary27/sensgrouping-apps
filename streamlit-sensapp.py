import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ============================
# Sidebar Menu
# ============================

st.sidebar.title("Menu")
menu = st.sidebar.radio(
    "Select analysis",
    ["Heatmap & Dendrogram", "Descriptive Result (PCA)"]
)

# =================================================================
# ====================== 1. HEATMAP & DENDROGRAM ==================
# =================================================================

if menu == "Heatmap & Dendrogram":

    st.title("Sensory Evaluation Result Analyzer (Grouping Method)")

    st.write("""
    This app helps you to analyze sensory grouping results by giving views 
    of the sample characteristics closeness in the form of heatmap and dendrogram.
    """)

    # --- Input sampel ---
    st.header("Input Sample & Panelist Information")

    num_samples = st.number_input("Number of samples", min_value=2, value=7)
    sample_names = st.text_area(
        "Sample Names (separate with new line, use coded names)",
        value="1\n2\n3\n4\n5\n6\n7"
    )

    num_panel = st.number_input("Number of panelists", min_value=1, value=6)
    panel_names = st.text_area(
        "Panelist Names (separate with new line)",
        value="A\nB\nC\nD\nE\nF"
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

    df = st.data_editor(default_df, width="stretch")

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

# =================================================================
# ========================== 2. PCA MODULE =========================
# =================================================================

if menu == "Descriptive Result (PCA)":

    st.title("Sensory Descriptive Analysis (PCA)")

    # ============================
    # Input Sampel & Panelis
    # ============================
    st.header("1. Input Sample & Panelist Information")

    sample_names = st.text_area(
        "Sample Names (one per line)",
        value="Sample1\nSample2\nSample3\nSample4\nSample5"
    )
    panelist_names = st.text_area(
        "Panelist Names (one per line)",
        value="P1\nP2\nP3\nP4\nP5\nP6\nP7"
    )
    parameter_list = st.text_area(
        "Sensory Parameters (one per line)",
        value="Eggy\nRancid\nCowy\nBitter\nSweet\nMilkiness\nCreamy\nStrange"
    )

    #sample_names = sample_names.strip().split("\n")
    #panelist_names = panelist_names.strip().split("\n")
    #params = parameter_list.strip().split("\n")

    #st.write(f"**Samples:** {sample_names}")
    #st.write(f"**Panelists:** {panelist_names}")
    #st.write(f"**Parameters:** {params}")

    # Mapping nilai komentar
    label_options = ["more", "slightly more", "comparable", "slightly less", "less"]
    score_map = {
        "more": 1,
        "slightly more": 0.5,
        "comparable": 0,
        "slightly less": -0.5,
        "less": -1,
    }

    def convert_label_to_score(x):
        if pd.isna(x) or x == "":
            return np.nan
        return score_map.get(x, np.nan)

    # ============================
    # Input Data Sensoris (FIXED)
    # ============================
    st.header("2. Panelist Comments Input")

    rows = []
    for pl in panelist_names:
        for sm in sample_names:
            row = {"Panelist": pl, "Sample": sm}
            for p in params:
                row[p] = ""
            rows.append(row)

    data_input = pd.DataFrame(rows)

    col_config = {
        "Panelist": st.column_config.TextColumn(disabled=True),
        "Sample": st.column_config.TextColumn(disabled=True)
    }

    for p in params:
        col_config[p] = st.column_config.SelectboxColumn(
            options=label_options,
            required=False
        )

    edited_data = st.data_editor(
        data_input,
        width="stretch",
        column_config=col_config
    )

    # ============================
    # PCA Calculation
    # ============================
    if st.button("Run PCA Analysis"):
        
        # Convert to numeric
        numeric_data = edited_data.copy()
        for col in params:
            numeric_data[col] = numeric_data[col].apply(convert_label_to_score)

        # Hitung mean per sample
        mean_by_sample = numeric_data.groupby("Sample")[params].mean()
        st.subheader("Average Score Per Sample")
        st.dataframe(mean_by_sample)

        # Standardize
        scaler = StandardScaler()
        scaled = scaler.fit_transform(mean_by_sample)

        # PCA
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(scaled)

        pc_df = pd.DataFrame(
            pcs,
            columns=["PC1", "PC2"],
            index=mean_by_sample.index
        )

        st.subheader("PCA Score Plot (PC1 vs PC2)")
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(pc_df["PC1"], pc_df["PC2"])

        for sample in mean_by_sample.index:
            ax.text(pc_df.loc[sample, "PC1"], pc_df.loc[sample, "PC2"], sample)

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        st.pyplot(fig)

        # PCA Loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            index=params,
            columns=["PC1", "PC2"]
        )

        st.subheader("PCA Loadings")
        st.dataframe(loadings)
