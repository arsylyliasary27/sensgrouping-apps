import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np
from sklearn.decomposition import PCA

# Disable warnings from pyplot
st.set_option('deprecation.showPyplotGlobalUse', False)


# ======================================
# Fungsi Konversi Sensory Label -> Skor
# ======================================
def convert_label_to_score(label: str):
    mapping = {
        "more": 1,
        "slightly more": 0.5,
        "comparable": 0,
        "slightly less": -0.5,
        "less": -1
    }
    if isinstance(label, str):
        label = label.strip().lower()
    return mapping.get(label, np.nan)

label_options = ["more", "slightly more", "comparable", "slightly less", "less"]


# ======================================
# Sidebar Menu
# ======================================
st.sidebar.title("Sensory Toolkit")
menu = st.sidebar.radio(
    "Select Feature",
    ["Heatmap & Dendrogram", "Descriptive Result (PCA)"]
)


# =====================================================
# ---------------  MENU 1: HEATMAP ---------------------
# =====================================================
if menu == "Heatmap & Dendrogram":

    st.title("Sensory Evaluation Result Analyzer (Grouping Method)")

    st.write("""
    This app helps you analyze sensory grouping results 
    by showing similarity matrices, heatmaps, and dendrograms.
    """)

    # Input sample-panel structure
    st.header("Input Sample & Panelist Information")

    num_samples = st.number_input("Number of samples", min_value=2, value=7)
    sample_names = st.text_area(
        "Sample Names (one per line, coded names recommended)",
        value="1\n2\n3\n4\n5\n6\n7"
    )

    num_panel = st.number_input("Number of panelists", min_value=1, value=6)
    panel_names = st.text_area(
        "Panelist Names (one per line)",
        value="A\nB\nC\nD\nE\nF"
    )

    samples = sample_names.strip().split("\n")
    panels = panel_names.strip().split("\n")

    # Input grouping data table
    st.header("Input Grouping Result")

    default_df = pd.DataFrame({
        "Sample": samples,
        **{p: "" for p in panels}
    })

    df = st.data_editor(default_df, num_rows="dynamic")

    if st.button("Proceed"):
        df = df.set_index("Sample")

        # Similarity Matrix
        similarity = pd.DataFrame(0, index=samples, columns=samples)

        for i in samples:
            for j in samples:
                if i != j:
                    similarity.loc[i, j] = sum(df.loc[i] == df.loc[j])
                else:
                    similarity.loc[i, j] = df.shape[1]

        st.subheader("Similarity Matrix")
        st.dataframe(similarity)

        # Heatmap
        st.subheader("Heatmap")

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(similarity, annot=True, cmap="Blues", fmt="d", ax=ax)
        st.pyplot(fig)

        # Dendrogram
        st.subheader("Dendrogram")

        max_similarity = df.shape[1]
        distance = max_similarity - similarity
        np.fill_diagonal(distance.values, 0)

        linked = linkage(squareform(distance), method="average")

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        dendrogram(linked, labels=similarity.index.tolist(), ax=ax2)
        st.pyplot(fig2)


# =====================================================
# ----------------  MENU 2: PCA RESULT  ----------------
# =====================================================
elif menu == "Descriptive Result (PCA)":

    st.title("Descriptive Sensory Result Analyzer (PCA)")

    # ============================
    # Konfigurasi Input
    # ============================
    st.header("1. Configuration")

    n_samples = st.number_input("Number of Samples", min_value=1, value=5)
    sample_names = []
    for i in range(n_samples):
        sample_names.append(st.text_input(f"Sample {i+1}", value=f"S{i+1}"))

    n_panelists = st.number_input("Number of Panelists", min_value=1, value=7)
    panelist_names = []
    for i in range(n_panelists):
        panelist_names.append(st.text_input(f"Panelist {i+1}", value=f"P{i+1}"))

    st.header("2. Sensory Parameters")
    n_params = st.number_input("Number of Parameters", min_value=1, value=8)
    params = []
    for i in range(n_params):
        params.append(st.text_input(f"Parameter {i+1}", value=f"Param{i+1}"))

    # ============================
    # Input Data Sensoris
    # ============================
    st.header("3. Panelist Comments Input")

    data_input = pd.DataFrame(
        "",
        index=pd.MultiIndex.from_product([panelist_names, sample_names], names=["Panelist", "Sample"]),
        columns=params
    )

    edited_data = st.data_editor(
        data_input,
        use_container_width=True,
        column_config={
            col: st.column_config.SelectboxColumn(
                options=label_options,
                required=False
            )
            for col in params
        }
    )

    # ============================
    # PROCESS DATA
    # ============================
    if st.button("Run PCA Analysis"):

        st.header("4. Numerical Conversion (Safe Mapping)")

        numeric_data = edited_data.copy()
        for col in params:
            numeric_data[col] = numeric_data[col].apply(convert_label_to_score)

        st.write("Converted Numerical Data:")
        st.dataframe(numeric_data)

        # Rata-rata per sampel
        st.header("5. Average per Sample")
        mean_by_sample = numeric_data.groupby("Sample").mean()
        st.dataframe(mean_by_sample)

        # PCA
        st.header("6. PCA Analysis")
        X = mean_by_sample.fillna(0).values
        pca = PCA(n_components=2)
        scores = pca.fit_transform(X)
        loadings = pca.components_.T

        st.write(f"PC1 Explained Variance: {pca.explained_variance_ratio_[0]:.2f}")
        st.write(f"PC2 Explained Variance: {pca.explained_variance_ratio_[1]:.2f}")

        # Biplot
        st.subheader("PCA Biplot")

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(scores[:, 0], scores[:, 1])
        for i, name in enumerate(sample_names):
            ax.text(scores[i, 0], scores[i, 1], name)

        for i, param in enumerate(params):
            ax.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                     head_width=0.02, color='red')
            ax.text(loadings[i, 0], loadings[i, 1], param, color='red')

        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        st.pyplot(fig)

        st.success("PCA Analysis Completed!")
