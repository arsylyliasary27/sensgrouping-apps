import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import io


# ---------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------
st.sidebar.title("Menu")
menu = st.sidebar.radio(
    "Select analysis",
    ["Heatmap & Dendrogram", "Descriptive Result (PCA)", "Radar Chart (Profil Sensori)"]
)


# ===================================================
# 1. HEATMAP & DENDROGRAM
# ===================================================
if menu == "Heatmap & Dendrogram":

    st.title("Heatmap & Dendrogram (for Grouping Method)")

    st.write("""
    This app helps you analyze sensory grouping results by visualizing 
    sample similarity in heatmap and dendrogram form.
    """)

    st.header("Input Sample & Panelist Information")

    num_samples = st.number_input("Number of samples", min_value=2, value=7)
    sample_names = st.text_area(
        "Sample Names (separate with new line)",
        value="1\n2\n3\n4\n5\n6\n7"
    )

    num_panel = st.number_input("Number of panelists", min_value=1, value=6)
    panel_names = st.text_area(
        "Panelist Names (separate with new line)",
        value="A\nB\nC\nD\nE\nF"
    )

    samples = sample_names.strip().split("\n")
    panels = panel_names.strip().split("\n")

    st.header("Input Grouping Result")

    default_df = pd.DataFrame({
        "Sample": samples,
        **{p: "" for p in panels}
    })

    df = st.data_editor(default_df, width="stretch")

    if st.button("Proceed"):
        df = df.set_index("Sample")

        similarity = pd.DataFrame(0, index=samples, columns=samples)

        for i in samples:
            for j in samples:
                if i != j:
                    similarity.loc[i, j] = sum(df.loc[i] == df.loc[j])
                else:
                    similarity.loc[i, j] = df.shape[1]

        st.subheader("Similarity Matrix")
        st.dataframe(similarity)

        st.subheader("Heatmap")

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(similarity, annot=True, cmap="Blues", fmt="d", ax=ax)
        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)

        st.download_button(
            label="Download heatmap plot (PNG)",
            data=buf,
            file_name="heatmap.png",
            mime="image/png"
        )

        st.subheader("Dendrogram")

        max_sim = df.shape[1]
        distance = max_sim - similarity
        np.fill_diagonal(distance.values, 0)

        linked = linkage(squareform(distance), method="average")

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        dendrogram(linked, labels=similarity.index.tolist(), ax=ax2)
        st.pyplot(fig2)

        buf2 = io.BytesIO()
        fig2.savefig(buf2, format="png", dpi=300, bbox_inches="tight")
        buf2.seek(0)

        st.download_button(
            label="Download dendrogram plot (PNG)",
            data=buf2,
            file_name="dendrogram.png",
            mime="image/png"
        )



# ===================================================
# 2. DESCRIPTIVE RESULT (PCA)
# ===================================================
elif menu == "Descriptive Result (PCA)":

    st.title("Sensory Descriptive Analysis (PCA)")

    st.header("Input Sample & Panelist Information")

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

    sample_names = sample_names.strip().split("\n")
    panelist_names = panelist_names.strip().split("\n")
    params = parameter_list.strip().split("\n")

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

    st.header("Panelist Comments Input")

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
        col_config[p] = st.column_config.SelectboxColumn(options=label_options)

    edited_data = st.data_editor(
        data_input,
        width="stretch",
        column_config=col_config
    )

    if st.button("Run PCA Analysis"):
        numeric = edited_data.copy()
        for col in params:
            numeric[col] = numeric[col].apply(convert_label_to_score)

        mean_by_sample = numeric.groupby("Sample")[params].mean()

        st.subheader("Average Score Per Sample")
        st.dataframe(mean_by_sample)

        scaler = StandardScaler()
        scaled = scaler.fit_transform(mean_by_sample)

        pca = PCA(n_components=2)
        pcs = pca.fit_transform(scaled)

        pc_df = pd.DataFrame(pcs, columns=["PC1", "PC2"], index=mean_by_sample.index)

        st.subheader("PCA Score Plot")
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(pc_df["PC1"], pc_df["PC2"])
        for s in pc_df.index:
            ax.text(pc_df.loc[s, "PC1"], pc_df.loc[s, "PC2"], s)

        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)

        st.download_button(
            label="Download PCA plot (PNG)",
            data=buf,
            file_name="pca_plot.png",
            mime="image/png"
        )

        loadings = pd.DataFrame(pca.components_.T, index=params, columns=["PC1", "PC2"])

        st.subheader("PCA Loadings")
        st.dataframe(loadings)



# ===================================================
# 3. RADAR CHART (UNTRAINED & TRAINED PANELISTS)
# ===================================================
elif menu == "Radar Chart (Profil Sensori)":

    st.title("Radar Chart – Sensory Profile Visualization")

    mode = st.radio("Select Panelist Mode", ["Untrained Panelist", "Trained Panelist"])

    st.header("Input Sample & Parameter Information")

    sample_names = st.text_area(
        "Sample Names (one per line)",
        value="Sample1\nSample2\nSample3"
    )
    parameter_list = st.text_area(
        "Sensory Parameters (one per line)",
        value="Sweet\nBitter\nCreamy\nRancid\nEggy"
    )

    samples = sample_names.strip().split("\n")
    params = parameter_list.strip().split("\n")

    # ===========================================
    # UNTRAINED PANELIST MODE
    # ===========================================
   # UNTRAINED PANELIST MODE
if mode == "Untrained Panelist":
    st.subheader("Input: Parameter mentioned by each panelist")

    num_panel = st.number_input("Number of panelists", min_value=1, value=5)

    # Buat rows untuk setiap panelist × sample
    rows = []
    for i in range(num_panel):
        for sm in samples:  # pastikan setiap sample muncul
            row = {"Panelist": f"P{i+1}", "Sample": sm}
            for p in params:
                row[p] = False
            rows.append(row)

    df = pd.DataFrame(rows)

    col_cfg = {
        "Panelist": st.column_config.TextColumn(disabled=True),
        "Sample": st.column_config.TextColumn(disabled=True)
    }
    for p in params:
        col_cfg[p] = st.column_config.CheckboxColumn()

    edited = st.data_editor(df, width="stretch", column_config=col_cfg)

    if st.button("Generate Radar Chart"):
        # Hitung jumlah cek per parameter untuk setiap sample
        scores = edited.groupby("Sample")[params].sum()

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(params), endpoint=False)

        for sm in scores.index:
            values = np.concatenate((scores.loc[sm].values, [scores.loc[sm].values[0]]))
            angles_loop = np.concatenate((angles, [angles[0]]))
            ax.plot(angles_loop, values, label=sm)
            ax.fill(angles_loop, values, alpha=0.1)

        ax.set_xticks(angles)
        ax.set_xticklabels(params)
        ax.set_title("Untrained Panelist – Sensory Profile")
        ax.legend()

        st.pyplot(fig)


        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)

        st.download_button(
                label="Download Radar Chart (PNG)",
                data=buf,
                file_name="radar_untrained.png",
                mime="image/png"
            )

    # ===========================================
    # TRAINED PANELIST MODE
    # ===========================================
   # TRAINED PANELIST MODE
else:
    st.subheader("Input: Scoring 0–5 for each Panelist per Parameter")

    num_panel = st.number_input("Number of panelists", min_value=1, value=5)

    # Buat rows untuk setiap panelist × sample
    rows = []
    for i in range(num_panel):
        for sm in samples:  # pastikan setiap sample muncul
            row = {"Panelist": f"P{i+1}", "Sample": sm}
            for p in params:
                row[p] = 0
            rows.append(row)

    df = pd.DataFrame(rows)

    col_cfg = {
        "Panelist": st.column_config.TextColumn(disabled=True),
        "Sample": st.column_config.TextColumn(disabled=True)
    }
    for p in params:
        col_cfg[p] = st.column_config.NumberColumn(min_value=0, max_value=5, step=1)

    edited = st.data_editor(df, width="stretch", column_config=col_cfg)

    if st.button("Generate Radar Chart"):
        # Hitung rata-rata per sample, bukan keseluruhan panelist
        scores = edited.groupby("Sample")[params].mean()

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(params), endpoint=False)
        
        for sm in scores.index:
            values = np.concatenate((scores.loc[sm].values, [scores.loc[sm].values[0]]))
            angles_loop = np.concatenate((angles, [angles[0]]))
            ax.plot(angles_loop, values, label=sm)
            ax.fill(angles_loop, values, alpha=0.1)
        
        ax.set_xticks(angles)
        ax.set_xticklabels(params)
        ax.set_title("Trained Panelist – Average Sensory Profile (0–5)")
        ax.legend()
        
        st.pyplot(fig)


        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)

        st.download_button(
                label="Download Radar Chart (PNG)",
                data=buf,
                file_name="radar_trained.png",
                mime="image/png"
            )

