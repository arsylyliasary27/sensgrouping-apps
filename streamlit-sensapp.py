import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import io

# ===================================================
# Sidebar Navigation
# ===================================================
st.sidebar.title("Menu")
menu = st.sidebar.radio(
    "Select analysis",
    ["5-Scale Evaluation", "Grouping Evaluation", "QDA-PCA Evaluation", "QDA-Radar Chart Evaluation"]
)

# ===================================================
# 0. 5-SCALE EVALUATION
# ===================================================
if menu == "5-Scale Evaluation":
    st.title("5-Scale Evaluation")
    
    # Input jumlah sample dan panelist
    num_samples = st.number_input("Number of sample:", min_value=1, value=3)
    num_panelists = st.number_input("Number of panelist:", min_value=1, value=5)
    sample_names = [st.text_input(f"Sample name {i+1}", f"Sample-{i+1}") for i in range(num_samples)]

    # Input tabel panelist
    st.subheader("Input Score Result")
    st.markdown("""
    Scoring guidance<br>
    5 = Identical to R<br>
    4 = Slightly different, but still OK<br>
    3 = Doubtful<br>
    2 = Different, propose to reject<br>
    1 = Clearly different, reject
    """, unsafe_allow_html=True)

    panelist_cols = [f"P{i+1}" for i in range(num_panelists)]
    default_data = pd.DataFrame(index=sample_names, columns=panelist_cols)

    col_config = {}
    for col in panelist_cols:
        col_config[col] = st.column_config.SelectboxColumn(
            options=[1,2,3,4,5],
            width="medium"
        )

    edited_data = st.data_editor(
        default_data,
        width="stretch",
        use_container_width=True,
        column_config=col_config
    )

    # Average dan Pass/Reject
    edited_data_numeric = edited_data.astype(float).fillna(0)
    edited_data_numeric["Average"] = edited_data_numeric.mean(axis=1)
    edited_data_numeric["Result"] = edited_data_numeric["Average"].apply(lambda x: "Pass" if x >= 3.5 else "Reject")
    st.subheader("Score Recapitulation")
    st.dataframe(edited_data_numeric)
    st.write("Based on ISO 22935-3, the result will be considered as within spec if score ≥ 3.5")

    # Download Excel
    def to_excel(df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Ratings")
        return output.getvalue()

    st.download_button(
        label="Download recap (.xlsx)",
        data=to_excel(edited_data_numeric),
        file_name="ratings.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Heatmap
    st.subheader("Heatmap of Scoring Frequency")
    heatmap_df = pd.DataFrame(0, index=sample_names, columns=[1,2,3,4,5])
    for samp in sample_names:
        for val in edited_data_numeric.loc[samp, panelist_cols]:
            if val in [1,2,3,4,5]:
                heatmap_df.loc[samp, val] += 1

    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(heatmap_df, annot=True, fmt="d", cmap="Blues", vmin=0, vmax=num_panelists, ax=ax)
    ax.set_xlabel("Score")
    ax.set_ylabel("Sample")
    st.pyplot(fig)

    buf_heatmap = io.BytesIO()
    fig.savefig(buf_heatmap, format="png", dpi=300, bbox_inches="tight")
    buf_heatmap.seek(0)
    st.download_button(
        label="Download heatmap (.png)",
        data=buf_heatmap,
        file_name="heatmap.png",
        mime="image/png"
    )

    # Dendrogram
    st.subheader("Dendrogram of Samples Closeness to Reference")
    dendro_data = edited_data_numeric[panelist_cols].copy().fillna(0)
    dendro_data.loc["Sample R"] = [5]*num_panelists  # Sample R reference
    avg_values = dendro_data.mean(axis=1).values.reshape(-1,1)
    dist_matrix = pdist(avg_values, metric="euclidean")
    linked = linkage(dist_matrix, method='ward')

    fig2, ax2 = plt.subplots(figsize=(8,5))
    dendrogram(linked, labels=dendro_data.index.tolist(), orientation='top', ax=ax2)
    ax2.set_ylabel("Euclidean Distance")
    st.pyplot(fig2)

    buf_dendro = io.BytesIO()
    fig2.savefig(buf_dendro, format="png", dpi=300, bbox_inches="tight")
    buf_dendro.seek(0)
    st.download_button(
        label="Download dendrogram (.png)",
        data=buf_dendro,
        file_name="dendrogram.png",
        mime="image/png"
    )

# ===================================================
# 1. HEATMAP & DENDROGRAM (Grouping Method)
# ===================================================
elif menu == "Grouping Evaluation":
    st.title("Grouping Evaluation (Heatmap & dendrogram)")
    st.write("This app helps analyze sensory grouping results.")

    num_samples = st.number_input("Number of samples", min_value=2, value=7, key="grp_num_samples")
    sample_names = st.text_area(
        "Sample Names (one per line)",
        value="\n".join([f"{i+1}" for i in range(num_samples)]),
        key="grp_sample_names"
    )
    num_panel = st.number_input("Number of panelists", min_value=1, value=6, key="grp_num_panel")
    panel_names = st.text_area(
        "Panelist Names (one per line)",
        value="\n".join([chr(65+i) for i in range(num_panel)]),
        key="grp_panel_names"
    )

    samples = sample_names.strip().split("\n")
    panels = panel_names.strip().split("\n")

    default_df = pd.DataFrame({"Sample": samples, **{p: "" for p in panels}})
    df = st.data_editor(default_df, width="stretch")

    if st.button("Proceed Grouping"):
        df = df.set_index("Sample")
        similarity = pd.DataFrame(0, index=samples, columns=samples)
        for i in samples:
            for j in samples:
                if i != j:
                    similarity.loc[i,j] = sum(df.loc[i]==df.loc[j])
                else:
                    similarity.loc[i,j] = df.shape[1]

        st.subheader("Similarity Matrix")
        st.dataframe(similarity)

        # Heatmap
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(similarity, annot=True, cmap="Blues", fmt="d", ax=ax)
        st.pyplot(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        st.download_button("Download heatmap plot (.png)", data=buf, file_name="heatmap.png", mime="image/png")

        # Dendrogram
        max_sim = df.shape[1]
        distance = max_sim - similarity
        np.fill_diagonal(distance.values, 0)
        linked = linkage(squareform(distance), method="average")
        fig2, ax2 = plt.subplots(figsize=(10,5))
        dendrogram(linked, labels=similarity.index.tolist(), ax=ax2)
        st.pyplot(fig2)
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format="png", dpi=300, bbox_inches="tight")
        buf2.seek(0)
        st.download_button("Download dendrogram plot (.png)", data=buf2, file_name="dendrogram.png", mime="image/png")

# ===================================================
# 2. DESCRIPTIVE RESULT (PCA) - MODIFIED
# ===================================================
elif menu == "QDA-PCA Evaluation":
    st.title("PCA Analysis for Sensory Evaluation")
    
    # Input sample, panelist, parameter
    sample_names = st.text_area("Sample Names (one per line)", value="Sample1\nSample2\nSample3\nSample4\nSample5")
    panelist_names = st.text_area("Panelist Names (one per line)", value="Panel1\nPanel2\nPanel3\nPanel4\nPanel5")
    parameter_list = st.text_area("Sensory Parameters (one per line)", value="Colour\nVanilla\nSweet\nBitter\nShininess\nChocolate")

    sample_names = sample_names.strip().split("\n")
    panelist_names = panelist_names.strip().split("\n")
    params = parameter_list.strip().split("\n")

    # Create data editor for panelist scoring
    rows = []
    for pl in panelist_names:
        for sm in sample_names:
            row = {"Panelist": pl, "Sample": sm}
            for p in params:
                row[p] = 0
            rows.append(row)

    col_config = {"Panelist": st.column_config.TextColumn(disabled=True),
                  "Sample": st.column_config.TextColumn(disabled=True)}
    for p in params:
        col_config[p] = st.column_config.SelectboxColumn(options=[0,1,2,3,4,5])

    edited_data = st.data_editor(pd.DataFrame(rows), width="stretch", column_config=col_config)

    if st.button("Run PCA"):
        # Convert to numeric
        numeric = edited_data.copy()
        for col in params:
            numeric[col] = pd.to_numeric(numeric[col])

        # Average score per sample
        mean_by_sample = numeric.groupby("Sample")[params].mean()
        st.subheader("Average Score Per Sample")
        st.dataframe(mean_by_sample)

        # Standardize and PCA
        scaler = StandardScaler()
        scaled = scaler.fit_transform(mean_by_sample)
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(scaled)
        pc_df = pd.DataFrame(pcs, columns=["PC1","PC2"], index=mean_by_sample.index)

        # PCA biplot with SPSS/XLSTAT-style scaling
        fig, ax = plt.subplots(figsize=(8,6))

        # Plot samples
        ax.scatter(pc_df["PC1"], pc_df["PC2"], color='red', s=50)
        for s in pc_df.index:
            ax.text(pc_df.loc[s,"PC1"], pc_df.loc[s,"PC2"], s, color='red', fontsize=10, fontweight='bold')

        # Scale loadings like SPSS/XLSTAT (correlation circle)
        loadings = pca.components_.T
        scale_factor = np.max(np.abs(pcs)) * 0.7
        for i, param in enumerate(params):
            ax.arrow(0, 0, loadings[i,0]*scale_factor, loadings[i,1]*scale_factor, 
                     color='blue', head_width=0.1, length_includes_head=True)
            ax.text(loadings[i,0]*scale_factor*1.1, loadings[i,1]*scale_factor*1.1, 
                    param, color='blue', fontsize=10, fontweight='bold')

        # Axes labels with explained variance
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.axhline(0, color='grey', linestyle='--')
        ax.axvline(0, color='grey', linestyle='--')
        ax.set_title("PCA Biplot")
        ax.grid(True)
        ax.set_aspect('equal')

        st.pyplot(fig)

        # Download plot
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        st.download_button("Download PCA (.png)", data=buf, file_name="pca_biplot.png", mime="image/png")

# ===================================================
# 3. RADAR CHART
# ===================================================
elif menu == "QDA-Radar Chart Evaluation":
    st.title("Radar Chart Visualization for Sensory Profiling")
    mode = st.radio("Select Panelist Mode", ["CATA","RATA"])

    sample_names = st.text_area("Sample Names (one per line)", value="Sample1\nSample2\nSample3")
    parameter_list = st.text_area("Sensory Parameters (one per line)", value="Sweet\nBitter\nCreamy\nRancid\nEggy")
    samples = sample_names.strip().split("\n")
    params = parameter_list.strip().split("\n")

    num_panel = st.number_input("Number of panelists", min_value=1, value=5)

    if mode=="CATA":
        st.subheader("Input: Parameter mentioned by each panelist")
        rows=[]
        for i in range(num_panel):
            for sm in samples:
                row={"Panelist":f"P{i+1}","Sample":sm}
                for p in params: row[p]=False
                rows.append(row)
        df=pd.DataFrame(rows)
        col_cfg={"Panelist":st.column_config.TextColumn(disabled=True),"Sample":st.column_config.TextColumn(disabled=True)}
        for p in params: col_cfg[p]=st.column_config.CheckboxColumn()
        edited=st.data_editor(df,width="stretch",column_config=col_cfg)

        if st.button("Generate Radar Chart (CATA)"):
            scores=edited.groupby("Sample")[params].sum()
            fig, ax=plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
            angles=np.linspace(0,2*np.pi,len(params),endpoint=False)
            for sm in samples:
                values = scores.loc[sm].values
                values = np.append(values, values[0])
                ax.plot(np.append(angles, angles[0]), values, label=sm)
            ax.set_xticks(angles)
            ax.set_xticklabels(params)
            ax.set_ylim(0, num_panel)
            ax.legend()
            st.pyplot(fig)
            buf=io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            buf.seek(0)
            st.download_button("Download radar chart (.png)", data=buf, file_name="radar_chart_cata.png", mime="image/png")

    else:
        st.subheader("Input: Parameter scores per panelist (0-5)")
        rows=[]
        for i in range(num_panel):
            for sm in samples:
                row={"Panelist":f"P{i+1}","Sample":sm}
                for p in params: row[p]=0
                rows.append(row)
        df=pd.DataFrame(rows)
        col_cfg={"Panelist":st.column_config.TextColumn(disabled=True),"Sample":st.column_config.TextColumn(disabled=True)}
        for p in params: col_cfg[p]=st.column_config.SelectboxColumn(options=[0,1,2,3,4,5])
        edited=st.data_editor(df,width="stretch",column_config=col_cfg)

        if st.button("Generate Radar Chart (RATA)"):
            scores = edited.groupby("Sample")[params].mean()
            fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
            angles = np.linspace(0, 2*np.pi, len(params), endpoint=False)
            for sm in samples:
                values = scores.loc[sm].values
                values = np.append(values, values[0])
                ax.plot(np.append(angles, angles[0]), values, label=sm)
            ax.set_xticks(angles)
            ax.set_xticklabels(params)
            ax.set_ylim(0,5)
            ax.legend()
            st.pyplot(fig)
            buf=io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            buf.seek(0)
            st.download_button("Download radar chart (.png)", data=buf, file_name="radar_chart_RATA.png", mime="image/png")
