"""
Streamlit Web Dashboard: SARS-CoV-2 Variant Classifier & Gene Conservation Explorer
Comparative Analysis of Genetic Conservation and Variation in Selected SARS-CoV-2 Genes
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Ensure repository root is in python search path
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from src.ingestion import RawGenomeIngestor

# ------------------------------------------------------------------------------
# 1. Page Configuration & Custom Theme
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="SARS-CoV-2 Variant Classifier",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean layout and badges
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    .badge-alpha {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .badge-beta {
        background-color: #FFEDD5;
        color: #C2410C;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .badge-delta {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .badge-omicron {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .badge-unknown {
        background-color: #F3F4F6;
        color: #4B5563;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. Resource Caching: Load Model & Ingestor Once
# ------------------------------------------------------------------------------
@st.cache_resource
def get_ingestor():
    artifact_path = APP_DIR / "results" / "sars_cov2_variant_classifier.joblib"
    if not artifact_path.exists():
        artifact_path = APP_DIR / "notebooks" / "sars_cov2_variant_classifier.joblib"
    return RawGenomeIngestor(artifact_path=artifact_path)

ingestor = get_ingestor()

# ------------------------------------------------------------------------------
# 3. Sidebar: Adjustable Diagnostic Thresholds & Architecture
# ------------------------------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/82/SARS-CoV-2_without_background.png", width=120)
    st.title("Settings & Diagnostics")
    
    st.markdown("### Diagnostic Thresholds")
    conf_threshold = st.slider(
        "Confidence Gate Cutoff (%)",
        min_value=50, max_value=95, value=70, step=5,
        help="Predictions with maximum probability below this cutoff are classified as Ambiguous or Potential Recombinant."
    ) / 100.0
    
    max_ambig = st.slider(
        "Max Ambiguous Base Tolerance (%)",
        min_value=0.5, max_value=5.0, value=2.0, step=0.5,
        help="Genomes with N and gap proportions exceeding this percentage will fail the pre-prediction QC filter."
    ) / 100.0
    
    min_len = st.number_input(
        "Min Core Genome Length (nt)",
        min_value=25000, max_value=29903, value=28500, step=100,
        help="Assemblies shorter than this threshold (excluding padding dashes) fail the QC length filter."
    )

    st.markdown("---")
    st.markdown("### Model Architecture")
    st.markdown("""
    - **Classifier:** Random Forest (200 depth-constrained trees)
    - **Feature Space:** 320 gene-tagged 3-mers
    - **Target Genes:** S, E, M, N, RdRp
    - **Classes:** Alpha, Beta, Delta, Omicron
    - **Anti-Leakage:** 100% deduplication before split
    """)
    st.markdown("---")
    st.caption("Comparative Analysis of Genetic Conservation in SARS-CoV-2 | Semester Project")

# ------------------------------------------------------------------------------
# 4. Main Banner (Retains the frontend 🧬 icon as requested)
# ------------------------------------------------------------------------------
st.markdown('<div class="main-title">🧬 SARS-CoV-2 Multi-Gene Variant Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Automated whole-genome FASTA ingestion, quality control, gene extraction, and alignment-free variant prediction.</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 5. Main Dashboard Tabs (Clean, Emoji-Free Navigation)
# ------------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "Single Genome Classification",
    "High-Throughput Batch Screening",
    "Gene Conservation & Model Insights"
])

# ==============================================================================
# TAB 1: Single Genome Ingestion & Prediction
# ==============================================================================
with tab1:
    st.subheader("1. Ingest Whole-Genome Consensus Sequence")
    
    input_mode = st.radio(
        "Select Sequence Input Source:",
        ["Choose Pre-loaded Example Genome", "Upload FASTA File (.fasta / .fa / .txt)", "Paste Raw FASTA Text"],
        horizontal=True
    )
    
    raw_fasta_text = None
    example_files = {
        "Delta Isolate (RUS/Primorye_8507/2021)": "delta_sample.fasta",
        "Omicron Isolate (USA/NE-SALIVA_S576105/2022)": "omicron_sample.fasta",
        "Wuhan-Hu-1 Reference (NC_045512.2)": "wuhan_reference.fasta"
    }
    
    if input_mode == "Choose Pre-loaded Example Genome":
        selected_ex = st.selectbox("Select verified genomic sample:", list(example_files.keys()))
        example_path = APP_DIR / "data" / "examples" / example_files[selected_ex]
        if example_path.exists():
            raw_fasta_text = example_path.read_text(encoding="utf-8")
            with st.expander("Preview Selected FASTA Sequence", expanded=False):
                st.code(raw_fasta_text[:800] + "\n... [truncated]", language="text")
        else:
            st.error(f"Sample file not found at {example_path}")
            
    elif input_mode == "Upload FASTA File (.fasta / .fa / .txt)":
        uploaded_file = st.file_uploader("Upload Consensus Genome FASTA", type=["fasta", "fa", "txt", "xz", "gz"])
        if uploaded_file is not None:
            raw_fasta_text = uploaded_file
            
    elif input_mode == "Paste Raw FASTA Text":
        pasted_text = st.text_area(
            "Paste complete FASTA string (including >header):",
            height=200,
            placeholder=">SAMPLE_ISOLATE_01\nNNNNN...ATGTTTGTTTTT...NNNNN"
        )
        if pasted_text.strip():
            raw_fasta_text = pasted_text.strip()
            
    # Action Button
    if st.button("Analyze Genome", type="primary", use_container_width=True):
        if not raw_fasta_text:
            st.warning("Please provide or select a FASTA sequence first.")
        else:
            with st.spinner("Executing QC filter, extracting 5 viral genes, and generating prediction..."):
                results = ingestor.predict_genome(
                    raw_fasta_text,
                    confidence_threshold=conf_threshold,
                    min_length=min_len,
                    max_ambig_frac=max_ambig,
                    allow_failed_qc=True
                )
                
            if not results:
                st.error("Could not parse valid FASTA records from the input.")
            else:
                res = results[0]
                qc = res["qc_report"]
                
                # --- QC Evaluation Banner ---
                st.markdown("### 2. Quality Control & Genome Extraction Report")
                st.caption(f"Applied Diagnostics: Min Length = {min_len:,} nt | Max Ambiguity = {max_ambig:.1%} | Confidence Gate = {conf_threshold:.0%}")
                
                qc_col1, qc_col2, qc_col3, qc_col4 = st.columns(4)
                
                with qc_col1:
                    st.metric("Total Length", f"{qc['total_length']:,} nt")
                with qc_col2:
                    st.metric("Core Length (No ends)", f"{qc['core_length']:,} nt")
                with qc_col3:
                    st.metric("Ambiguous Bases", f"{qc['ambiguous_fraction']:.2%}")
                with qc_col4:
                    if qc["qc_passed"]:
                        st.success("QC PASSED")
                    else:
                        st.error("QC FAILED")
                        
                if qc["issues"]:
                    st.warning("QC Warnings:\n" + "\n".join([f"- {issue}" for issue in qc["issues"]]))
                    
                # Extracted Genes breakdown
                if res["gene_lengths"]:
                    st.markdown("**Extracted Gene Regions (nt):**")
                    gene_cols = st.columns(5)
                    expected_lens = {"S": 3822, "E": 228, "M": 669, "N": 1260, "RdRp": 2796}
                    for i, (g, length) in enumerate(res["gene_lengths"].items()):
                        with gene_cols[i]:
                            exp = expected_lens.get(g, "-")
                            delta_txt = "exact" if length == exp else f"{length-exp:+d} nt"
                            st.metric(f"Gene {g}", f"{length:,} nt", delta=delta_txt, delta_color="normal" if length == exp else "inverse")
                            
                st.markdown("---")
                
                # --- Prediction Outcome ---
                st.markdown("### 3. Machine Learning Prediction Results")
                pred_col1, pred_col2, pred_col3 = st.columns([1.2, 1, 1.8])
                
                pred_var = res["predicted_variant"]
                conf = res["confidence"]
                status = res["status"]
                
                with pred_col1:
                    badge_class = f"badge-{pred_var.lower()}" if pred_var in ["Alpha", "Beta", "Delta", "Omicron"] else "badge-unknown"
                    st.markdown("**Predicted Lineage:**")
                    st.markdown(f'<span class="{badge_class}">{pred_var}</span>', unsafe_allow_html=True)
                    st.caption(f"Accession: `{res['accession'][:32]}`")
                    
                with pred_col2:
                    st.metric("Model Confidence", f"{conf:.1%}")
                    if "CONFIDENT" in status:
                        st.markdown("**Status:** High Confidence")
                    else:
                        st.markdown("**Status:** Ambiguous / Recombinant")
                        
                with pred_col3:
                    st.markdown("**Class Probability Breakdown:**")
                    prob_df = pd.DataFrame({
                        "Variant": list(res["probabilities"].keys()),
                        "Probability": list(res["probabilities"].values())
                    })
                    
                    fig, ax = plt.subplots(figsize=(6, 2.2))
                    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
                    bars = ax.barh(prob_df["Variant"], prob_df["Probability"], color=colors, edgecolor="black", linewidth=0.5)
                    ax.set_xlim(0, 1.05)
                    ax.set_xlabel("Predicted Probability", fontsize=9)
                    for bar in bars:
                        w = bar.get_width()
                        ax.text(w + 0.02, bar.get_y() + bar.get_height()/2.0, f"{w:.1%}", ha="left", va="center", fontsize=8, fontweight="bold")
                    sns.despine(top=True, right=True)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

# ==============================================================================
# TAB 2: High-Throughput Batch FASTA Screening
# ==============================================================================
with tab2:
    st.subheader("Automated Batch Ingestion & Surveillance Screening")
    st.markdown("Upload a multi-FASTA file containing dozens or hundreds of whole genomes to screen variants simultaneously.")
    
    batch_mode = st.radio("Choose Batch Source:", ["Use Pre-loaded Demo Batch (3 Genomes)", "Upload Multi-FASTA File"], horizontal=True)
    
    batch_fasta = None
    if batch_mode == "Use Pre-loaded Demo Batch (3 Genomes)":
        bpath = APP_DIR / "data" / "examples" / "batch_test_genomes.fasta"
        if bpath.exists():
            batch_fasta = bpath
            st.info("Loaded `batch_test_genomes.fasta` (Wuhan-Hu-1 Reference, Delta RUS isolate, Omicron USA isolate).")
    else:
        batch_file = st.file_uploader("Upload Multi-FASTA (.fasta / .fa / .txt)", type=["fasta", "fa", "txt"])
        if batch_file:
            batch_fasta = batch_file
            
    if st.button("Process Batch Run", type="primary", use_container_width=True):
        if not batch_fasta:
            st.warning("Please upload or select a batch FASTA file.")
        else:
            with st.spinner("Processing batch genomes through QC and Random Forest..."):
                results = ingestor.predict_genome(
                    batch_fasta,
                    confidence_threshold=conf_threshold,
                    min_length=min_len,
                    max_ambig_frac=max_ambig,
                    allow_failed_qc=True
                )
                
            if results:
                st.success(f"Successfully processed {len(results)} genomes!")
                st.caption(f"Applied Diagnostics: Min Length = {min_len:,} nt | Max Ambiguity = {max_ambig:.1%} | Confidence Gate = {conf_threshold:.0%}")
                
                # Format into table
                table_rows = []
                for r in results:
                    qc = r["qc_report"]
                    table_rows.append({
                        "Accession / Strain": r["accession"][:40],
                        "QC Status": "PASSED" if qc["qc_passed"] else "FAILED",
                        "Length (nt)": qc["core_length"],
                        "Ambig (%)": f"{qc['ambiguous_fraction']:.2%}",
                        "Predicted Variant": r["predicted_variant"],
                        "Confidence": f"{r['confidence']:.1%}",
                        "Status": r["status"]
                    })
                df_batch = pd.DataFrame(table_rows)
                
                # Summary metrics
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                with mcol1:
                    st.metric("Total Genomes", len(df_batch))
                with mcol2:
                    st.metric("QC Passed", sum(df_batch["QC Status"] == "PASSED"))
                with mcol3:
                    st.metric("QC Failed", sum(df_batch["QC Status"] == "FAILED"))
                with mcol4:
                    confident_cnt = sum(df_batch["Status"] == "CONFIDENT")
                    st.metric("Confident Predictions", f"{confident_cnt}/{len(df_batch)}")
                    
                st.dataframe(df_batch, use_container_width=True)
                
                # Download CSV
                csv_bytes = df_batch.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Batch Surveillance Report (CSV)",
                    data=csv_bytes,
                    file_name="sars_cov2_batch_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# ==============================================================================
# TAB 3: Gene Conservation & Research Insights
# ==============================================================================
with tab3:
    st.subheader("Biological Insights & Hypothesis Verification")
    st.markdown("""
    **Core Research Hypothesis:**  
    *Proteins coding for essential replication and virion assembly (RdRp, M, E) are highly conserved under strong purifying selection, 
    whereas surface antigens under host immune pressure (Spike and Nucleocapsid) display high variation.*
    """)
    
    fig_col1, fig_col2 = st.columns(2)
    fig_dir = APP_DIR / "results" / "figures"
    
    with fig_col1:
        st.markdown("#### 1. Variant Separability in 2D PCA Space")
        pca_path = fig_dir / "fig8a_ml_pca_variants.png"
        if pca_path.exists():
            st.image(str(pca_path), use_container_width=True)
        else:
            st.info("PCA figure generated during model training.")
            
    with fig_col2:
        st.markdown("#### 2. Holdout Test Set Confusion Matrices")
        cm_path = fig_dir / "fig8b_ml_confusion_matrices.png"
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
        else:
            st.info("Confusion matrix figure generated during model training.")
            
    st.markdown("---")
    st.markdown("#### 3. Random Forest Feature Importance: Proving the Conservation Hypothesis")
    imp_path = fig_dir / "fig8c_ml_feature_importance.png"
    if imp_path.exists():
        st.image(str(imp_path), use_container_width=True)
        
    st.markdown("""
    | Gene Region | Length (nt) | Physiological Function | Total Gini Importance | Evolutionary Status |
    |---|---|---|---|---|
    | **Spike (`S`)** | 3,822 | ACE2 host entry receptor, primary neutralizing antibody target | **53.3%** | **Adaptive / High Diversification** |
    | **Nucleocapsid (`N`)** | 1,260 | Viral RNA packaging, subgenomic transcription efficiency | **32.6%** | **Adaptive / Moderately Diversified** |
    | **Membrane (`M`)** | 669 | Essential virion scaffold and envelope assembly | **8.1%** | **Conserved (Purifying Selection)** |
    | **RdRp (`nsp12`)** | 2,796 | Core catalytic replication-transcription polymerase | **5.9%** | **Highly Conserved (Catalytic Constraints)** |
    | **Envelope (`E`)** | 228 | Viroporin channel, budding, viral pathogenesis | **0.2%** | **Extremely Conserved** |
    """)
    
    st.success("""
    **Biological Conclusion:** Surface antigens (S + N = 85.9%) provide over five times more variant-defining power 
    than essential replication machinery (RdRp + M + E = 14.1%), confirming that sequence variation is heavily constrained in structural and replication genes.
    """)
