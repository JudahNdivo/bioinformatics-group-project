"""
src/ingestion.py - Automated Raw Whole-Genome FASTA Ingestion Pipeline
for SARS-CoV-2 Variant Classification & Quality Control.
"""

from collections import Counter
import gzip
import io
import itertools
import lzma
from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd


class RawGenomeIngestor:
  """Automates end-to-end ingestion, quality control, gene extraction,

  and variant prediction from raw full-length SARS-CoV-2 FASTA genomes.
  """

  # Wuhan-Hu-1 (NC_045512.2) 0-based coordinate definitions
  COORDINATES = {
      "S": (21562, 25384),  # 3,822 nt
      "E": (26244, 26472),  # 228 nt
      "M": (26522, 27191),  # 669 nt
      "N": (28273, 29533),  # 1,260 nt
      "RdRp_p1": (13441, 13468),  # nsp12 pre-slip segment (27 nt)
      "RdRp_p2": (13467, 16236),  # nsp12 post-slip segment (2,769 nt)
  }

  # Ultra-conserved 20-nt flanking anchors for trimmed/unaligned sequences
  ANCHORS = {
      "S_start": "ATGTTTGTTTTTCTTGTTTT",
      "E_start": "ATGTACTCATTCGTTTCGGA",
      "M_start": "ATGGCAGATTCCAACGGTACT",
      "N_start": "ATGTCTGATAATGGACCCCA",
      "RdRp_start": "TCAGCTGATGCACAATCGTT",
  }

  def __init__(self, artifact_path=None):
    """Initializes the ingestor and loads the serialized model artifact."""
    if artifact_path is None:
      # Search default paths
      candidates = [
          Path(
              "Semester"
              " Project/bioinformatics-group-project/results/sars_cov2_variant_classifier.joblib"
          ),
          Path("results/sars_cov2_variant_classifier.joblib"),
          Path("../results/sars_cov2_variant_classifier.joblib"),
          Path(
              "/content/bioinformatics-group-project/results/sars_cov2_variant_classifier.joblib"
          ),
      ]
      for c in candidates:
        if c.exists():
          artifact_path = c
          break

    if artifact_path and Path(artifact_path).exists():
      self.artifact = joblib.load(artifact_path)
      self.model = self.artifact["model"]
      self.feature_columns = self.artifact["feature_columns"]
      self.genes = self.artifact["genes"]
      self.k = self.artifact["k"]
      self.classes = self.artifact["classes"]
    else:
      self.artifact = None
      self.model = None
      self.feature_columns = None
      self.genes = ["S", "E", "M", "N", "RdRp"]
      self.k = 3
      self.classes = ["Alpha", "Beta", "Delta", "Omicron"]

  def parse_fasta(self, fasta_input):
    """Parses a FASTA string, file path (.fasta, .gz, .xz), or file-like object.

    Returns a list of tuples: [(header, sequence), ...]
    """
    records = []
    current_header = None
    current_seq = []

    # Handle file path
    if isinstance(fasta_input, (str, Path)) and Path(fasta_input).is_file():
      p = Path(fasta_input)
      if p.suffix == ".xz":
        f = lzma.open(p, "rt", encoding="utf-8")
      elif p.suffix == ".gz":
        f = gzip.open(p, "rt", encoding="utf-8")
      else:
        f = open(p, "r", encoding="utf-8")
      lines = f
      should_close = True
    elif hasattr(fasta_input, "read"):
      # File-like object (e.g. from Streamlit or requests)
      content = fasta_input.read()
      if isinstance(content, bytes):
        content = content.decode("utf-8")
      lines = content.splitlines()
      should_close = False
    elif isinstance(fasta_input, str):
      lines = fasta_input.strip().splitlines()
      should_close = False
    else:
      raise ValueError("Invalid FASTA input type.")

    try:
      for line in lines:
        line = line.strip()
        if not line:
          continue
        if line.startswith(">"):
          if current_header:
            records.append((current_header, "".join(current_seq)))
          current_header = line[1:].strip()
          current_seq = []
        else:
          current_seq.append(line)
      if current_header:
        records.append((current_header, "".join(current_seq)))
    finally:
      if should_close:
        lines.close()

    return records

  def check_qc(self, sequence, min_length=28500, max_ambig_frac=0.02):
    """Evaluates sequence quality and ambiguous base proportion."""
    seq_clean = str(sequence).upper().replace(" ", "").replace("\r", "")
    seq_len = len(seq_clean)

    # Exclude terminal padding dashes when measuring internal length
    stripped_seq = seq_clean.strip("-")
    core_len = len(stripped_seq)

    n_ambig = seq_clean.count("N") + seq_clean.count("-")
    ambig_frac = n_ambig / seq_len if seq_len > 0 else 1.0

    qc_passed = True
    issues = []

    if core_len < min_length:
      qc_passed = False
      issues.append(
          f"Sequence length ({core_len:,} nt) is below threshold ({min_length:,}"
          " nt)"
      )

    if ambig_frac > max_ambig_frac:
      qc_passed = False
      issues.append(
          f"Ambiguous base content ({ambig_frac:.2%}) exceeds tolerance"
          f" ({max_ambig_frac:.1%})"
      )

    return {
        "qc_passed": qc_passed,
        "total_length": seq_len,
        "core_length": core_len,
        "ambiguous_fraction": round(ambig_frac, 4),
        "issues": issues,
    }

  def extract_genes(self, sequence):
    """Extracts S, E, M, N, and RdRp coding sequences from a full genome."""
    seq = str(sequence).upper().replace(" ", "").replace("\r", "")

    # Mode 1: Reference-aligned (length 29,903 nt standard alignment)
    if len(seq) == 29903:
      s_seq = seq[self.COORDINATES["S"][0] : self.COORDINATES["S"][1]]
      e_seq = seq[self.COORDINATES["E"][0] : self.COORDINATES["E"][1]]
      m_seq = seq[self.COORDINATES["M"][0] : self.COORDINATES["M"][1]]
      n_seq = seq[self.COORDINATES["N"][0] : self.COORDINATES["N"][1]]
      rdrp_seq = (
          seq[self.COORDINATES["RdRp_p1"][0] : self.COORDINATES["RdRp_p1"][1]]
          + seq[self.COORDINATES["RdRp_p2"][0] : self.COORDINATES["RdRp_p2"][1]]
      )
      return {
          "S": s_seq,
          "E": e_seq,
          "M": m_seq,
          "N": n_seq,
          "RdRp": rdrp_seq,
          "extraction_mode": "reference_coordinates",
      }

    # Mode 2: Anchor alignment for unaligned or trimmed consensus genomes
    extracted = {}
    offsets = {}
    for g, anchor in self.ANCHORS.items():
      pos = seq.find(anchor)
      offsets[g] = pos

    # If anchor offsets are found, slice relative to detected start
    if offsets["S_start"] >= 0:
      st = offsets["S_start"]
      extracted["S"] = seq[st : st + 3822]
    else:
      extracted["S"] = seq[self.COORDINATES["S"][0] : self.COORDINATES["S"][1]]

    if offsets["E_start"] >= 0:
      st = offsets["E_start"]
      extracted["E"] = seq[st : st + 228]
    else:
      extracted["E"] = seq[self.COORDINATES["E"][0] : self.COORDINATES["E"][1]]

    if offsets["M_start"] >= 0:
      st = offsets["M_start"]
      extracted["M"] = seq[st : st + 669]
    else:
      extracted["M"] = seq[self.COORDINATES["M"][0] : self.COORDINATES["M"][1]]

    if offsets["N_start"] >= 0:
      st = offsets["N_start"]
      extracted["N"] = seq[st : st + 1260]
    else:
      extracted["N"] = seq[self.COORDINATES["N"][0] : self.COORDINATES["N"][1]]

    if offsets["RdRp_start"] >= 0:
      st = offsets["RdRp_start"]
      extracted["RdRp"] = seq[st : st + 27] + seq[st + 26 : st + 2795]
    else:
      extracted["RdRp"] = (
          seq[self.COORDINATES["RdRp_p1"][0] : self.COORDINATES["RdRp_p1"][1]]
          + seq[self.COORDINATES["RdRp_p2"][0] : self.COORDINATES["RdRp_p2"][1]]
      )

    extracted["extraction_mode"] = "anchor_search"
    return extracted

  def extract_kmer_features(self, gene_dict):
    """Transforms extracted gene sequences into a normalized 320-feature k-mer DataFrame."""
    canonical_kmers = [
        "".join(p) for p in itertools.product("ACGT", repeat=self.k)
    ]
    feat_dict = {}

    for g in self.genes:
      seq = gene_dict.get(g, "").upper()
      seq_len = len(seq)
      total_possible_kmers = max(seq_len - self.k + 1, 1)
      counts = Counter(seq[i : i + self.k] for i in range(seq_len - self.k + 1))

      for kmer in canonical_kmers:
        feat_dict[f"{g}_{kmer}"] = [
            counts.get(kmer, 0) / total_possible_kmers
        ]

    df_feats = pd.DataFrame(feat_dict)
    if self.feature_columns:
      df_feats = df_feats.reindex(columns=self.feature_columns, fill_value=0.0)
    return df_feats

  def predict_genome(
      self,
      fasta_input,
      confidence_threshold=0.70,
      min_length=28500,
      max_ambig_frac=0.02,
      allow_failed_qc=False,
  ):
    """End-to-end inference function for a single or multi-FASTA genome input.

    Parameters:
    -----------
    fasta_input : str, Path, or file-like
        FASTA file path, raw sequence string, or uploaded stream.
    confidence_threshold : float
        Probability cutoff below which a prediction is flagged as ambiguous / novel.
    min_length : int
        Minimum core sequence length required to pass quality control.
    max_ambig_frac : float
        Maximum fraction of ambiguous bases (N or -) allowed to pass QC.
    allow_failed_qc : bool
        If True, continues gene extraction and prediction even if QC fails.

    Returns:
    --------
    list of dict : Structured prediction reports.
    """
    if self.model is None:
      raise RuntimeError(
          "Model artifact is not loaded. Train the model first or provide"
          " artifact_path."
      )

    records = self.parse_fasta(fasta_input)
    results = []

    for header, seq in records:
      # Step 1: Quality Control using adjustable diagnostic thresholds
      qc_report = self.check_qc(
          seq, min_length=min_length, max_ambig_frac=max_ambig_frac
      )

      if not qc_report["qc_passed"] and not allow_failed_qc:
        results.append({
            "accession": header,
            "status": "QC_FAILED",
            "predicted_variant": "Unknown",
            "confidence": 0.0,
            "probabilities": {c: 0.0 for c in self.classes},
            "qc_report": qc_report,
            "gene_lengths": {},
        })
        continue

      # Step 2: Gene Extraction
      gene_dict = self.extract_genes(seq)
      gene_lens = {
          g: len(gene_dict[g]) for g in self.genes if g in gene_dict
      }

      # Step 3: Feature Extraction
      df_features = self.extract_kmer_features(gene_dict)

      # Step 4: Machine Learning Inference
      pred_variant = self.model.predict(df_features)[0]
      probs = self.model.predict_proba(df_features)[0]
      conf = float(np.max(probs))

      # Step 5: Anomaly / OOD Detection
      status = (
          "CONFIDENT"
          if conf >= confidence_threshold
          else "AMBIGUOUS / NOVEL_VARIANT_CANDIDATE"
      )

      prob_dict = {
          cls_name: round(float(p), 4) for cls_name, p in zip(self.classes, probs)
      }

      results.append({
          "accession": header,
          "status": status,
          "predicted_variant": pred_variant,
          "confidence": round(conf, 4),
          "probabilities": prob_dict,
          "qc_report": qc_report,
          "gene_lengths": gene_lens,
      })

    return results
