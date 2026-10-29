"""
Personalized TED Talks Recommendation System Using Machine Learning
Module 3: Feature Engineering and Vectorization

Author: Leena Jarapala
Internship: Infosys Springboard Virtual Internship 7.0
GitHub: https://github.com/leena-11/ted-talks-recommendation-system

Description:
This module transforms preprocessed TED Talks text and metadata into numerical
TF-IDF feature vectors, saves the vectorizer, sparse matrix, and index mappings,
computes verified cosine similarity scores, and outputs similarity analytics.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import joblib

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# Color definitions
PRIMARY_COLOR = "#E62B1E"    # TED Red
SECONDARY_COLOR = "#1D3557"  # Deep Navy
ACCENT_COLOR = "#457B9D"     # Slate Blue
HIGHLIGHT_COLOR = "#2A9D8F"  # Teal


# ---------------------------------------------------------------------------
# TF-IDF Configuration Parameters
# ---------------------------------------------------------------------------
TFIDF_PARAMS = {
    "max_features": 5000,
    "min_df": 2,
    "max_df": 0.70,
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "norm": "l2",
    "stop_words": "english"
}


def build_combined_content(df: pd.DataFrame) -> pd.Series:
    """
    Construct weighted composite content representation combining title, topics,
    description, and transcript.

    Weighting rationale:
    - Titles and tags contain concentrated topical signals; repeating them 2x gives
      them proportional salience against long conversational transcripts (~800+ words).

    Args:
        df: Cleaned TED Talks DataFrame.

    Returns:
        Series of combined text strings.
    """
    clean_title = df["clean_title"].fillna("").astype(str)
    clean_tags = df["clean_tags"].fillna("").astype(str)
    clean_desc = df["clean_description"].fillna("").astype(str)
    clean_trans = df["clean_transcript"].fillna("").astype(str)

    # Weighted composite
    combined = (
        clean_title + " " + clean_title + " " +
        clean_tags + " " + clean_tags + " " +
        clean_desc + " " +
        clean_trans
    ).str.strip()

    return combined


def engineer_features(
    df: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None
) -> Tuple[TfidfVectorizer, sparse.csr_matrix, pd.Series]:
    """
    Apply TF-IDF vectorization to composite talk content.

    Args:
        df: Cleaned TED Talks DataFrame.
        params: TF-IDF vectorizer hyperparameters.

    Returns:
        Tuple of (fitted_vectorizer, sparse_tfidf_matrix, combined_content_series).
    """
    if params is None:
        params = TFIDF_PARAMS

    required_cols = ["clean_title", "clean_tags", "clean_description", "clean_transcript"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset for feature engineering: {missing}")

    print("[Module 3] Constructing weighted composite content field...")
    combined_content = build_combined_content(df)

    print(f"[Module 3] Fitting TfidfVectorizer (vocabulary cap: {params['max_features']:,}, "
          f"ngrams: {params['ngram_range']}, min_df: {params['min_df']}, max_df: {params['max_df']})...")

    vectorizer = TfidfVectorizer(
        max_features=params["max_features"],
        min_df=params["min_df"],
        max_df=params["max_df"],
        ngram_range=params["ngram_range"],
        sublinear_tf=params["sublinear_tf"],
        norm=params["norm"],
        stop_words=params["stop_words"]
    )

    tfidf_matrix = vectorizer.fit_transform(combined_content)
    # Ensure CSR format for fast row slicing
    if not isinstance(tfidf_matrix, sparse.csr_matrix):
        tfidf_matrix = tfidf_matrix.tocsr()

    vocab_size = len(vectorizer.vocabulary_)
    matrix_shape = tfidf_matrix.shape
    sparsity = (1.0 - (tfidf_matrix.nnz / (matrix_shape[0] * matrix_shape[1]))) * 100

    print(f"[Module 3] TF-IDF Matrix Shape: {matrix_shape[0]:,} talks x {matrix_shape[1]:,} features")
    print(f"[Module 3] Learned Vocabulary:   {vocab_size:,} distinct terms/ngrams")
    print(f"[Module 3] Matrix Sparsity:       {sparsity:.2f}% (stored as efficient scipy sparse CSR)")

    return vectorizer, tfidf_matrix, combined_content


def save_artifacts(
    vectorizer: TfidfVectorizer,
    tfidf_matrix: sparse.csr_matrix,
    df: pd.DataFrame,
    models_dir: Path
) -> Dict[str, Path]:
    """
    Save model artifacts: fitted vectorizer, sparse matrix, and talk index mapping.

    Args:
        vectorizer: Fitted TfidfVectorizer.
        tfidf_matrix: Scipy sparse CSR matrix.
        df: Cleaned TED Talks DataFrame.
        models_dir: Output directory for models.

    Returns:
        Dict mapping artifact name to saved file Path.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    vec_path = models_dir / "tfidf_vectorizer.pkl"
    mat_path = models_dir / "tfidf_matrix.npz"
    map_path = models_dir / "talk_index_mapping.csv"

    # 1. Save Vectorizer
    joblib.dump(vectorizer, vec_path)
    print(f"[Module 3] Saved TF-IDF Vectorizer to: {vec_path.resolve()}")

    # 2. Save Sparse Matrix
    sparse.save_npz(mat_path, tfidf_matrix)
    print(f"[Module 3] Saved Sparse Feature Matrix to: {mat_path.resolve()}")

    # 3. Save Talk Index Mapping
    mapping_cols = ["title", "main_speaker", "url", "duration_minutes", "views", "clean_tags"]
    # Fallback if any column has slightly different name
    avail_cols = [c for c in mapping_cols if c in df.columns]
    mapping_df = df[avail_cols].copy()
    mapping_df.insert(0, "talk_index", mapping_df.index)
    mapping_df.to_csv(map_path, index=False)
    print(f"[Module 3] Saved Talk Index Mapping to: {map_path.resolve()} ({len(mapping_df):,} rows)")

    return {
        "vectorizer": vec_path,
        "matrix": mat_path,
        "mapping": map_path
    }


def compute_talk_similarities(
    query_idx: int,
    tfidf_matrix: sparse.csr_matrix
) -> np.ndarray:
    """
    Compute cosine similarity between a single talk vector and all talks in the corpus.
    Since row vectors are L2-normalized, linear_kernel (dot product) equals cosine similarity.

    Args:
        query_idx: Row index of the query talk.
        tfidf_matrix: Scipy sparse CSR matrix.

    Returns:
        1D numpy array of similarity scores.
    """
    query_vec = tfidf_matrix[query_idx]
    # linear_kernel performs fast BLAS dot product
    sims = linear_kernel(query_vec, tfidf_matrix).flatten()
    return sims


def generate_sample_similarity_report(
    df: pd.DataFrame,
    tfidf_matrix: sparse.csr_matrix,
    sample_titles: List[str],
    top_n: int = 5
) -> pd.DataFrame:
    """
    Compute and document top-N recommendation matches for a curated set of landmark talks.

    Args:
        df: Cleaned TED Talks DataFrame.
        tfidf_matrix: Feature matrix.
        sample_titles: Titles to evaluate.
        top_n: Number of matches per query.

    Returns:
        DataFrame containing sample match results.
    """
    results = []
    # Build case-insensitive title lookup
    title_to_idx = {title.strip().lower(): idx for idx, title in enumerate(df["title"])}

    for q_title in sample_titles:
        q_norm = q_title.strip().lower()
        if q_norm not in title_to_idx:
            # Fuzzy match fallback
            matches = [t for t in title_to_idx.keys() if q_norm in t]
            if not matches:
                continue
            q_norm = matches[0]

        idx = title_to_idx[q_norm]
        actual_title = df.loc[idx, "title"]
        speaker = df.loc[idx, "main_speaker"]

        sims = compute_talk_similarities(idx, tfidf_matrix)
        # Exclude self
        sims[idx] = -1.0

        top_indices = np.argsort(sims)[::-1][:top_n]

        for rank, match_idx in enumerate(top_indices, 1):
            score = float(sims[match_idx])
            results.append({
                "Query_Talk": actual_title,
                "Query_Speaker": speaker,
                "Rank": rank,
                "Recommended_Talk": df.loc[match_idx, "title"],
                "Recommended_Speaker": df.loc[match_idx, "main_speaker"],
                "Similarity_Score": round(score, 4),
                "Similarity_Percentage": f"{score * 100:.1f}%",
                "Recommended_Tags": df.loc[match_idx, "clean_tags"] if "clean_tags" in df.columns else ""
            })

    return pd.DataFrame(results)


def plot_similarity_analytics(
    df: pd.DataFrame,
    tfidf_matrix: sparse.csr_matrix,
    output_path: Path
):
    """
    Generate publication visualization showing:
    1. Cosine similarity heatmap between diverse landmark TED talks across distinct domains.
    2. Distribution of highest similarity scores across the corpus.

    Args:
        df: Cleaned TED Talks DataFrame.
        tfidf_matrix: Sparse matrix.
        output_path: Path to save PNG.
    """
    landmark_titles = [
        "Do schools kill creativity?",
        "Your body language may shape who you are",
        "Averting the climate crisis",
        "The thrilling potential of SixthSense technology",
        "The surprising science of happiness",
        "How great leaders inspire action",
        "My stroke of insight",
        "The puzzle of motivation"
    ]

    title_to_idx = {title.strip().lower(): idx for idx, title in enumerate(df["title"])}
    valid_indices = []
    short_labels = []

    for t in landmark_titles:
        t_low = t.lower()
        if t_low in title_to_idx:
            valid_indices.append(title_to_idx[t_low])
            speaker = df.loc[title_to_idx[t_low], "main_speaker"]
            # Truncated label: Speaker (Short Title)
            short_labels.append(f"{speaker}: {t[:22]}...")

    if len(valid_indices) < 4:
        return

    # Compute pairwise similarity submatrix
    sub_matrix = tfidf_matrix[valid_indices]
    pairwise_sims = linear_kernel(sub_matrix, sub_matrix)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    sns.heatmap(
        pairwise_sims,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=short_labels,
        yticklabels=short_labels,
        cbar_kws={"label": "Cosine Similarity"},
        linewidths=0.5,
        ax=ax
    )

    ax.set_title("Cosine Similarity Matrix Among Landmark TED Talks", fontsize=14, weight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Module 3] Saved similarity heatmap to: {output_path.resolve()}")


def verify_saved_artifacts(models_dir: Path) -> bool:
    """
    Test loading the saved vectorizer, matrix, and mapping to ensure Module 4 compatibility.

    Args:
        models_dir: Directory containing model files.

    Returns:
        True if all files load cleanly.
    """
    print("[Module 3] Verifying artifact loading for Module 4 compatibility...")
    vec_p = models_dir / "tfidf_vectorizer.pkl"
    mat_p = models_dir / "tfidf_matrix.npz"
    map_p = models_dir / "talk_index_mapping.csv"

    if not (vec_p.is_file() and mat_p.is_file() and map_p.is_file()):
        raise FileNotFoundError(f"One or more artifacts missing in {models_dir}")

    vec = joblib.load(vec_p)
    mat = sparse.load_npz(mat_p)
    mapping = pd.read_csv(map_p)

    assert hasattr(vec, "transform"), "Vectorizer missing transform method"
    assert mat.shape[0] == len(mapping), f"Matrix rows ({mat.shape[0]}) != mapping rows ({len(mapping)})"

    print(f"[Module 3] Verification passed: {mat.shape[0]:,} talks successfully mapped to {mat.shape[1]:,} features.")
    return True


def main():
    print("=" * 70)
    print("MODULE 3: FEATURE ENGINEERING AND VECTORIZATION")
    print("=" * 70)

    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / "models"
    reports_dir = base_dir / "reports"
    vis_dir = base_dir / "visualizations"

    # 1. Load cleaned dataset
    data_path = base_dir / "cleaned_ted_talks.csv"
    if not data_path.is_file():
        data_path = base_dir / "data" / "cleaned_ted_talks.csv"
    if not data_path.is_file():
        raise FileNotFoundError(f"Cleaned dataset not found at {data_path}. Run preprocess_ted_data.py first.")

    print(f"[Module 3] Loading cleaned dataset from: {data_path.resolve()}")
    df = pd.read_csv(data_path)
    print(f"[Module 3] Loaded {len(df):,} TED Talks records.")

    # 2. Engineer features
    vectorizer, tfidf_matrix, combined_content = engineer_features(df, TFIDF_PARAMS)

    # 3. Save artifacts
    artifacts = save_artifacts(vectorizer, tfidf_matrix, df, models_dir)

    # 4. Compute sample similarities for landmark talks
    sample_queries = [
        "Do schools kill creativity?",
        "Your body language may shape who you are",
        "Averting the climate crisis",
        "The thrilling potential of SixthSense technology",
        "The surprising science of happiness"
    ]
    sample_df = generate_sample_similarity_report(df, tfidf_matrix, sample_queries, top_n=5)

    sample_report_path = reports_dir / "sample_similarity_matches.csv"
    sample_report_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(sample_report_path, index=False)
    sample_df.to_csv(models_dir / "sample_similarity_matches.csv", index=False)
    print(f"[Module 3] Saved sample similarity report to: {sample_report_path.resolve()}")

    # 5. Generate heatmap visualization
    heatmap_path = vis_dir / "sample_similarity_heatmap.png"
    plot_similarity_analytics(df, tfidf_matrix, heatmap_path)

    # 6. Verify artifacts
    verify_saved_artifacts(models_dir)

    print("\n[Sample Recommendation Results Preview]")
    preview = sample_df[sample_df["Query_Talk"].str.contains("creativity", case=False)]
    for _, r in preview.iterrows():
        print(f" #{r['Rank']} [{r['Similarity_Percentage']}] {r['Recommended_Talk']} by {r['Recommended_Speaker']}")

    print("=" * 70)
    print("MODULE 3 EXECUTION COMPLETE - READY FOR RECOMMENDATION ENGINE")
    print("=" * 70)


if __name__ == "__main__":
    main()
