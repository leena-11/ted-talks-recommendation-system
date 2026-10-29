"""
Personalized TED Talks Recommendation System Using Machine Learning
Module 2: Data Exploration and Data Preprocessing

Author: Leena Jarapala
Internship: Infosys Springboard Virtual Internship 7.0
GitHub: https://github.com/leena-11/ted-talks-recommendation-system

Description:
This module cleans and normalizes TED Talks text fields, handles missing values,
calculates token and character metrics, parses tags and ratings, creates
publication-grade EDA visualizations, and outputs cleaned data and quality audits.
"""

import os
import sys
import ast
import re
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Set
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# Standard styling configuration
PRIMARY_COLOR = "#E62B1E"    # TED Red
SECONDARY_COLOR = "#1D3557"  # Deep Navy
ACCENT_COLOR = "#457B9D"     # Slate Blue
HIGHLIGHT_COLOR = "#2A9D8F"  # Teal
NEUTRAL_DARK = "#2B2D42"     # Charcoal


def clean_text(
    text: Any,
    remove_stage_directions: bool = True,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    remove_stopwords: bool = True,
    custom_stopwords: Optional[Set[str]] = None
) -> str:
    """
    Standardized text normalization utility.

    Steps:
    1. Removes parenthetical stage directions (e.g., (Applause), (Laughter)).
    2. Lowercases text.
    3. Replaces punctuation and special symbols with spaces.
    4. Filters out English stopwords.
    5. Normalizes whitespace and strips.

    Args:
        text: Input text string or value.
        remove_stage_directions: Strip parenthesized cues.
        lowercase: Convert to lowercase.
        remove_punctuation: Remove non-alphanumeric symbols.
        remove_stopwords: Filter common English stopwords.
        custom_stopwords: Optional custom stopwords set.

    Returns:
        Cleaned text string.
    """
    if pd.isna(text):
        return ""

    s = str(text)

    # 1. Remove stage directions in transcripts
    if remove_stage_directions:
        s = re.sub(r"\([^)]*\)", " ", s)

    # 2. Lowercase
    if lowercase:
        s = s.lower()

    # 3. Punctuation handling
    if remove_punctuation:
        s = re.sub(r"[^a-z0-9\s]", " ", s)

    # 4. Token normalization and stopword removal
    tokens = s.split()
    if remove_stopwords:
        stops = custom_stopwords if custom_stopwords is not None else ENGLISH_STOP_WORDS
        tokens = [t for t in tokens if t not in stops and len(t) > 1]

    return " ".join(tokens)


def safe_parse_tags(tags_val: Any) -> List[str]:
    """
    Safely parse tag strings formatted as python list strings into a list of strings.

    Args:
        tags_val: String representation of list, list, or NaN.

    Returns:
        List of cleaned, lowercased tag strings.
    """
    if pd.isna(tags_val):
        return []
    if isinstance(tags_val, list):
        return [str(t).strip().lower() for t in tags_val if str(t).strip()]

    s = str(tags_val).strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [str(t).strip().lower() for t in parsed if str(t).strip()]
    except Exception:
        pass

    # Regex fallback
    matches = re.findall(r"'([^']*)'", s)
    if matches:
        return [m.strip().lower() for m in matches if m.strip()]
    return [t.strip().lower() for t in s.strip("[]").split(",") if t.strip()]


def safe_parse_ratings(ratings_val: Any) -> List[Dict[str, Any]]:
    """
    Safely parse ratings string into a list of rating dictionaries.

    Args:
        ratings_val: String representation of list of dicts.

    Returns:
        List of dicts with 'id', 'name', and 'count'.
    """
    if pd.isna(ratings_val):
        return []
    if isinstance(ratings_val, list):
        return ratings_val

    s = str(ratings_val).strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return []


def generate_quality_report(df: pd.DataFrame, dataset_name: str = "cleaned_ted_talks.csv") -> pd.DataFrame:
    """
    Generate comprehensive data quality audit across all columns.

    Args:
        df: Input DataFrame.
        dataset_name: Label for report.

    Returns:
        DataFrame with quality metrics.
    """
    records = []
    total_rows = len(df)

    for col in df.columns:
        s = df[col]
        null_count = int(s.isnull().sum())
        null_pct = round((null_count / total_rows) * 100, 3)

        blank_count = 0
        padded_count = 0
        if s.dtype == "object":
            non_null = s.dropna().astype(str)
            blank_count = int((non_null.str.strip() == "").sum())
            padded_count = int((non_null != non_null.str.strip()).sum())

        unique_count = int(s.nunique(dropna=True))
        sample_val = str(s.dropna().iloc[0]) if not s.dropna().empty else "N/A"
        if len(sample_val) > 40:
            sample_val = sample_val[:37] + "..."

        issues = []
        if null_count > 0:
            issues.append(f"{null_count} nulls ({null_pct}%)")
        if blank_count > 0:
            issues.append(f"{blank_count} empty strings")
        if padded_count > 0:
            issues.append(f"{padded_count} padded strings")

        assessment = "; ".join(issues) if issues else "Clean"

        records.append({
            "Dataset": dataset_name,
            "Column": col,
            "Data_Type": str(s.dtype),
            "Total_Records": total_rows,
            "Non_Null_Count": total_rows - null_count,
            "Missing_Count": null_count,
            "Missing_Pct": null_pct,
            "Blank_Strings": blank_count,
            "Whitespace_Padded": padded_count,
            "Unique_Values": unique_count,
            "Sample_Value": sample_val,
            "Quality_Assessment": assessment
        })

    return pd.DataFrame(records)


def preprocess_data(
    input_path: Optional[Path] = None,
    base_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw or collected dataset and perform complete cleaning and feature calculation.

    Args:
        input_path: Path to input dataset.
        base_dir: Project root directory.

    Returns:
        Tuple of (cleaned_df, quality_report_df, summary_df).
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent

    if input_path is None:
        # Check collected dataset first, then fall back to ted_main.csv
        collected_file = base_dir / "data" / "collected_ted_talks.csv"
        if collected_file.is_file():
            input_path = collected_file
        else:
            from data_collection import collect_ted_data
            df_collected, _ = collect_ted_data(base_dir=base_dir)
            input_path = base_dir / "data" / "collected_ted_talks.csv"

    print(f"[Module 2] Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)
    initial_records = len(df)
    print(f"[Module 2] Verified input record count: {initial_records:,} talks")

    # 1. Strip string whitespace and newlines
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # 2. Impute missing speaker_occupation (6 records)
    df["speaker_occupation"] = df["speaker_occupation"].replace({"nan": "Unknown", "": "Unknown"}).fillna("Unknown")

    # 3. Categorical encoding for speaker occupation
    occupation_counts = df["speaker_occupation"].value_counts()
    top_occupations = occupation_counts.head(50).index.tolist()
    df["speaker_occupation_group"] = df["speaker_occupation"].apply(lambda occ: occ if occ in top_occupations else "Other")
    df["speaker_occupation_code"] = pd.Categorical(df["speaker_occupation_group"]).codes

    # 4. Parse tags and topics
    parsed_tags = df["tags"].apply(safe_parse_tags)
    df["clean_tags"] = parsed_tags.apply(lambda tags: " ".join(tags))
    df["clean_topics"] = df["clean_tags"]
    df["tag_count"] = parsed_tags.apply(len)

    # 5. Handle transcript availability and text cleaning
    if "transcript" not in df.columns:
        df["transcript"] = ""

    df["transcript"] = df["transcript"].replace({"nan": ""}).fillna("")
    df["has_transcript"] = (df["transcript"].str.strip() != "").astype(int)

    # Clean text columns while strictly preserving original raw columns
    print("[Module 2] Normalizing text fields (lowercasing, punctuation, stopwords)...")
    df["clean_title"] = df["title"].apply(lambda t: clean_text(t, remove_stage_directions=False, remove_stopwords=False))
    df["clean_description"] = df["description"].apply(lambda d: clean_text(d, remove_stage_directions=False, remove_stopwords=True))
    df["clean_transcript"] = df["transcript"].apply(lambda t: clean_text(t, remove_stage_directions=True, remove_stopwords=True))

    # 6. Combined text representation
    df["combined_text"] = (
        df["clean_title"] + " " +
        df["clean_tags"] + " " +
        df["clean_description"] + " " +
        df["clean_transcript"]
    ).str.strip()

    # 7. Word and character counts
    print("[Module 2] Calculating token length and character metrics...")
    df["title_word_count"] = df["clean_title"].apply(lambda s: len(s.split()) if s else 0)
    df["description_word_count"] = df["clean_description"].apply(lambda s: len(s.split()) if s else 0)
    df["transcript_word_count"] = df["clean_transcript"].apply(lambda s: len(s.split()) if s else 0)
    df["combined_text_word_count"] = df["combined_text"].apply(lambda s: len(s.split()) if s else 0)
    df["combined_text_char_count"] = df["combined_text"].str.len()

    # 8. Duration metrics and ratings
    if "duration" in df.columns:
        df["duration_minutes"] = round(df["duration"] / 60.0, 2)

    def extract_rating_summary(r_val):
        parsed = safe_parse_ratings(r_val)
        if not parsed:
            return 0, "Unknown"
        total = sum(item.get("count", 0) for item in parsed if isinstance(item, dict))
        top = max(parsed, key=lambda x: x.get("count", 0) if isinstance(x, dict) else 0)
        return total, top.get("name", "Unknown") if isinstance(top, dict) else "Unknown"

    if "ratings" in df.columns:
        rating_summary = df["ratings"].apply(extract_rating_summary)
        df["total_ratings"] = rating_summary.apply(lambda x: x[0])
        df["top_rating"] = rating_summary.apply(lambda x: x[1])
        top_cats = sorted(df["top_rating"].unique().tolist())
        rating_cat_map = {cat: idx for idx, cat in enumerate(top_cats)}
        df["top_rating_code"] = df["top_rating"].map(rating_cat_map)

    # 9. Generate Quality Report
    quality_report = generate_quality_report(df, dataset_name="cleaned_ted_talks.csv")

    # 10. Generate Summary Statistics
    summary_data = [
        {"Metric": "Total Talks Processed", "Value": str(len(df))},
        {"Metric": "Talks with Genuine Transcripts", "Value": f"{(df['has_transcript'] == 1).sum():,} ({(df['has_transcript'] == 1).mean()*100:.2f}%)"},
        {"Metric": "Non-Spoken / Performance Talks (No Transcript)", "Value": f"{(df['has_transcript'] == 0).sum():,} (Handled safely with metadata)"},
        {"Metric": "Average Transcript Word Count", "Value": f"{df[df['has_transcript'] == 1]['transcript_word_count'].mean():.1f} words"},
        {"Metric": "Median Talk Duration", "Value": f"{df['duration_minutes'].median():.1f} minutes"},
        {"Metric": "Average Views", "Value": f"{df['views'].mean():,.0f} views"},
        {"Metric": "Unique Speakers", "Value": f"{df['main_speaker'].nunique():,}"},
        {"Metric": "Unique Topic Tags Extracted", "Value": f"{len(set([t for sub in parsed_tags for t in sub])):,}"},
        {"Metric": "User-Item Interaction Matrix", "Value": "Unavailable (Public TED dataset contains aggregate community views/ratings; individual user watch logs are not present in genuine data)."}
    ]
    summary_df = pd.DataFrame(summary_data)

    return df, quality_report, summary_df


def generate_visualizations(
    df: pd.DataFrame,
    vis_dir: Path,
    eda_dir: Optional[Path] = None
) -> List[Path]:
    """
    Generate required visualizations:
    1. Topic / tag frequency distribution
    2. Transcript word count distribution

    Args:
        df: Cleaned TED Talks DataFrame.
        vis_dir: Path to visualizations directory.
        eda_dir: Optional additional reports/eda directory.

    Returns:
        List of generated image paths.
    """
    vis_dir = Path(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    generated_charts = []

    # 1. Topic Frequency Chart
    parsed_tags = df["tags"].apply(safe_parse_tags)
    all_tags = [tag for sublist in parsed_tags for tag in sublist]
    tag_counts = Counter(all_tags).most_common(20)
    tag_df = pd.DataFrame(tag_counts, columns=["Tag", "Frequency"]).sort_values("Frequency", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    bars = ax.barh(tag_df["Tag"], tag_df["Frequency"], color=PRIMARY_COLOR, edgecolor=SECONDARY_COLOR, linewidth=0.8)

    ax.set_title("Top 20 Most Frequent TED Talk Topics & Tags", fontsize=15, weight="bold", pad=15)
    ax.set_xlabel("Number of Talks", fontsize=12, labelpad=10)
    ax.set_ylabel("Topic / Tag", fontsize=12)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 8, bar.get_y() + bar.get_height() / 2, f"{int(width):,}",
                ha="left", va="center", fontsize=9, weight="bold", color=NEUTRAL_DARK)

    ax.set_xlim(0, max(tag_df["Frequency"]) * 1.15)
    plt.tight_layout()
    chart1_path = vis_dir / "topic_frequency.png"
    fig.savefig(chart1_path, bbox_inches="tight")
    plt.close(fig)
    generated_charts.append(chart1_path)

    # 2. Transcript Word Count Distribution
    trans_talks = df[df["has_transcript"] == 1]
    word_counts = trans_talks["transcript_word_count"]
    median_wc = float(word_counts.median())
    mean_wc = float(word_counts.mean())

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    sns.histplot(
        word_counts,
        bins=40,
        kde=True,
        color=SECONDARY_COLOR,
        edgecolor="white",
        line_kws={"color": PRIMARY_COLOR, "linewidth": 2.5},
        ax=ax
    )

    ax.axvline(median_wc, color=PRIMARY_COLOR, linestyle="--", linewidth=2, label=f"Median: {int(median_wc):,} words")
    ax.axvline(mean_wc, color=HIGHLIGHT_COLOR, linestyle="-.", linewidth=2, label=f"Mean: {int(mean_wc):,} words")

    ax.set_title("Distribution of Transcript Word Counts (Cleaned)", fontsize=15, weight="bold", pad=15)
    ax.set_xlabel("Word Count (Stopwords Removed)", fontsize=12, labelpad=10)
    ax.set_ylabel("Number of Talks", fontsize=12)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart2_path = vis_dir / "transcript_word_count_distribution.png"
    fig.savefig(chart2_path, bbox_inches="tight")
    plt.close(fig)
    generated_charts.append(chart2_path)

    # Also sync copies to reports/eda if directory exists
    if eda_dir is not None:
        eda_dir = Path(eda_dir)
        eda_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(chart1_path, eda_dir / "topic_frequency.png")
        shutil.copy(chart2_path, eda_dir / "transcript_word_count_distribution.png")

    return generated_charts


def assess_user_item_matrix(base_dir: Path) -> Dict[str, Any]:
    """
    Check for genuine user-item interaction logs and build interaction matrix schema documentation.

    Returns:
        Dict documenting status and limitations.
    """
    info = {
        "status": "Unavailable",
        "reason": "The public TED Talks dataset contains community aggregate metrics (views, comments, ratings counts) per talk rather than individualized user IDs or per-user watch histories.",
        "collaborative_filtering_readiness": "Cannot be genuinely trained without fabricating data. A content-based recommendation approach using TF-IDF and metadata cosine similarity is the appropriate and principled solution.",
        "schema_design": "If user logs are introduced, user-item interaction matrix should be configured with (user_id, talk_id, rating_or_watch_ratio)."
    }

    schema_file = base_dir / "reports" / "interaction_matrix_schema.csv"
    schema_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([info]).to_csv(schema_file, index=False)

    return info


def main():
    print("=" * 70)
    print("MODULE 2: TED TALKS DATA EXPLORATION AND DATA PREPROCESSING")
    print("=" * 70)

    base_dir = Path(__file__).resolve().parent
    vis_dir = base_dir / "visualizations"
    reports_dir = base_dir / "reports"

    # Preprocess dataset
    cleaned_df, quality_df, summary_df = preprocess_data(base_dir=base_dir)

    # Save cleaned dataset to both root and data/
    root_cleaned = base_dir / "cleaned_ted_talks.csv"
    data_cleaned = base_dir / "data" / "cleaned_ted_talks.csv"
    cleaned_df.to_csv(root_cleaned, index=False)
    cleaned_df.to_csv(data_cleaned, index=False)
    print(f"[Module 2] Saved cleaned dataset to: {root_cleaned} ({len(cleaned_df):,} talks)")
    print(f"[Module 2] Saved cleaned dataset to: {data_cleaned}")

    # Save quality and summary reports to root and reports/
    quality_df.to_csv(base_dir / "data_quality_report.csv", index=False)
    quality_df.to_csv(reports_dir / "data_quality_report.csv", index=False)
    quality_df.to_csv(base_dir / "preprocessing_quality_report.csv", index=False)
    quality_df.to_csv(reports_dir / "preprocessing_quality_report.csv", index=False)

    summary_df.to_csv(base_dir / "preprocessing_summary.csv", index=False)
    summary_df.to_csv(reports_dir / "preprocessing_summary.csv", index=False)
    print(f"[Module 2] Saved quality reports to data_quality_report.csv and preprocessing_summary.csv")

    # Generate charts
    charts = generate_visualizations(cleaned_df, vis_dir=vis_dir, eda_dir=reports_dir / "eda")
    print(f"[Module 2] Generated visualizations in {vis_dir}:")
    for c in charts:
        print(f"  - {c.name}")

    # Assess user-item matrix
    user_status = assess_user_item_matrix(base_dir)
    print(f"[Module 2] Interaction Matrix Status: {user_status['status']}")
    print(f"  -> {user_status['reason']}")

    print("\n[Module 2 Summary Statistics]")
    for _, row in summary_df.iterrows():
        print(f" - {row['Metric']:<45}: {row['Value']}")

    print("=" * 70)
    print("MODULE 2 EXECUTION COMPLETE - READY FOR FEATURE ENGINEERING")
    print("=" * 70)


if __name__ == "__main__":
    main()
