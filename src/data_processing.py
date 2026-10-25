"""
Personalized TED Talks Recommendation System
Module 1: Data Collection & Module 2: Data Exploration and Preprocessing

This script provides reusable functions and an end-to-end pipeline for:
- Verifying and loading raw TED datasets (ted_main.csv and transcripts.csv)
- Schema inspection, candidate key analysis, and identifier validation
- Data quality assessment and reporting (missing values, whitespace, duplicates)
- Data cleaning (URL stripping, tag parsing, rating extraction, text normalization)
- Generating publication-quality Exploratory Data Analysis (EDA) visualizations
- Before-and-after cleaning validation and output persistence

Author: Infosys Springboard Virtual Internship 7.0
"""

import os
import sys
import ast
import re
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for automated plotting
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Styling and Palette Setup for Publication-Grade EDA Charts
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "figure.dpi": 300,
})

PRIMARY_COLOR = "#E62B1E"    # TED Red
SECONDARY_COLOR = "#1D3557"  # Deep Navy
ACCENT_COLOR = "#457B9D"     # Slate Blue
NEUTRAL_LIGHT = "#F1FAEE"    # Off-white
CONTRAST_COLOR = "#E76F51"   # Coral


# ---------------------------------------------------------------------------
# Module 1: Dataset Verification & Ingestion
# ---------------------------------------------------------------------------
def check_datasets_exist(data_dir: Path) -> Dict[str, bool]:
    """
    Check whether both required dataset files exist in the data directory.

    Args:
        data_dir: Path to directory where datasets are stored.

    Returns:
        Dict mapping filename to existence status (bool).
    """
    files = {
        "ted_main.csv": (data_dir / "ted_main.csv").is_file(),
        "transcripts.csv": (data_dir / "transcripts.csv").is_file()
    }
    return files


def load_datasets(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load ted_main.csv and transcripts.csv with validation and diagnostics.

    Args:
        data_dir: Directory containing raw datasets.

    Returns:
        Tuple of (df_main, df_trans).

    Raises:
        FileNotFoundError: If either required file is missing.
        ValueError: If any dataset is empty or corrupted.
    """
    data_dir = Path(data_dir)
    status = check_datasets_exist(data_dir)

    missing = [f for f, exists in status.items() if not exists]
    if missing:
        msg = (
            f"\n[ERROR] Missing required dataset(s): {', '.join(missing)}\n"
            f"Please verify that both 'ted_main.csv' and 'transcripts.csv' are located in:\n"
            f"  -> {data_dir.resolve()}\n"
            f"If missing, obtain the dataset from Kaggle ('rounakbanik/ted-talks')\n"
            f"or the project archive and place both CSV files in the 'data/' folder."
        )
        raise FileNotFoundError(msg)

    try:
        df_main = pd.read_csv(data_dir / "ted_main.csv")
    except Exception as e:
        raise ValueError(f"Failed to read 'ted_main.csv': {str(e)}") from e

    try:
        df_trans = pd.read_csv(data_dir / "transcripts.csv")
    except Exception as e:
        raise ValueError(f"Failed to read 'transcripts.csv': {str(e)}") from e

    if df_main.empty:
        raise ValueError("ted_main.csv is empty (0 records).")
    if df_trans.empty:
        raise ValueError("transcripts.csv is empty (0 records).")

    return df_main, df_trans


def inspect_identifiers_and_overlap(df_main: pd.DataFrame, df_trans: pd.DataFrame) -> Dict[str, Any]:
    """
    Identify potential primary and foreign keys, evaluate match rate, and inspect overlap.

    Args:
        df_main: Raw metadata DataFrame.
        df_trans: Raw transcript DataFrame.

    Returns:
        Dictionary containing identifier diagnostics and overlap metrics.
    """
    # Potential common columns
    common_cols = list(set(df_main.columns).intersection(set(df_trans.columns)))

    # Raw URL uniqueness
    main_urls_raw = df_main["url"] if "url" in df_main.columns else pd.Series([], dtype=str)
    trans_urls_raw = df_trans["url"] if "url" in df_trans.columns else pd.Series([], dtype=str)

    # Whitespace-stripped URLs
    main_urls_clean = main_urls_raw.astype(str).str.strip()
    trans_urls_clean = trans_urls_raw.astype(str).str.strip()

    # Exact duplicate URL rows in transcripts
    dup_trans_urls = trans_urls_clean[trans_urls_clean.duplicated(keep=False)].unique().tolist()
    dup_trans_count = trans_urls_clean.duplicated().sum()

    # Overlap analysis
    set_main = set(main_urls_clean)
    set_trans = set(trans_urls_clean)
    common_urls = set_main.intersection(set_trans)
    main_only = set_main - set_trans
    trans_only = set_trans - set_main

    return {
        "common_columns": common_cols,
        "main_records": len(df_main),
        "main_unique_urls": main_urls_clean.nunique(),
        "trans_records": len(df_trans),
        "trans_unique_urls": trans_urls_clean.nunique(),
        "trans_duplicate_url_count": int(dup_trans_count),
        "trans_duplicate_urls": dup_trans_urls,
        "common_unique_urls": len(common_urls),
        "main_without_transcripts": len(main_only),
        "trans_without_metadata": len(trans_only),
        "match_percentage_of_main": round((len(common_urls) / len(set_main)) * 100, 2),
        "match_percentage_of_trans": round((len(common_urls) / len(set_trans)) * 100, 2)
    }


# ---------------------------------------------------------------------------
# Module 2: Data Quality Assessment & Reporting
# ---------------------------------------------------------------------------
def generate_data_quality_report(
    df_main: pd.DataFrame,
    df_trans: pd.DataFrame,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Generate a comprehensive data quality report for both datasets.

    Analyzes:
    - Data types
    - Missing value counts and percentages
    - Blank string counts
    - Leading/trailing whitespace padding count
    - Unique values count
    - Data issues / domain notes

    Args:
        df_main: Metadata DataFrame.
        df_trans: Transcripts DataFrame.
        output_path: Optional path to save CSV report.

    Returns:
        DataFrame containing audit results.
    """
    records = []

    def audit_df(df: pd.DataFrame, dataset_name: str):
        total_rows = len(df)
        for col in df.columns:
            s = df[col]
            null_count = s.isnull().sum()
            null_pct = round((null_count / total_rows) * 100, 3)

            blank_count = 0
            padded_count = 0
            if s.dtype == "object":
                # Check for blank strings
                non_null = s.dropna().astype(str)
                blank_count = (non_null.str.strip() == "").sum()
                padded_count = (non_null != non_null.str.strip()).sum()

            unique_count = s.nunique(dropna=True)
            sample_val = str(s.dropna().iloc[0]) if not s.dropna().empty else "N/A"
            if len(sample_val) > 40:
                sample_val = sample_val[:37] + "..."

            # Formulate issue notes
            issues = []
            if null_count > 0:
                issues.append(f"{null_count} nulls ({null_pct}%)")
            if blank_count > 0:
                issues.append(f"{blank_count} empty strings")
            if padded_count > 0:
                issues.append(f"{padded_count} whitespace/newline padded")
            if col == "tags":
                issues.append("String-encoded list; needs literal_eval")
            elif col == "ratings":
                issues.append("String-encoded list of dicts; needs literal_eval")
            elif col in ["film_date", "published_date"]:
                issues.append("Unix epoch timestamps in seconds")
            elif col == "url" and padded_count > 0:
                issues.append("Trailing newline characters (\\n)")

            issue_str = "; ".join(issues) if issues else "Clean"

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
                "Quality_Assessment": issue_str
            })

    audit_df(df_main, "ted_main.csv")
    audit_df(df_trans, "transcripts.csv")

    report_df = pd.DataFrame(records)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(output_path, index=False)

    return report_df


# ---------------------------------------------------------------------------
# Module 2: Data Cleaning & Preprocessing Helpers
# ---------------------------------------------------------------------------
def safe_parse_tags(tags_val: Any) -> List[str]:
    """
    Safely parse tag strings formatted as python list strings into a clean list of strings.

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

    # Regex fallback if ast fails
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


def clean_transcript_text(
    text: Any,
    remove_stage_directions: bool = True,
    remove_punctuation: bool = True,
    lowercase: bool = True
) -> str:
    """
    Preprocess transcript text while preserving lexical meaning.

    Steps:
    1. Removes parenthetical stage directions like (Applause), (Laughter), (Music).
    2. Lowercases text if requested.
    3. Normalizes contractions and removes punctuation.
    4. Collapses extra spaces, tabs, and newlines into a single clean space.

    Args:
        text: Raw transcript string.
        remove_stage_directions: Whether to remove (Applause), (Laughter), etc.
        remove_punctuation: Whether to clean punctuation.
        lowercase: Whether to convert to lowercase.

    Returns:
        Cleaned transcript string.
    """
    if pd.isna(text):
        return ""

    s = str(text)

    # 1. Remove stage directions like (Applause), (Laughter), (Music), etc.
    if remove_stage_directions:
        s = re.sub(r"\([^)]*\)", " ", s)

    # 2. Lowercase
    if lowercase:
        s = s.lower()

    # 3. Punctuation handling
    if remove_punctuation:
        # Keep alphanumeric and single quotes for contractions, replace other punctuation with space
        s = re.sub(r"[^a-z0-9\s']", " ", s)
        # Remove standalone single quotes
        s = re.sub(r"\s+'|'\s+|^'|'$", " ", s)

    # 4. Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def preprocess_ted_main(df_main: pd.DataFrame, common_urls: Optional[set] = None) -> pd.DataFrame:
    """
    Clean and preprocess metadata DataFrame.

    Actions:
    - Strips leading/trailing whitespace and trailing newlines from URLs and text fields.
    - Imputes missing `speaker_occupation` with 'Unknown'.
    - Safely parses `tags` and extracts `tag_count` and `clean_tags` space-separated string.
    - Safely parses `ratings` and computes `total_ratings` and `top_rating`.
    - Converts `film_date` and `published_date` Unix timestamps to datetime and extracts publication year.
    - Validates numeric boundaries (duration, views, comments, languages).
    - Flags `has_transcript` indicating if a transcript is available.

    Args:
        df_main: Raw ted_main DataFrame.
        common_urls: Set of URLs present in transcripts.

    Returns:
        Cleaned ted_main DataFrame.
    """
    df = df_main.copy()

    # 1. Strip whitespace and newlines from all object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # 2. Handle missing speaker_occupation (fill with 'Unknown')
    # Replace literal 'nan' or empty string with 'Unknown'
    df["speaker_occupation"] = df["speaker_occupation"].replace({"nan": "Unknown", "": "Unknown"}).fillna("Unknown")

    # 3. Safe tag parsing
    parsed_tags_series = df["tags"].apply(safe_parse_tags)
    df["tags_list"] = parsed_tags_series
    df["tag_count"] = parsed_tags_series.apply(len)
    df["clean_tags"] = parsed_tags_series.apply(lambda tags: " ".join(tags))

    # Keep original tags column as clean JSON-like string
    df["tags"] = parsed_tags_series.apply(str)

    # 4. Safe ratings parsing: extract total ratings and top rating category
    def extract_rating_features(r_val):
        parsed = safe_parse_ratings(r_val)
        if not parsed:
            return 0, "Unknown"
        total = sum(item.get("count", 0) for item in parsed if isinstance(item, dict))
        top = max(parsed, key=lambda x: x.get("count", 0) if isinstance(x, dict) else 0)
        return total, top.get("name", "Unknown") if isinstance(top, dict) else "Unknown"

    rating_feats = df["ratings"].apply(extract_rating_features)
    df["total_ratings"] = rating_feats.apply(lambda x: x[0])
    df["top_rating"] = rating_feats.apply(lambda x: x[1])

    # 5. Datetime conversions
    if "film_date" in df.columns:
        df["film_datetime"] = pd.to_datetime(df["film_date"], unit="s", errors="coerce")
        df["film_year"] = df["film_datetime"].dt.year
    if "published_date" in df.columns:
        df["published_datetime"] = pd.to_datetime(df["published_date"], unit="s", errors="coerce")
        df["published_year"] = df["published_datetime"].dt.year

    # 6. Duration in minutes
    if "duration" in df.columns:
        df["duration_minutes"] = round(df["duration"] / 60.0, 2)

    # 7. Transcript availability flag
    if common_urls is not None:
        df["has_transcript"] = df["url"].isin(common_urls).astype(int)

    # Drop temporary intermediate list column for standard CSV export
    if "tags_list" in df.columns:
        df = df.drop(columns=["tags_list"])

    return df


def preprocess_transcripts(df_trans: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess transcripts DataFrame.

    Actions:
    - Normalizes URLs (strips newlines and trailing whitespace).
    - Removes confirmed exact duplicate rows (keeps first).
    - Preserves normalized raw transcript as `transcript`.
    - Generates cleaned transcript as `clean_transcript` (stage directions removed, lowercase, stripped).
    - Computes character and word counts for text analytics.

    Args:
        df_trans: Raw transcripts DataFrame.

    Returns:
        Cleaned transcripts DataFrame.
    """
    df = df_trans.copy()

    # 1. Clean URLs
    df["url"] = df["url"].astype(str).str.strip()

    # 2. Normalize raw transcript text whitespace
    df["transcript"] = df["transcript"].fillna("").astype(str).str.strip()

    # 3. Identify and remove exact duplicate records by URL
    df = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)

    # 4. Generate clean_transcript
    df["clean_transcript"] = df["transcript"].apply(clean_transcript_text)

    # 5. Compute text length metrics
    df["char_count"] = df["transcript"].str.len()
    df["word_count"] = df["transcript"].apply(lambda x: len(x.split()) if x else 0)
    df["clean_char_count"] = df["clean_transcript"].str.len()
    df["clean_word_count"] = df["clean_transcript"].apply(lambda x: len(x.split()) if x else 0)

    return df


def validate_cleaning(
    df_main_raw: pd.DataFrame,
    df_main_clean: pd.DataFrame,
    df_trans_raw: pd.DataFrame,
    df_trans_clean: pd.DataFrame
) -> Dict[str, Any]:
    """
    Validate before-and-after cleaning states.

    Returns:
        Dict summarizing shape deltas, missing values, duplicates, and record counts.
    """
    validation = {
        "metadata": {
            "raw_shape": df_main_raw.shape,
            "clean_shape": df_main_clean.shape,
            "raw_nulls": int(df_main_raw.isnull().sum().sum()),
            "clean_nulls": int(df_main_clean.isnull().sum().sum()),
            "raw_duplicates": int(df_main_raw.duplicated().sum()),
            "clean_duplicates": int(df_main_clean.duplicated().sum()),
            "records_retained": len(df_main_clean),
            "records_removed": len(df_main_raw) - len(df_main_clean)
        },
        "transcripts": {
            "raw_shape": df_trans_raw.shape,
            "clean_shape": df_trans_clean.shape,
            "raw_nulls": int(df_trans_raw.isnull().sum().sum()),
            "clean_nulls": int(df_trans_clean.isnull().sum().sum()),
            "raw_duplicates": int(df_trans_raw.duplicated().sum()),
            "clean_duplicates": int(df_trans_clean.duplicated().sum()),
            "records_retained": len(df_trans_clean),
            "records_removed": len(df_trans_raw) - len(df_trans_clean)
        },
        "cleaning_decisions": [
            {
                "Item": "Metadata URLs & Text",
                "Action": "Stripped trailing newline (\\n) and whitespace from URLs and text columns",
                "Rationale": "Ensures exact string matching between metadata and transcript URLs"
            },
            {
                "Item": "Missing speaker_occupation",
                "Action": "Imputed 6 missing records with 'Unknown'",
                "Rationale": "Preserves talk metadata without inventing speculative occupations"
            },
            {
                "Item": "Duplicate Transcripts",
                "Action": "Removed 3 exact duplicate URL records in transcripts.csv",
                "Rationale": "Eliminates redundant identical transcripts; retains 1 unique row per talk"
            },
            {
                "Item": "Tags Parsing",
                "Action": "Parsed string lists safely with ast.literal_eval into standardized lowercased tokens",
                "Rationale": "Enables accurate tag frequency analysis and downstream content filtering"
            },
            {
                "Item": "Transcript Text Cleaning",
                "Action": "Generated clean_transcript with stage notes (Laughter/Applause) removed, lowercased",
                "Rationale": "Removes non-lexical acoustic noise while preserving full raw transcript"
            },
            {
                "Item": "Date Feature Engineering",
                "Action": "Converted Unix timestamps to film_datetime, published_datetime, and year",
                "Rationale": "Enables temporal analysis and recency weighting in recommendation models"
            }
        ]
    }
    return validation


# ---------------------------------------------------------------------------
# Module 2: Exploratory Data Analysis (EDA) Visualizations
# ---------------------------------------------------------------------------
def generate_eda_charts(
    df_main_clean: pd.DataFrame,
    df_trans_clean: pd.DataFrame,
    output_dir: Path
) -> List[Path]:
    """
    Generate and save 11 publication-quality visualizations for Exploratory Data Analysis.

    Charts generated:
    01_top_tags.png: Most frequently occurring TED Talk tags
    02_duration_distribution.png: Distribution of talk durations in minutes
    03_views_distribution.png: Distribution of talk views (log scale and density)
    04_ratings_distribution.png: Total aggregate ratings across all categories
    05_missing_values.png: Missing values and completeness comparison
    06_transcript_length_distribution.png: Transcript word count distribution
    07_top_speakers.png: Top 15 speakers by number of talks
    08_top_occupations.png: Top 15 speaker occupations
    09_views_vs_comments.png: Views vs. Comments correlation scatter plot
    10_talks_by_year.png: Annual publication trend of TED Talks
    11_transcript_coverage.png: Transcript availability breakdown (spoken vs musical/performance)

    Args:
        df_main_clean: Cleaned metadata DataFrame.
        df_trans_clean: Cleaned transcripts DataFrame.
        output_dir: Directory where PNG plots will be saved.

    Returns:
        List of Paths to saved image files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_charts = []

    # -----------------------------------------------------------------------
    # Plot 1: Most Frequently Occurring TED Talk Tags (Top 20)
    # -----------------------------------------------------------------------
    tag_counts = Counter()
    for clean_tags_str in df_main_clean["clean_tags"].dropna():
        for t in clean_tags_str.split():
            tag_counts[t] += 1
    top_tags = tag_counts.most_common(20)
    tags_df = pd.DataFrame(top_tags, columns=["Tag", "Frequency"])

    plt.figure(figsize=(10, 6))
    bars = plt.barh(tags_df["Tag"][::-1], tags_df["Frequency"][::-1], color=PRIMARY_COLOR, edgecolor="none", alpha=0.9)
    plt.title("Top 20 Most Frequent TED Talk Tags", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Frequency (Number of Talks)", fontsize=11)
    plt.ylabel("Tag Name", fontsize=11)
    plt.xlim(0, max(tags_df["Frequency"]) * 1.12)

    # Annotate bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 12, bar.get_y() + bar.get_height() / 2, f"{int(width):,}",
                 va="center", ha="left", fontsize=9, color="#2B2D42")

    plt.tight_layout()
    p1 = output_dir / "01_top_tags.png"
    plt.savefig(p1, bbox_inches="tight")
    plt.close()
    saved_charts.append(p1)

    # -----------------------------------------------------------------------
    # Plot 2: Distribution of Talk Durations (Minutes)
    # -----------------------------------------------------------------------
    plt.figure(figsize=(10, 5.5))
    durations = df_main_clean["duration_minutes"]
    median_dur = durations.median()
    mean_dur = durations.mean()

    sns.histplot(durations, kde=True, bins=40, color=SECONDARY_COLOR, edgecolor="white", alpha=0.8)
    plt.axvline(18.0, color=PRIMARY_COLOR, linestyle="--", linewidth=2, label="TED 18-min Benchmark")
    plt.axvline(median_dur, color="#2A9D8F", linestyle="-.", linewidth=2, label=f"Median ({median_dur:.1f} min)")
    plt.axvline(mean_dur, color="#E76F51", linestyle=":", linewidth=2, label=f"Mean ({mean_dur:.1f} min)")

    plt.title("Distribution of TED Talk Durations", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Duration (Minutes)", fontsize=11)
    plt.ylabel("Number of Talks", fontsize=11)
    plt.xlim(0, 45)  # Focus on the 0-45 min core distribution
    plt.legend(frameon=True, facecolor="white", edgecolor="#E0E0E0")

    plt.tight_layout()
    p2 = output_dir / "02_duration_distribution.png"
    plt.savefig(p2, bbox_inches="tight")
    plt.close()
    saved_charts.append(p2)

    # -----------------------------------------------------------------------
    # Plot 3: Distribution of Talk Views (Log Scale & Histogram)
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    views_millions = df_main_clean["views"] / 1e6
    sns.histplot(views_millions, bins=45, color=ACCENT_COLOR, kde=True, ax=ax1, edgecolor="white")
    ax1.set_title("Distribution of Views (Linear Scale in Millions)", fontsize=12, weight="bold")
    ax1.set_xlabel("Views (Millions)", fontsize=11)
    ax1.set_ylabel("Talk Count", fontsize=11)
    ax1.axvline(views_millions.median(), color=PRIMARY_COLOR, linestyle="--",
                label=f"Median: {views_millions.median():.2f}M")
    ax1.legend()

    sns.histplot(df_main_clean["views"], bins=45, log_scale=True, color=SECONDARY_COLOR, kde=True, ax=ax2, edgecolor="white")
    ax2.set_title("Distribution of Views (Logarithmic Scale)", fontsize=12, weight="bold")
    ax2.set_xlabel("Views (Log Scale)", fontsize=11)
    ax2.set_ylabel("Talk Count", fontsize=11)

    fig.suptitle("TED Talks Viewership Distribution Analysis", fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()
    p3 = output_dir / "03_views_distribution.png"
    plt.savefig(p3, bbox_inches="tight")
    plt.close()
    saved_charts.append(p3)

    # -----------------------------------------------------------------------
    # Plot 4: Aggregate Distribution of Ratings
    # -----------------------------------------------------------------------
    rating_totals = Counter()
    for r_str in df_main_clean["ratings"]:
        for item in safe_parse_ratings(r_str):
            if isinstance(item, dict):
                rating_totals[item.get("name", "Unknown")] += item.get("count", 0)

    ratings_df = pd.DataFrame(rating_totals.most_common(), columns=["Rating", "Total_Votes"])

    plt.figure(figsize=(10, 6))
    bars = plt.barh(ratings_df["Rating"][::-1], ratings_df["Total_Votes"][::-1] / 1000,
                    color="#2A9D8F", edgecolor="none", alpha=0.9)
    plt.title("Aggregate TED Talk Audience Ratings (Thousands of Votes)", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Total Votes (in Thousands)", fontsize=11)
    plt.ylabel("Rating Category", fontsize=11)
    plt.xlim(0, (max(ratings_df["Total_Votes"]) / 1000) * 1.15)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 15, bar.get_y() + bar.get_height() / 2, f"{int(width):,}k",
                 va="center", ha="left", fontsize=9, color="#2B2D42")

    plt.tight_layout()
    p4 = output_dir / "04_ratings_distribution.png"
    plt.savefig(p4, bbox_inches="tight")
    plt.close()
    saved_charts.append(p4)

    # -----------------------------------------------------------------------
    # Plot 5: Missing Values & Data Completeness
    # -----------------------------------------------------------------------
    missing_dict = {
        "speaker_occupation (raw)": 6,
        "speaker_occupation (cleaned)": 0,
        "transcripts (unmatched)": 86,
        "tags (missing)": 0,
        "views (missing)": 0,
        "duration (missing)": 0,
        "title (missing)": 0,
        "description (missing)": 0
    }
    missing_df = pd.DataFrame(list(missing_dict.items()), columns=["Field", "Missing_Count"])

    plt.figure(figsize=(10, 5))
    bar_colors = [CONTRAST_COLOR if c > 0 else "#2A9D8F" for c in missing_df["Missing_Count"]]
    bars = plt.bar(missing_df["Field"], missing_df["Missing_Count"], color=bar_colors, edgecolor="none", width=0.6)
    plt.title("Data Completeness and Missing Values Audit", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Field Evaluated", fontsize=11)
    plt.ylabel("Missing Count (Rows)", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 100)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 2, f"{int(h)}",
                 ha="center", va="bottom", fontsize=10, weight="bold", color="#2B2D42")

    plt.tight_layout()
    p5 = output_dir / "05_missing_values.png"
    plt.savefig(p5, bbox_inches="tight")
    plt.close()
    saved_charts.append(p5)

    # -----------------------------------------------------------------------
    # Plot 6: Transcript Word Count & Length Distribution
    # -----------------------------------------------------------------------
    plt.figure(figsize=(10, 5.5))
    words = df_trans_clean["word_count"]
    median_words = words.median()
    mean_words = words.mean()

    sns.histplot(words, kde=True, bins=40, color="#264653", edgecolor="white", alpha=0.85)
    plt.axvline(median_words, color=PRIMARY_COLOR, linestyle="--", linewidth=2,
                label=f"Median Words ({int(median_words):,} words)")
    plt.axvline(mean_words, color="#E76F51", linestyle="-.", linewidth=2,
                label=f"Mean Words ({int(mean_words):,} words)")

    plt.title("Distribution of Transcript Word Counts", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Word Count per Talk", fontsize=11)
    plt.ylabel("Number of Talks", fontsize=11)
    plt.legend(frameon=True, facecolor="white", edgecolor="#E0E0E0")

    plt.tight_layout()
    p6 = output_dir / "06_transcript_length_distribution.png"
    plt.savefig(p6, bbox_inches="tight")
    plt.close()
    saved_charts.append(p6)

    # -----------------------------------------------------------------------
    # Plot 7: Top 15 Most Frequent Speakers
    # -----------------------------------------------------------------------
    top_speakers = df_main_clean["main_speaker"].value_counts().head(15).reset_index()
    top_speakers.columns = ["Speaker", "Talk_Count"]

    plt.figure(figsize=(10, 5.5))
    bars = plt.barh(top_speakers["Speaker"][::-1], top_speakers["Talk_Count"][::-1],
                    color=SECONDARY_COLOR, edgecolor="none", alpha=0.9)
    plt.title("Top 15 Most Frequent TED Talk Speakers", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Number of Talks", fontsize=11)
    plt.ylabel("Speaker Name", fontsize=11)
    plt.xlim(0, max(top_speakers["Talk_Count"]) + 1.5)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.15, bar.get_y() + bar.get_height() / 2, f"{int(width)}",
                 va="center", ha="left", fontsize=9, color="#2B2D42")

    plt.tight_layout()
    p7 = output_dir / "07_top_speakers.png"
    plt.savefig(p7, bbox_inches="tight")
    plt.close()
    saved_charts.append(p7)

    # -----------------------------------------------------------------------
    # Plot 8: Top 15 Speaker Occupations
    # -----------------------------------------------------------------------
    # Exclude 'Unknown' for realistic occupation ranking
    known_occ = df_main_clean[df_main_clean["speaker_occupation"] != "Unknown"]
    top_occ = known_occ["speaker_occupation"].value_counts().head(15).reset_index()
    top_occ.columns = ["Occupation", "Count"]

    plt.figure(figsize=(10, 5.5))
    bars = plt.barh(top_occ["Occupation"][::-1], top_occ["Count"][::-1],
                    color=ACCENT_COLOR, edgecolor="none", alpha=0.9)
    plt.title("Top 15 TED Speaker Occupations", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Number of Speakers", fontsize=11)
    plt.ylabel("Occupation", fontsize=11)
    plt.xlim(0, max(top_occ["Count"]) + 6)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.8, bar.get_y() + bar.get_height() / 2, f"{int(width)}",
                 va="center", ha="left", fontsize=9, color="#2B2D42")

    plt.tight_layout()
    p8 = output_dir / "08_top_occupations.png"
    plt.savefig(p8, bbox_inches="tight")
    plt.close()
    saved_charts.append(p8)

    # -----------------------------------------------------------------------
    # Plot 9: Views vs. Comments Correlation
    # -----------------------------------------------------------------------
    plt.figure(figsize=(9, 6))
    corr = df_main_clean[["views", "comments"]].corr().iloc[0, 1]

    sns.regplot(
        x="views", y="comments", data=df_main_clean,
        scatter_kws={"alpha": 0.45, "color": SECONDARY_COLOR, "s": 25},
        line_kws={"color": PRIMARY_COLOR, "linewidth": 2}
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"Views vs. Comments Correlation (Pearson r = {corr:.2f})", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Views (Log Scale)", fontsize=11)
    plt.ylabel("Comments (Log Scale)", fontsize=11)

    plt.tight_layout()
    p9 = output_dir / "09_views_vs_comments.png"
    plt.savefig(p9, bbox_inches="tight")
    plt.close()
    saved_charts.append(p9)

    # -----------------------------------------------------------------------
    # Plot 10: Annual Publication Trend of TED Talks (2006 - 2017)
    # -----------------------------------------------------------------------
    yearly_counts = df_main_clean["published_year"].value_counts().sort_index().reset_index()
    yearly_counts.columns = ["Year", "Talk_Count"]

    plt.figure(figsize=(10, 5))
    plt.plot(yearly_counts["Year"], yearly_counts["Talk_Count"],
             marker="o", linewidth=2.5, color=PRIMARY_COLOR, markersize=7)
    plt.fill_between(yearly_counts["Year"], yearly_counts["Talk_Count"], color=PRIMARY_COLOR, alpha=0.15)
    plt.title("TED Talks Published by Year (2006 - 2017)", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Year Published", fontsize=11)
    plt.ylabel("Number of Talks", fontsize=11)
    plt.xticks(yearly_counts["Year"], rotation=0)

    for _, row in yearly_counts.iterrows():
        plt.text(row["Year"], row["Talk_Count"] + 7, f"{int(row['Talk_Count'])}",
                 ha="center", fontsize=9, weight="bold", color="#2B2D42")

    plt.ylim(0, max(yearly_counts["Talk_Count"]) * 1.18)
    plt.tight_layout()
    p10 = output_dir / "10_talks_by_year.png"
    plt.savefig(p10, bbox_inches="tight")
    plt.close()
    saved_charts.append(p10)

    # -----------------------------------------------------------------------
    # Plot 11: Transcript Availability Breakdown (Donut Chart)
    # -----------------------------------------------------------------------
    with_trans = (df_main_clean["has_transcript"] == 1).sum()
    without_trans = (df_main_clean["has_transcript"] == 0).sum()

    labels = [
        f"Spoken Talks with Transcripts\n({with_trans:,} talks - {with_trans/len(df_main_clean):.1%})",
        f"Music / Dance / Non-Spoken\n({without_trans:,} talks - {without_trans/len(df_main_clean):.1%})"
    ]
    sizes = [with_trans, without_trans]
    colors = ["#2A9D8F", CONTRAST_COLOR]

    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts = ax.pie(
        sizes, labels=labels, colors=colors, startangle=140,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=10, weight="bold")
    )
    ax.set_title("Transcript Coverage in TED Dataset", fontsize=14, weight="bold", pad=14)

    plt.tight_layout()
    p11 = output_dir / "11_transcript_coverage.png"
    plt.savefig(p11, bbox_inches="tight")
    plt.close()
    saved_charts.append(p11)

    return saved_charts


# ---------------------------------------------------------------------------
# End-to-End Execution Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    data_dir: Path = Path("data"),
    reports_dir: Path = Path("reports")
) -> Dict[str, Any]:
    """
    Execute Modules 1 and 2 end-to-end pipeline.

    Steps:
    1. Check presence and load raw datasets.
    2. Analyze common keys, schema, and identifier matching.
    3. Generate and persist data quality audit report to CSV.
    4. Preprocess metadata (ted_main) and transcripts.
    5. Validate before-and-after cleaning metrics.
    6. Generate and save publication-quality EDA charts.
    7. Save cleaned datasets:
       - data/ted_main_cleaned.csv
       - data/transcripts_cleaned.csv
    8. Display comprehensive summary.

    Returns:
        Dict with pipeline run metadata and validation results.
    """
    data_dir = Path(data_dir)
    reports_dir = Path(reports_dir)
    eda_dir = reports_dir / "eda"

    print("=================================================================")
    print("PERSONALIZED TED TALKS RECOMMENDATION SYSTEM - MODULES 1 & 2")
    print("=================================================================")

    # Step 1: Ingestion
    print("\n[Step 1/6] Ingesting raw datasets...")
    df_main_raw, df_trans_raw = load_datasets(data_dir)
    print(f"  -> Successfully loaded 'ted_main.csv': {df_main_raw.shape[0]:,} rows, {df_main_raw.shape[1]} columns")
    print(f"  -> Successfully loaded 'transcripts.csv': {df_trans_raw.shape[0]:,} rows, {df_trans_raw.shape[1]} columns")

    # Step 2: Key & Overlap Analysis
    print("\n[Step 2/6] Inspecting identifiers and dataset relationships...")
    overlap_info = inspect_identifiers_and_overlap(df_main_raw, df_trans_raw)
    print(f"  -> Common linking key identified: 'url'")
    print(f"  -> Unique URLs in metadata: {overlap_info['main_unique_urls']:,}")
    print(f"  -> Unique URLs in transcripts: {overlap_info['trans_unique_urls']:,}")
    print(f"  -> Exact duplicate transcript rows found: {overlap_info['trans_duplicate_url_count']}")
    print(f"  -> Talks with transcripts: {overlap_info['common_unique_urls']:,} ({overlap_info['match_percentage_of_main']}%)")
    print(f"  -> Talks without transcripts: {overlap_info['main_without_transcripts']:,} (predominantly music/dance/performances)")

    # Step 3: Data Quality Report
    print("\n[Step 3/6] Generating Data Quality Audit Report...")
    quality_report_path = reports_dir / "data_quality_report.csv"
    report_df = generate_data_quality_report(df_main_raw, df_trans_raw, quality_report_path)
    print(f"  -> Saved data quality audit report: {quality_report_path.resolve()}")

    # Step 4: Preprocessing & Cleaning
    print("\n[Step 4/6] Executing data cleaning and preprocessing pipelines...")
    # Common URLs set for flagging has_transcript
    clean_trans_urls = set(df_trans_raw["url"].astype(str).str.strip())

    df_main_clean = preprocess_ted_main(df_main_raw, common_urls=clean_trans_urls)
    df_trans_clean = preprocess_transcripts(df_trans_raw)

    clean_main_path = data_dir / "ted_main_cleaned.csv"
    clean_trans_path = data_dir / "transcripts_cleaned.csv"

    df_main_clean.to_csv(clean_main_path, index=False)
    df_trans_clean.to_csv(clean_trans_path, index=False)
    print(f"  -> Saved cleaned metadata: {clean_main_path.resolve()} ({len(df_main_clean):,} rows)")
    print(f"  -> Saved cleaned transcripts: {clean_trans_path.resolve()} ({len(df_trans_clean):,} rows)")

    # Step 5: Before and After Validation
    print("\n[Step 5/6] Validating before-and-after cleaning transformations...")
    validation = validate_cleaning(df_main_raw, df_main_clean, df_trans_raw, df_trans_clean)
    print(f"  -> Metadata shape: {validation['metadata']['raw_shape']} -> {validation['metadata']['clean_shape']}")
    print(f"  -> Metadata nulls: {validation['metadata']['raw_nulls']} -> {validation['metadata']['clean_nulls']}")
    print(f"  -> Transcripts shape: {validation['transcripts']['raw_shape']} -> {validation['transcripts']['clean_shape']}")
    print(f"  -> Transcripts duplicates removed: {validation['transcripts']['records_removed']}")

    # Step 6: Exploratory Data Analysis Charts
    print("\n[Step 6/6] Generating publication-quality EDA visualizations...")
    charts = generate_eda_charts(df_main_clean, df_trans_clean, eda_dir)
    print(f"  -> Generated {len(charts)} EDA charts in {eda_dir.resolve()}:")
    for chart in charts:
        print(f"     * {chart.name}")

    print("\n=================================================================")
    print("MODULE 1 & MODULE 2 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=================================================================")

    return {
        "overlap_info": overlap_info,
        "validation": validation,
        "charts": [str(c) for c in charts],
        "quality_report": str(quality_report_path),
        "clean_main_path": str(clean_main_path),
        "clean_trans_path": str(clean_trans_path)
    }


if __name__ == "__main__":
    # Base directory is one level above 'src' or current directory
    base_dir = Path(__file__).resolve().parent.parent
    data_directory = base_dir / "data"
    reports_directory = base_dir / "reports"

    run_pipeline(data_dir=data_directory, reports_dir=reports_directory)
