"""
Personalized TED Talks Recommendation System Using Machine Learning
Module 1: Data Collection & Ingestion Pipeline

Author: Leena Jarapala
Internship: Infosys Springboard Virtual Internship 7.0
GitHub: https://github.com/leena-11/ted-talks-recommendation-system

Description:
This module verifies, ingests, and merges raw TED Talks datasets into a consolidated,
structured dataset ready for preprocessing and feature engineering.
"""

import os
import sys
import ast
import re
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np


# Candidate filenames for TED Talks datasets
METADATA_CANDIDATE_NAMES = [
    "ted_main.csv",
    "ted_talks_dataset.csv",
    "ted_talks_dataset.csv.csv",
    "ted_talks_en.csv",
    "collected_ted_talks.csv"
]

TRANSCRIPT_CANDIDATE_NAMES = [
    "transcripts.csv",
    "ted_transcripts.csv",
    "transcripts_cleaned.csv"
]


def find_file_in_paths(candidate_names: List[str], search_dirs: List[Path]) -> Optional[Path]:
    """
    Search for the first matching candidate file across specified directories.

    Args:
        candidate_names: List of acceptable filenames.
        search_dirs: Directories to search within.

    Returns:
        Path to existing file or None if not found.
    """
    for d in search_dirs:
        for name in candidate_names:
            candidate_path = d / name
            if candidate_path.is_file() and candidate_path.stat().st_size > 0:
                return candidate_path
    return None


def locate_datasets(base_dir: Optional[Path] = None) -> Dict[str, Optional[Path]]:
    """
    Locate available metadata and transcript datasets across the workspace.

    Args:
        base_dir: Base directory to search (defaults to project root).

    Returns:
        Dict with keys 'metadata' and 'transcript' mapping to resolved Paths.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent

    search_dirs = [
        base_dir / "data",
        base_dir / "data" / "raw",
        base_dir,
    ]

    metadata_path = find_file_in_paths(METADATA_CANDIDATE_NAMES, search_dirs)
    transcript_path = find_file_in_paths(TRANSCRIPT_CANDIDATE_NAMES, search_dirs)

    return {
        "metadata": metadata_path,
        "transcript": transcript_path
    }


def inspect_topic_diversity(df: pd.DataFrame, tags_col: str = "tags") -> Dict[str, int]:
    """
    Analyze the distribution and diversity of broad categories in the dataset.

    Args:
        df: DataFrame containing TED Talks data.
        tags_col: Column name containing talk tags.

    Returns:
        Dict of category counts for core TED topic domains.
    """
    if tags_col not in df.columns:
        return {}

    core_categories = [
        "technology", "science", "global issues", "culture", "design",
        "business", "education", "health", "psychology", "art",
        "entertainment", "creativity", "society", "communication"
    ]
    counts = {cat: 0 for cat in core_categories}

    for tags_val in df[tags_col].dropna():
        tags_str = str(tags_val).lower()
        for cat in core_categories:
            if cat in tags_str:
                counts[cat] += 1

    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))


def load_and_merge_datasets(
    metadata_path: Path,
    transcript_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load raw metadata and transcripts, resolve identifier anomalies (e.g., trailing newlines),
    deduplicate transcript records, and perform a left-join on 'url'.

    Args:
        metadata_path: Path to metadata CSV file.
        transcript_path: Optional path to transcript CSV file.

    Returns:
        Tuple of (merged_dataframe, audit_summary_dict).
    """
    print(f"[Module 1] Ingesting metadata from: {metadata_path}")
    df_meta = pd.read_csv(metadata_path)
    total_meta_records = len(df_meta)

    # Normalize url identifier column in metadata
    if "url" in df_meta.columns:
        df_meta["url"] = df_meta["url"].astype(str).str.strip()
    else:
        raise ValueError(f"Expected linking identifier 'url' missing from metadata in {metadata_path}")

    # Check if transcripts are already merged into this file
    if "transcript" in df_meta.columns and transcript_path is None:
        print("[Module 1] Transcript column already present in unified dataset.")
        merged_df = df_meta.copy()
        trans_records = (merged_df["transcript"].notnull() & (merged_df["transcript"].str.strip() != "")).sum()
        dup_trans = 0
    elif transcript_path is not None and transcript_path.is_file():
        print(f"[Module 1] Ingesting transcripts from: {transcript_path}")
        df_trans = pd.read_csv(transcript_path)
        trans_records = len(df_trans)

        # Normalize url in transcripts (stripping trailing whitespace and newlines)
        if "url" in df_trans.columns:
            df_trans["url"] = df_trans["url"].astype(str).str.strip()
        else:
            raise ValueError(f"Expected linking identifier 'url' missing from transcripts in {transcript_path}")

        # Count and drop duplicate transcript rows by URL
        dup_trans = int(df_trans.duplicated(subset=["url"]).sum())
        if dup_trans > 0:
            print(f"[Module 1] Found {dup_trans} duplicate transcript records; removing duplicates (keeping first).")
            df_trans = df_trans.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)

        # Merge metadata with transcripts via left-join
        merged_df = pd.merge(df_meta, df_trans[["url", "transcript"]], on="url", how="left")
    else:
        print("[Module 1] No separate transcript file found; proceeding with metadata only.")
        merged_df = df_meta.copy()
        merged_df["transcript"] = ""
        trans_records = 0
        dup_trans = 0

    # Fill missing transcripts with empty string for safe text processing
    has_trans_count = int((merged_df["transcript"].notnull() & (merged_df["transcript"].astype(str).str.strip() != "")).sum())
    missing_trans_count = total_meta_records - has_trans_count

    # Evaluate topic diversity
    topic_diversity = inspect_topic_diversity(merged_df, "tags")

    audit_summary = {
        "metadata_source": str(metadata_path),
        "transcript_source": str(transcript_path) if transcript_path else "None",
        "total_talks": total_meta_records,
        "unique_urls": merged_df["url"].nunique(),
        "talks_with_transcript": has_trans_count,
        "talks_without_transcript": missing_trans_count,
        "duplicate_transcripts_resolved": dup_trans,
        "core_topic_coverage": topic_diversity,
        "available_columns": list(merged_df.columns),
        "user_ratings_data": "Unavailable (Public TED dataset provides aggregate talk-level community ratings, views, and comments; individual user watch logs are not present in the genuine dataset)."
    }

    return merged_df, audit_summary


def collect_ted_data(
    output_path: Optional[Path] = None,
    base_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute Module 1 Data Collection pipeline end-to-end.

    Args:
        output_path: Optional path to save the consolidated collection CSV.
        base_dir: Optional workspace directory.

    Returns:
        Tuple of (collected_dataframe, audit_summary_dict).
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent

    paths = locate_datasets(base_dir)
    metadata_path = paths["metadata"]
    transcript_path = paths["transcript"]

    if metadata_path is None:
        raise FileNotFoundError(
            f"Could not find any TED Talks dataset in {base_dir / 'data'}. "
            f"Expected one of: {METADATA_CANDIDATE_NAMES}"
        )

    merged_df, audit = load_and_merge_datasets(metadata_path, transcript_path)

    if output_path is None:
        output_path = base_dir / "data" / "collected_ted_talks.csv"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    print(f"[Module 1] Saved consolidated collection to: {output_path} ({len(merged_df):,} talks)")

    return merged_df, audit


def main():
    print("=" * 70)
    print("MODULE 1: TED TALKS DATA COLLECTION & INGESTION")
    print("=" * 70)

    base_dir = Path(__file__).resolve().parent
    df, audit = collect_ted_data(base_dir=base_dir)

    print("\n[Summary of Collected Dataset]")
    print(f" - Total Talks Collected:         {audit['total_talks']:,}")
    print(f" - Unique Talk URLs:              {audit['unique_urls']:,}")
    print(f" - Talks With Transcripts:        {audit['talks_with_transcript']:,} ({(audit['talks_with_transcript']/audit['total_talks'])*100:.2f}%)")
    print(f" - Talks Without Transcripts:     {audit['talks_without_transcript']:,} (predominantly music/dance/performances)")
    print(f" - Duplicate Transcripts Removed: {audit['duplicate_transcripts_resolved']}")
    print(f" - User Interaction Logs:         {audit['user_ratings_data']}")

    print("\n[Topic Diversity Across Key Domains]")
    for topic, count in list(audit["core_topic_coverage"].items())[:8]:
        print(f" - {topic.title():<20} : {count:>5} talks")

    print("\n[Dataset Columns]")
    print(", ".join(audit["available_columns"]))
    print("=" * 70)
    print("MODULE 1 EXECUTION COMPLETE - READY FOR PREPROCESSING")
    print("=" * 70)


if __name__ == "__main__":
    main()
