"""
Personalized TED Talks Recommendation System Using Machine Learning
Module 4: Recommendation Engine

Author: Leena Jarapala
Internship: Infosys Springboard Virtual Internship 7.0
GitHub: https://github.com/leena-11/ted-talks-recommendation-system

Description:
This module implements the content-based recommendation engine using TF-IDF feature
vectors and cosine similarity, supports title and keyword searching, excludes the
queried talk, formats rich metadata, handles edge cases, and provides a hybrid
architecture interface for future collaborative filtering extensions.
"""

import os
import sys
import re
import difflib
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union

import pandas as pd
import numpy as np
import joblib
from scipy import sparse
from sklearn.metrics.pairwise import linear_kernel


class TEDRecommendationEngine:
    """
    Production-ready content-based recommendation engine for TED Talks.
    """

    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        models_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize and load dataset and model artifacts.

        Args:
            data_path: Path to cleaned_ted_talks.csv.
            models_dir: Directory containing tfidf_vectorizer.pkl, tfidf_matrix.npz, talk_index_mapping.csv.
        """
        base_dir = Path(__file__).resolve().parent

        # Resolve data path
        if data_path is None:
            candidate_paths = [
                base_dir / "cleaned_ted_talks.csv",
                base_dir / "data" / "cleaned_ted_talks.csv"
            ]
            for cp in candidate_paths:
                if cp.is_file():
                    data_path = cp
                    break

        if data_path is None or not Path(data_path).is_file():
            raise FileNotFoundError(
                f"Cleaned dataset not found at {data_path}. Please execute 'preprocess_ted_data.py' first."
            )
        self.data_path = Path(data_path)

        # Resolve models dir
        if models_dir is None:
            models_dir = base_dir / "models"
        self.models_dir = Path(models_dir)

        self._load_artifacts()
        self._build_search_indices()

    def _load_artifacts(self):
        """Load saved TF-IDF vectorizer, sparse feature matrix, and dataset."""
        vec_path = self.models_dir / "tfidf_vectorizer.pkl"
        mat_path = self.models_dir / "tfidf_matrix.npz"
        map_path = self.models_dir / "talk_index_mapping.csv"

        if not (vec_path.is_file() and mat_path.is_file()):
            raise FileNotFoundError(
                f"Model artifacts missing in {self.models_dir}. Please execute 'feature_engineering.py' first."
            )

        self.vectorizer = joblib.load(vec_path)
        self.tfidf_matrix = sparse.load_npz(mat_path)
        self.df = pd.read_csv(self.data_path)

        if len(self.df) != self.tfidf_matrix.shape[0]:
            raise ValueError(
                f"Dataset row count ({len(self.df)}) does not match TF-IDF matrix rows ({self.tfidf_matrix.shape[0]})."
            )

    def _build_search_indices(self):
        """Build exact and normalized index lookups for responsive title searching."""
        self.title_to_idx: Dict[str, int] = {}
        self.normalized_to_idx: Dict[str, int] = {}
        self.all_titles: List[str] = []

        for idx, row in self.df.iterrows():
            title = str(row["title"]).strip()
            self.all_titles.append(title)
            # Exact lowercase lookup
            self.title_to_idx[title.lower()] = idx
            # Normalized alphanumeric lookup
            norm = re.sub(r"[^a-z0-9]", "", title.lower())
            if norm:
                self.normalized_to_idx[norm] = idx

    def find_talk_index(self, query_title: str) -> int:
        """
        Locate talk index using exact, normalized, or fuzzy matching.

        Args:
            query_title: Title string searched by user.

        Returns:
            Integer row index of the matched talk.

        Raises:
            ValueError: If input is invalid, blank, or cannot be resolved to any talk.
        """
        if not query_title or not isinstance(query_title, str) or not query_title.strip():
            raise ValueError("Talk title cannot be empty or blank. Please provide a valid TED Talk title.")

        q_clean = query_title.strip()
        q_low = q_clean.lower()

        # 1. Exact case-insensitive match
        if q_low in self.title_to_idx:
            return self.title_to_idx[q_low]

        # 2. Normalized match (punctuation stripped)
        q_norm = re.sub(r"[^a-z0-9]", "", q_low)
        if q_norm in self.normalized_to_idx:
            return self.normalized_to_idx[q_norm]

        # 3. Substring matching (if unique or dominant)
        substring_matches = [
            title for title in self.all_titles
            if q_low in title.lower()
        ]
        if len(substring_matches) == 1:
            return self.title_to_idx[substring_matches[0].lower()]
        elif len(substring_matches) > 1:
            # If multiple substring matches, pick the one with closest length or first
            best_match = min(substring_matches, key=lambda t: abs(len(t) - len(q_clean)))
            return self.title_to_idx[best_match.lower()]

        # 4. Fuzzy candidate suggestions using difflib
        close_matches = difflib.get_close_matches(q_clean, self.all_titles, n=3, cutoff=0.5)
        suggestions_str = ""
        if close_matches:
            suggestions_str = f" Did you mean:\n  - " + "\n  - ".join(close_matches)

        raise ValueError(
            f"Talk title '{query_title}' not found in the TED Talks dataset.{suggestions_str}"
        )

    def recommend_talks(self, title: str, top_n: int = 5) -> pd.DataFrame:
        """
        Generate top-N content-based recommendations for a selected TED Talk.

        Args:
            title: Title of the TED Talk to search.
            top_n: Number of recommendations to return (default: 5).

        Returns:
            DataFrame containing recommended talks with complete metadata and similarity scores.
        """
        if top_n <= 0:
            raise ValueError(f"top_n must be a positive integer, received: {top_n}")

        talk_idx = self.find_talk_index(title)
        query_title = self.df.loc[talk_idx, "title"]

        # Compute cosine similarity with all talks
        query_vec = self.tfidf_matrix[talk_idx]
        sims = linear_kernel(query_vec, self.tfidf_matrix).flatten()

        # Strictly exclude the query talk itself
        sims[talk_idx] = -1.0

        # Retrieve top-N highest scoring talks
        top_indices = np.argsort(sims)[::-1][:top_n]

        results = []
        for rank, match_idx in enumerate(top_indices, 1):
            score = float(sims[match_idx])
            rec_row = self.df.iloc[match_idx]

            # Safe metadata retrieval
            speaker = rec_row.get("main_speaker", "Unknown Speaker")
            tags = rec_row.get("clean_tags", rec_row.get("tags", ""))
            desc = rec_row.get("description", "")
            url = rec_row.get("url", "")
            duration = rec_row.get("duration_minutes", 0.0)
            views = rec_row.get("views", 0)

            results.append({
                "rank": rank,
                "title": rec_row["title"],
                "main_speaker": speaker,
                "clean_tags": tags,
                "description": desc,
                "url": url,
                "similarity_score": round(score, 4),
                "similarity_pct": f"{score * 100:.1f}%",
                "duration_minutes": duration,
                "views": views
            })

        return pd.DataFrame(results)

    def recommend_from_text(self, text_query: str, top_n: int = 5) -> pd.DataFrame:
        """
        Generate top-N recommendations from a free-form keyword or thematic query.

        Args:
            text_query: Free-form text query (e.g. 'artificial intelligence in healthcare').
            top_n: Number of recommendations to return.

        Returns:
            DataFrame containing recommended talks.
        """
        if not text_query or not isinstance(text_query, str) or not text_query.strip():
            raise ValueError("Query text cannot be empty or blank.")

        query_vec = self.vectorizer.transform([text_query])
        sims = linear_kernel(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(sims)[::-1][:top_n]

        results = []
        for rank, match_idx in enumerate(top_indices, 1):
            score = float(sims[match_idx])
            rec_row = self.df.iloc[match_idx]

            results.append({
                "rank": rank,
                "title": rec_row["title"],
                "main_speaker": rec_row.get("main_speaker", "Unknown"),
                "clean_tags": rec_row.get("clean_tags", ""),
                "description": rec_row.get("description", ""),
                "url": rec_row.get("url", ""),
                "similarity_score": round(score, 4),
                "similarity_pct": f"{score * 100:.1f}%",
                "duration_minutes": rec_row.get("duration_minutes", 0.0),
                "views": rec_row.get("views", 0)
            })

        return pd.DataFrame(results)

    def hybrid_recommendation_interface(
        self,
        title: str,
        user_id: Optional[str] = None,
        top_n: int = 5,
        alpha: float = 0.8
    ) -> pd.DataFrame:
        """
        Architectural interface for hybrid recommendation (Content-Based + Collaborative Filtering).

        Note:
        Genuine individualized user watch histories are absent from the public TED Talks
        dataset. When user logs are available, this method computes:
            Hybrid_Score = alpha * Content_Score + (1 - alpha) * Collaborative_Score

        In the current release, it safely routes to content-based recommendation while
        documenting the collaborative filtering limitation without fabricating ratings.
        """
        return self.recommend_talks(title=title, top_n=top_n)


# ---------------------------------------------------------------------------
# Global Singleton Cache & Reusable Function
# ---------------------------------------------------------------------------
_GLOBAL_ENGINE: Optional[TEDRecommendationEngine] = None


def get_recommendation_engine(
    data_path: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None
) -> TEDRecommendationEngine:
    """Get or initialize singleton recommendation engine."""
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = TEDRecommendationEngine(data_path=data_path, models_dir=models_dir)
    return _GLOBAL_ENGINE


def recommend_talks(
    title: str,
    top_n: int = 5,
    data_path: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Reusable standalone function to recommend talks by title.

    Args:
        title: Title of TED Talk.
        top_n: Number of recommendations (default: 5).
        data_path: Optional custom dataset path.
        models_dir: Optional custom models directory.

    Returns:
        DataFrame with recommendation results.
    """
    engine = get_recommendation_engine(data_path=data_path, models_dir=models_dir)
    return engine.recommend_talks(title=title, top_n=top_n)


def display_recommendations(query: str, results_df: pd.DataFrame):
    """Format and print recommendations nicely in the console."""
    print("\n" + "=" * 78)
    print(f"RECOMMENDATIONS FOR: \"{query}\"")
    print("=" * 78)
    if results_df.empty:
        print("No recommendations found.")
        return

    for _, row in results_df.iterrows():
        print(f"\n#{row['rank']} | {row['title']} ({row['similarity_pct']} match)")
        print(f"   Speaker:     {row['main_speaker']}")
        print(f"   Topics/Tags: {row['clean_tags'][:65]}...")
        desc = row['description'] if len(row['description']) < 120 else row['description'][:117] + "..."
        print(f"   Description: {desc}")
        print(f"   URL:         {row['url']}")

    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="TED Talks Recommendation Engine (Module 4)")
    parser.add_argument("--title", type=str, help="TED Talk title to find recommendations for")
    parser.add_argument("--keyword", type=str, help="Free-form keyword query to search")
    parser.add_argument("--top_n", type=int, default=5, help="Number of recommendations (default: 5)")
    args = parser.parse_args()

    engine = get_recommendation_engine()

    if args.title:
        try:
            recs = engine.recommend_talks(args.title, top_n=args.top_n)
            display_recommendations(args.title, recs)
        except Exception as e:
            print(f"[Error] {e}")
    elif args.keyword:
        try:
            recs = engine.recommend_from_text(args.keyword, top_n=args.top_n)
            display_recommendations(f"Keyword Query: '{args.keyword}'", recs)
        except Exception as e:
            print(f"[Error] {e}")
    else:
        # Run demo on landmark talks
        print("======================================================================")
        print("MODULE 4: TED TALKS RECOMMENDATION ENGINE - DEMO")
        print("======================================================================")
        demo_queries = [
            "Do schools kill creativity?",
            "Your body language may shape who you are",
            "Averting the climate crisis"
        ]
        for query in demo_queries:
            recs = engine.recommend_talks(query, top_n=3)
            display_recommendations(query, recs)

        print("\nEngine is ready for interactive and programmatic use.")
        print("Usage examples:")
        print("  python recommendation_engine.py --title \"Do schools kill creativity?\" --top_n 5")
        print("  python recommendation_engine.py --keyword \"artificial intelligence in medicine\" --top_n 5")
        print("======================================================================")


if __name__ == "__main__":
    main()
