"""
Automated Test Suite for Module 4: TED Talks Recommendation Engine

Author: Leena Jarapala
Internship: Infosys Springboard Virtual Internship 7.0
GitHub: https://github.com/leena-11/ted-talks-recommendation-system
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from recommendation_engine import (
    TEDRecommendationEngine,
    get_recommendation_engine,
    recommend_talks
)


@pytest.fixture(scope="module")
def engine():
    """Fixture providing initialized TED recommendation engine."""
    return get_recommendation_engine()


class TestRecommendationArtifacts:
    """Verify presence and validity of required model artifacts and datasets."""

    def test_artifacts_exist(self):
        base_dir = Path(__file__).resolve().parent
        data_p = base_dir / "cleaned_ted_talks.csv"
        vec_p = base_dir / "models" / "tfidf_vectorizer.pkl"
        mat_p = base_dir / "models" / "tfidf_matrix.npz"
        map_p = base_dir / "models" / "talk_index_mapping.csv"

        assert data_p.is_file(), f"Missing dataset: {data_p}"
        assert vec_p.is_file(), f"Missing TF-IDF vectorizer: {vec_p}"
        assert mat_p.is_file(), f"Missing TF-IDF matrix: {mat_p}"
        assert map_p.is_file(), f"Missing talk index mapping: {map_p}"

    def test_matrix_dimensions_match_dataset(self, engine):
        assert len(engine.df) == engine.tfidf_matrix.shape[0]
        assert engine.tfidf_matrix.shape[0] == 2550
        assert engine.tfidf_matrix.shape[1] == 5000


class TestRecommendationOutputs:
    """Verify recommendation generation, metadata completeness, and ranking order."""

    def test_valid_talk_recommendations(self, engine):
        query = "Do schools kill creativity?"
        recs = engine.recommend_talks(query, top_n=5)

        assert isinstance(recs, pd.DataFrame)
        assert len(recs) == 5

        expected_cols = [
            "rank", "title", "main_speaker", "clean_tags",
            "description", "url", "similarity_score", "similarity_pct"
        ]
        for col in expected_cols:
            assert col in recs.columns, f"Expected column '{col}' missing from recommendations"

        # Assert scores are between 0 and 1
        scores = recs["similarity_score"].tolist()
        assert all(0.0 <= s <= 1.0 for s in scores), f"Similarity scores outside [0, 1]: {scores}"

        # Assert strictly sorted in descending order
        assert scores == sorted(scores, reverse=True), "Recommendations not sorted by similarity descending"

    def test_exclude_selected_talk(self, engine):
        query = "Do schools kill creativity?"
        recs = engine.recommend_talks(query, top_n=10)

        recommended_titles = recs["title"].str.lower().tolist()
        assert query.lower() not in recommended_titles, "Queried talk should never appear in its own recommendations"

    @pytest.mark.parametrize("top_n", [1, 3, 5, 8, 10])
    def test_correct_number_of_recommendations(self, engine, top_n):
        query = "Your body language may shape who you are"
        recs = engine.recommend_talks(query, top_n=top_n)
        assert len(recs) == top_n

    def test_standalone_recommend_talks_function(self):
        recs = recommend_talks("Averting the climate crisis", top_n=4)
        assert len(recs) == 4
        assert "Averting the climate crisis".lower() not in recs["title"].str.lower().tolist()


class TestEdgeCasesAndInputs:
    """Verify handling of invalid, empty, or unknown inputs."""

    def test_unknown_title_raises_value_error(self, engine):
        fake_title = "Totally Nonexistent Talk 987654321 XYZ"
        with pytest.raises(ValueError) as excinfo:
            engine.recommend_talks(fake_title)
        assert "not found" in str(excinfo.value).lower()

    @pytest.mark.parametrize("empty_query", ["", "   ", None])
    def test_empty_or_blank_input_raises_value_error(self, engine, empty_query):
        with pytest.raises(ValueError):
            engine.recommend_talks(empty_query)

    def test_case_insensitive_matching(self, engine):
        recs_lower = engine.recommend_talks("do schools kill creativity?", top_n=3)
        recs_upper = engine.recommend_talks("DO SCHOOLS KILL CREATIVITY?", top_n=3)

        assert recs_lower["title"].tolist() == recs_upper["title"].tolist()
        assert recs_lower["similarity_score"].tolist() == recs_upper["similarity_score"].tolist()

    def test_partial_title_matching(self, engine):
        # "body language" should match "Your body language may shape who you are"
        recs = engine.recommend_talks("body language", top_n=3)
        assert len(recs) == 3
        # Should not recommend the matched talk itself
        assert "Your body language may shape who you are".lower() not in recs["title"].str.lower().tolist()

    def test_talk_without_transcript_generates_recommendations(self, engine):
        """Talk without transcript (e.g. musical/dance performance) must still produce valid recommendations."""
        perf_title = 'A dance of "Symbiosis"'
        assert perf_title in engine.all_titles

        recs = engine.recommend_talks(perf_title, top_n=5)
        assert len(recs) == 5
        assert perf_title.lower() not in recs["title"].str.lower().tolist()
        assert all(s >= 0.0 for s in recs["similarity_score"])

    def test_free_text_keyword_search(self, engine):
        recs = engine.recommend_from_text("machine learning artificial intelligence robotics", top_n=5)
        assert len(recs) == 5
        assert all("title" in recs.columns and "similarity_score" in recs.columns for _ in [0])
