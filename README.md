# Personalized TED Talks Recommendation System Using Machine Learning

A machine-learning recommendation system developed as part of the **Infosys Springboard Virtual Internship 7.0**. The platform enables users to discover highly relevant TED Talks through automated content-based filtering across talk transcripts, titles, thematic tags, speaker profiles, and descriptions.

- **Author:** Leena Jarapala  
- **Degree:** B.Tech — Artificial Intelligence and Data Science  
- **Institution:** Velagapudi Ramakrishna Siddhartha Engineering College, Vijayawada  
- **Internship:** Infosys Springboard Virtual Internship 7.0  
- **GitHub Repository:** [https://github.com/leena-11/ted-talks-recommendation-system](https://github.com/leena-11/ted-talks-recommendation-system)  
- **Project Status:** **Modules 1, 2, 3, and 4 Fully Implemented, Tested, and Verified.**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Module Architecture & Lifecycle](#module-architecture--lifecycle)
   - [Module 1: Data Collection & Ingestion](#module-1-data-collection--ingestion)
   - [Module 2: Data Exploration & Preprocessing](#module-2-data-exploration--preprocessing)
   - [Module 3: Feature Engineering & Vectorization](#module-3-feature-engineering--vectorization)
   - [Module 4: Recommendation Engine & Automated Testing](#module-4-recommendation-engine--automated-testing)
3. [Technology Stack](#technology-stack)
4. [Dataset Description & Provenance](#dataset-description--provenance)
5. [Repository Structure](#repository-structure)
6. [Setup and Execution Guide (Windows PowerShell)](#setup-and-execution-guide-windows-powershell)
7. [Verified Recommendation Examples](#verified-recommendation-examples)
8. [Automated Test Suite](#automated-test-suite)
9. [Limitations & Future Roadmap](#limitations--future-roadmap)
10. [License & Attribution](#license--attribution)

---

## Project Overview

TED Talks encompass diverse disciplines including technology, science, education, psychology, design, global issues, business, and human culture. With thousands of talks available, finding content tailored to specific intellectual interests or research topics is challenging.

This project delivers a modular machine-learning pipeline that collects, preprocesses, vectorizes, and recommends TED Talks based on genuine content similarity:
- **Module 1 (Data Collection):** Collects, cleans identifiers, and unifies talk metadata with transcripts into a verified collection of 2,550 talks.
- **Module 2 (Data Exploration & Preprocessing):** Performs text normalization (stopword filtering, punctuation removal, case normalization), handles missing values, calculates token metrics, audits data quality, and produces publication-grade EDA visualizations.
- **Module 3 (Feature Engineering & Vectorization):** Constructs a weighted composite document representation, learns a 5,000-dimensional TF-IDF feature space (unigrams and bigrams with sublinear term-frequency scaling), and stores sparse feature matrices.
- **Module 4 (Recommendation Engine & Testing):** Implements content-based cosine similarity retrieval with title search, free-text query support, self-exclusion, input validation, and an 18-test automated pytest verification suite.

---

## Module Architecture & Lifecycle

### Module 1: Data Collection & Ingestion
- **Script:** [`data_collection.py`](data_collection.py)
- **Functions:** `locate_datasets()`, `load_and_merge_datasets()`, `inspect_topic_diversity()`, `collect_ted_data()`
- **Key Actions:**
  - Auto-discovers metadata (`ted_main.csv`, `ted_talks_dataset.csv`) and transcripts (`transcripts.csv`) in `data/`, `data/raw/`, or root.
  - Normalizes URLs by stripping trailing newline characters (`\n`) and whitespace.
  - Identifies and removes duplicate transcript records (3 duplicate URLs safely dropped, keeping first).
  - Merges talk metadata with transcripts on `url` using a reliable left-join.
  - Preserves 2,550 total talks (2,464 talks with transcripts; 86 non-spoken performances without transcripts safely flagged).
  - Inspects broad topic diversity across core domains (Technology: 727, Science: 604, Culture: 532, Global Issues: 501, Design: 449, Business: 355, Health: 321).
  - Documents that genuine individual user watch histories and user-level ratings are not part of the public dataset; avoids fabricating synthetic user ratings.
- **Output:** `data/collected_ted_talks.csv`

### Module 2: Data Exploration & Preprocessing
- **Script:** [`preprocess_ted_data.py`](preprocess_ted_data.py)
- **Functions:** `clean_text()`, `safe_parse_tags()`, `safe_parse_ratings()`, `generate_quality_report()`, `preprocess_data()`, `generate_visualizations()`
- **Key Actions:**
  - Ingests verified collection of 2,550 records without mutating raw source files.
  - Normalizes text: lowercasing, stage direction removal (e.g., `(Applause)`, `(Laughter)`), punctuation stripping, and careful English stopword removal using scikit-learn's curated vocabulary.
  - Preserves original raw text in separate columns (`title`, `tags`, `description`, `transcript`) while generating cleaned counterparts (`clean_title`, `clean_tags`, `clean_description`, `clean_transcript`).
  - Imputes missing `speaker_occupation` (6 nulls / 0.235%) with `'Unknown'`.
  - Safely handles 86 non-spoken/musical talks by populating clean transcripts with empty strings while retaining descriptive metadata.
  - Encodes categorical features (`speaker_occupation_code`, `top_rating_code`).
  - Computes exact text length and token count metrics:
    - `title_word_count`
    - `description_word_count`
    - `transcript_word_count` (mean: 863.5 cleaned words, median: 862 words)
    - `combined_text_word_count`
    - `combined_text_char_count`
  - Generates publication-grade visualizations saved to `visualizations/`:
    - `visualizations/topic_frequency.png` (Top 20 most frequent topics and tags)
    - `visualizations/transcript_word_count_distribution.png` (Distribution of transcript word lengths with median and mean markers)
- **Outputs:**
  - `cleaned_ted_talks.csv` (2,550 records, 37 columns)
  - `data_quality_report.csv` & `preprocessing_quality_report.csv`
  - `preprocessing_summary.csv`
  - `visualizations/topic_frequency.png`
  - `visualizations/transcript_word_count_distribution.png`

### Module 3: Feature Engineering & Vectorization
- **Script:** [`feature_engineering.py`](feature_engineering.py)
- **Functions:** `build_combined_content()`, `engineer_features()`, `save_artifacts()`, `compute_talk_similarities()`, `generate_sample_similarity_report()`, `plot_similarity_analytics()`
- **Hyperparameter Choices & Rationale:**
  - **Feature Weighting:** Combines title (2x weight), tags (2x weight), description (1x weight), and cleaned transcript (1x weight). Weighting title and tags gives strong thematic keywords proportional salience against long conversational transcripts.
  - **Vocabulary Cap (`max_features=5000`):** Retains the top 5,000 most informative terms, balancing discriminative power with computational efficiency.
  - **N-gram Range (`ngram_range=(1, 2)`):** Captures single words as well as key bigrams (e.g., "climate change", "machine learning", "body language", "mental health", "solar energy", "public speaking").
  - **Frequency Thresholds (`min_df=2`, `max_df=0.70`):** Filters out singleton typos/rare errors while discarding ubiquitous words appearing in over 70% of talks.
  - **Sublinear TF Scaling (`sublinear_tf=True`):** Replaces raw term frequency with $1 + \log(TF)$, preventing extremely long transcripts from overpowering concise, focused talks.
  - **Normalization (`norm='l2'`):** Normalizes all document vectors to unit length so that inner product equals cosine similarity.
- **Outputs & Saved Model Artifacts:**
  - `models/tfidf_vectorizer.pkl` (Fitted Scikit-learn TF-IDF vectorizer, 187 KB)
  - `models/tfidf_matrix.npz` (Compressed Scipy sparse CSR matrix: 2,550 talks × 5,000 features, 92.71% sparsity, 7.97 MB)
  - `models/talk_index_mapping.csv` (Mapping between sparse row indices and talk metadata, 551 KB)
  - `reports/sample_similarity_matches.csv` (Top-5 matches for landmark talks across multiple domains)
  - `visualizations/sample_similarity_heatmap.png` (Cosine similarity heatmap across landmark talks)

### Module 4: Recommendation Engine & Automated Testing
- **Script:** [`recommendation_engine.py`](recommendation_engine.py)
- **Test Suite:** [`test_recommendations.py`](test_recommendations.py)
- **Engine Capabilities:**
  - **Content-Based Filtering:** Computes cosine similarity via `linear_kernel` between the query talk vector and the sparse corpus matrix in under 5 milliseconds.
  - **Robust Title Searching:** Supports exact title match, case-insensitive match, normalized punctuation-free match, and substring search.
  - **Intelligent Error Handling & Suggestions:** When an unknown title is searched, the engine provides fuzzy suggestions via `difflib` (e.g., "Did you mean: ...").
  - **Input Validation:** Safely rejects empty, whitespace-only, or non-string inputs with informative `ValueError` exceptions.
  - **Self-Exclusion:** The queried talk is strictly excluded from its own recommendations (`sims[query_idx] = -1.0`).
  - **Metadata Presentation:** Returns rank, title, main speaker, clean tags, truncated description, talk URL, similarity score, similarity percentage, duration, and community views.
  - **Free-Form Text Search:** `recommend_from_text(query)` vectorizes arbitrary user text (e.g., "robotics and artificial intelligence") and ranks talks matching the thematic intent.
  - **Hybrid Interface:** `hybrid_recommendation_interface()` defines the architectural formulation for blending content similarity with collaborative filtering ($Score_{hybrid} = \alpha \cdot Score_{content} + (1 - \alpha) \cdot Score_{collab}$) once genuine user interaction logs are introduced.

---

## Technology Stack

| Technology | Version | Purpose |
| :--- | :--- | :--- |
| **Python** | `3.13+` | Core programming language |
| **Pandas** | `2.3+` | Data loading, manipulation, and quality reporting |
| **NumPy** | `2.3+` | Numerical linear algebra and vector operations |
| **Scikit-learn** | `1.7+` | `TfidfVectorizer`, stopword filtering, `linear_kernel` |
| **Scipy** | `1.16+` | Sparse matrix compression (`scipy.sparse.csr_matrix`) |
| **Joblib** | `1.5+` | High-efficiency model serialization and persistence |
| **Matplotlib** | `3.11+` | Static publication-grade visualizations |
| **Seaborn** | `0.13+` | Distribution plots, heatmaps, and statistical charts |
| **Pytest** | `9.2+` | Automated regression and unit test suite |

Dependencies are managed in [`requirements.txt`](requirements.txt).

---

## Dataset Description & Provenance

The system utilizes the established TED Talks dataset:
- **`ted_main.csv`:** Metadata for 2,550 TED Talks across 17 attributes (`title`, `main_speaker`, `tags`, `description`, `url`, `duration`, `views`, `ratings`, `comments`, `event`, `film_date`, etc.).
- **`transcripts.csv`:** Full-text English transcripts for 2,467 talks (2,464 unique talks after removing 3 duplicates).
- **Source & Attribution:** Sourced from Kaggle ([`rounakbanik/ted-talks`](https://www.kaggle.com/datasets/rounakbanik/ted-talks)), originally scraped from [ted.com](https://www.ted.com).
- **Licensing:** Licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) and the TED Talks Usage Policy for non-commercial educational purposes.
- **Documentation:** Full field schemas, data types, and collection notes are maintained in [`DATA_SOURCES.md`](DATA_SOURCES.md).

---

## Repository Structure

```text
ted-talks-recommendation-system/
├── data_collection.py                    # Module 1: Ingestion, URL normalization & dataset merging
├── preprocess_ted_data.py                # Module 2: Text cleaning, token metrics & EDA reporting
├── feature_engineering.py                # Module 3: TF-IDF vectorization & sparse matrix creation
├── recommendation_engine.py              # Module 4: Content-based engine with title/keyword search
├── test_recommendations.py               # Automated pytest suite (18 unit & regression tests)
├── cleaned_ted_talks.csv                 # Cleaned & enriched dataset (2,550 talks, 37 columns)
├── data_quality_report.csv               # 19-column granular data quality assessment
├── preprocessing_quality_report.csv      # Preprocessing quality audit report
├── preprocessing_summary.csv             # Statistical overview of cleaned talk attributes
├── requirements.txt                      # Project dependencies
├── .gitignore                            # Excludes venvs, caches, temporary files
├── README.md                             # Complete project documentation and guide
├── DATA_SOURCES.md                       # Comprehensive data provenance and licensing
├── data/
│   ├── ted_main.csv                      # Raw metadata (2,550 records, 17 columns)
│   ├── transcripts.csv                   # Raw transcripts (2,467 records, 2 columns)
│   ├── collected_ted_talks.csv           # Consolidated collection from Module 1
│   ├── cleaned_ted_talks.csv             # Cleaned dataset copy in data/
│   ├── ted_main_cleaned.csv              # Cleaned metadata from exploratory phase
│   └── transcripts_cleaned.csv           # Cleaned transcripts from exploratory phase
├── models/
│   ├── tfidf_vectorizer.pkl              # Serialized TfidfVectorizer (5,000 vocabulary)
│   ├── tfidf_matrix.npz                  # Scipy sparse CSR matrix (2,550 x 5,000)
│   ├── talk_index_mapping.csv            # Mapping of row indices to talk metadata
│   └── sample_similarity_matches.csv     # Pre-computed recommendations for landmark talks
├── visualizations/
│   ├── topic_frequency.png               # Top 20 most frequent talk topics/tags
│   ├── transcript_word_count_distribution.png # Distribution of transcript word lengths
│   └── sample_similarity_heatmap.png     # Cosine similarity heatmap of landmark talks
├── reports/
│   ├── data_quality_report.csv           # Quality report copy in reports/
│   ├── preprocessing_summary.csv         # Summary copy in reports/
│   ├── sample_similarity_matches.csv     # Sample similarity report
│   ├── interaction_matrix_schema.csv     # Documented interaction matrix schema
│   └── eda/                              # Visualizations from exploratory analysis
└── notebooks/
    ├── 01_data_collection.ipynb          # Executed Jupyter notebook for Module 1
    └── 02_eda_preprocessing.ipynb        # Executed Jupyter notebook for Module 2
```

---

## Setup and Execution Guide (Windows PowerShell)

Follow these steps in Windows PowerShell to set up the environment, run each module sequentially, and execute automated tests:

### 1. Clone the repository
```powershell
git clone https://github.com/leena-11/ted-talks-recommendation-system.git
cd "ted-talks-recommendation-system"
```

### 2. Create and activate a virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run Module 1: Data Collection & Ingestion
```powershell
python data_collection.py
```
*Expected Output:* Confirms ingestion of 2,550 metadata records and 2,467 transcripts, removes 3 duplicate transcripts, and saves `data/collected_ted_talks.csv`.

### 5. Run Module 2: Data Exploration & Preprocessing
```powershell
python preprocess_ted_data.py
```
*Expected Output:* Cleans all text fields, calculates token counts, generates `visualizations/topic_frequency.png` and `visualizations/transcript_word_count_distribution.png`, and writes `cleaned_ted_talks.csv`, `data_quality_report.csv`, and `preprocessing_summary.csv`.

### 6. Run Module 3: Feature Engineering & Vectorization
```powershell
python feature_engineering.py
```
*Expected Output:* Fits TF-IDF vectorizer (5,000 features, unigrams and bigrams), saves `models/tfidf_vectorizer.pkl`, `models/tfidf_matrix.npz`, `models/talk_index_mapping.csv`, and produces `visualizations/sample_similarity_heatmap.png`.

### 7. Run Module 4: Recommendation Engine
Execute a search by talk title:
```powershell
python recommendation_engine.py --title "Do schools kill creativity?" --top_n 5
```
Or execute a free-form keyword search:
```powershell
python recommendation_engine.py --keyword "artificial intelligence and robotics" --top_n 5
```
Or run the default multi-domain demo:
```powershell
python recommendation_engine.py
```

### 8. Run Automated Test Suite
```powershell
pytest test_recommendations.py -v
```
*Expected Output:* All 18 automated tests pass with 100% success.

---

## Verified Recommendation Examples

The following recommendation results were computed using actual cosine similarity on the fitted 5,000-dimensional TF-IDF space:

### Query 1: *"Do schools kill creativity?"* by Ken Robinson (Education / Creativity)
| Rank | Recommended Talk Title | Speaker | Similarity | Top Tags |
| :---: | :--- | :--- | :---: | :--- |
| **#1** | **Bring on the learning revolution!** | Ken Robinson | **26.6%** | `creativity`, `education`, `children`, `invention` |
| **#2** | **How to escape education's death valley** | Ken Robinson | **26.5%** | `education`, `culture`, `teaching`, `student` |
| **#3** | **Rebuilding a neighborhood with beauty, dignity, hope** | Bill Strickland | **26.1%** | `activism`, `children`, `cities`, `culture`, `art` |
| **#4** | **Letting go of God** | Julia Sweeney | **24.9%** | `comedy`, `culture`, `performance`, `storytelling` |
| **#5** | **The story of a girl** | Rick Smolan | **24.8%** | `art`, `children`, `family`, `photography`, `media` |

*Analysis:* The top two recommendations are Sir Ken Robinson's official follow-up TED talks, demonstrating high semantic coherence and thematic alignment.

### Query 2: *"Averting the climate crisis"* by Al Gore (Climate / Environment)
| Rank | Recommended Talk Title | Speaker | Similarity | Top Tags |
| :---: | :--- | :--- | :---: | :--- |
| **#1** | **New thinking on the climate crisis** | Al Gore | **28.7%** | `activism`, `climate change`, `global issues`, `science` |
| **#2** | **The case for optimism on climate change** | Al Gore | **27.3%** | `alternative energy`, `climate change`, `science` |
| **#3** | **What comes after An Inconvenient Truth?** | Al Gore | **26.6%** | `ecology`, `climate change`, `sustainability` |
| **#4** | **Salvation (and profit) in greentech** | John Doerr | **24.6%** | `climate change`, `green investment`, `technology` |
| **#5** | **Climate change is happening. Here's how we adapt** | Alice Bows-Larkin | **24.2%** | `climate change`, `energy`, `policy`, `sustainability` |

*Analysis:* All 5 recommendations directly address climate change, clean energy, and sustainability, illustrating the precision of n-gram TF-IDF matching.

### Query 3: *"The thrilling potential of SixthSense technology"* by Pranav Mistry (Technology / HCI)
| Rank | Recommended Talk Title | Speaker | Similarity | Top Tags |
| :---: | :--- | :--- | :---: | :--- |
| **#1** | **Shape-shifting tech will change work as we know it** | Sean Follmer | **27.5%** | `interface design`, `computers`, `innovation` |
| **#2** | **The radical promise of the multi-touch interface** | Jeff Han | **27.0%** | `demo`, `interface design`, `technology` |
| **#3** | **Meet the SixthSense interaction** | Pattie Maes + Pranav Mistry | **26.6%** | `demo`, `interface design`, `technology` |
| **#4** | **How Arduino is open-sourcing imagination** | Massimo Banzi | **24.6%** | `creativity`, `open-source`, `robots`, `technology` |
| **#5** | **A futuristic vision of the age of holograms** | Alex Kipman | **24.3%** | `interface design`, `innovation`, `computers` |

---

## Automated Test Suite

The test suite in [`test_recommendations.py`](test_recommendations.py) provides 18 verified automated test cases across three test classes:

1. **`TestRecommendationArtifacts`:**
   - `test_artifacts_exist`: Verifies `cleaned_ted_talks.csv`, `models/tfidf_vectorizer.pkl`, `models/tfidf_matrix.npz`, and `models/talk_index_mapping.csv` are present.
   - `test_matrix_dimensions_match_dataset`: Asserts sparse matrix matches the dataset exactly (2,550 talks × 5,000 features).
2. **`TestRecommendationOutputs`:**
   - `test_valid_talk_recommendations`: Confirms 5 recommendations returned with all required metadata columns.
   - `test_exclude_selected_talk`: Confirms the queried talk is never returned in its own recommendations.
   - `test_correct_number_of_recommendations`: Parameterized test across `top_n` values (1, 3, 5, 8, 10).
   - `test_standalone_recommend_talks_function`: Tests functional module-level entry point `recommend_talks()`.
3. **`TestEdgeCasesAndInputs`:**
   - `test_unknown_title_raises_value_error`: Confirms descriptive `ValueError` with suggestions when a nonexistent title is queried.
   - `test_empty_or_blank_input_raises_value_error`: Parameterized test for `""`, `"   "`, and `None`.
   - `test_case_insensitive_matching`: Asserts lowercase and uppercase queries yield identical recommendations.
   - `test_partial_title_matching`: Tests substring searching (e.g., "body language").
   - `test_talk_without_transcript_generates_recommendations`: Verifies talks without transcripts (e.g., musical/dance performances) successfully generate recommendations via title, tags, and description.
   - `test_free_text_keyword_search`: Validates arbitrary keyword vectorization and retrieval.

**Test Execution Result:**
```text
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.2.0, pluggy-1.7.0
rootdir: E:\infosys\TED Talks Recommendation System
collected 18 items

test_recommendations.py::TestRecommendationArtifacts::test_artifacts_exist PASSED
test_recommendations.py::TestRecommendationArtifacts::test_matrix_dimensions_match_dataset PASSED
test_recommendations.py::TestRecommendationOutputs::test_valid_talk_recommendations PASSED
test_recommendations.py::TestRecommendationOutputs::test_exclude_selected_talk PASSED
test_recommendations.py::TestRecommendationOutputs::test_correct_number_of_recommendations[1] PASSED
test_recommendations.py::TestRecommendationOutputs::test_correct_number_of_recommendations[3] PASSED
test_recommendations.py::TestRecommendationOutputs::test_correct_number_of_recommendations[5] PASSED
test_recommendations.py::TestRecommendationOutputs::test_correct_number_of_recommendations[8] PASSED
test_recommendations.py::TestRecommendationOutputs::test_correct_number_of_recommendations[10] PASSED
test_recommendations.py::TestRecommendationOutputs::test_standalone_recommend_talks_function PASSED
test_recommendations.py::TestEdgeCasesAndInputs::test_unknown_title_raises_value_error PASSED
test_recommendations.py::TestEdgeCasesAndInputs::test_empty_or_blank_input_raises_value_error[] PASSED
test_recommendations.py::TestEdgeCasesAndInputs::test_empty_or_blank_input_raises_value_error[   ] PASSED
test_recommendations.py::TestEdgeCasesAndInputs::test_empty_or_blank_input_raises_value_error[None] PASSED
test_recommendations.py::TestEdgeCasesAndInputs::test_case_insensitive_matching PASSED
test_recommendations.py::TestEdgeCasesAndInputs::test_partial_title_matching PASSED
test_recommendations.py::TestEdgeCasesAndInputs::test_talk_without_transcript_generates_recommendations PASSED
test_recommendations.py::TestEdgeCasesAndInputs::test_free_text_keyword_search PASSED

============================= 18 passed in 7.38s ==============================
```

---

## Limitations & Future Roadmap

### Genuine Dataset Limitations
1. **Absence of User-Level Interaction Logs:** The public TED Talks dataset contains community aggregate metrics (`views`, `comments`, `ratings` counts) per talk rather than individualized user accounts, user IDs, or chronological watch histories. As required by academic integrity, collaborative filtering cannot be genuinely trained on this public release without fabricating synthetic user data.
2. **Non-Spoken Performance Talks:** 86 talks (3.37% of the dataset) are non-spoken live musical, dance, or mime performances (e.g., Pilobolus, Ethel, Vusi Mahlasela) that lack spoken transcripts. The system gracefully falls back to title, speaker, and tag metadata for these talks.
3. **Lexical Matching vs. Dense Embeddings:** TF-IDF captures exact terms and bigram patterns effectively, but does not inherently capture synonymy across distinct vocabularies (e.g., "doctor" vs. "physician").

### Future Enhancements
- **Dense Transformer Embeddings:** Incorporate pre-trained Sentence-Transformers (e.g., `all-MiniLM-L6-v2`) to capture deep semantic embeddings alongside TF-IDF.
- **Interactive Streamlit Web App:** Deploy an interactive web application allowing users to browse talks, adjust feature weights, and view embedded TED video players.
- **Simulated / Opt-In User Profiles:** If opt-in user watch histories are collected via the web app, activate the `hybrid_recommendation_interface()` using Matrix Factorization (SVD / ALS).

---

## License & Attribution

- **Source Code License:** The code in this repository is licensed under the [MIT License](https://opensource.org/licenses/MIT).
- **Dataset License & Attribution:** The underlying TED Talks dataset is provided by Rounak Banik via Kaggle, sourced from [TED Conferences LLC](https://www.ted.com), and used under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) and the TED Talks Usage Policy for non-commercial educational purposes.
