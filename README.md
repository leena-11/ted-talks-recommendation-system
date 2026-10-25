# Personalized TED Talks Recommendation System Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Infosys Springboard](https://img.shields.io/badge/Infosys%20Springboard-Virtual%20Internship%207.0-brightgreen.svg)](https://infyspringboard.onwingspan.com/)

An end-to-end Machine Learning recommendation system project developed under the **Infosys Springboard Virtual Internship 7.0**. The system analyzes TED Talk metadata, audience ratings, and full spoken transcripts to deliver personalized talk recommendations.

> **Current Milestone (Milestone 1 / Weeks 1-2):**
> Implementation of **Module 1: Data Collection** and **Module 2: Data Exploration and Data Preprocessing** only.
> *(Advanced feature engineering, TF-IDF, vector embeddings, collaborative filtering, and Streamlit app deployment are reserved for subsequent modules).*

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Technology Stack](#technology-stack)
4. [Module 1: Data Collection & Dataset Relationship Analysis](#module-1-data-collection--dataset-relationship-analysis)
   - [Dataset Attribution & Licensing](#dataset-attribution--licensing)
   - [Dataset Schema & Field Descriptions](#dataset-schema--field-descriptions)
   - [Candidate Key & Linking Analysis](#candidate-key--linking-analysis)
   - [Missing Transcripts Investigation](#missing-transcripts-investigation)
   - [Dataset Limitations](#dataset-limitations)
5. [Module 2: Data Exploration & Preprocessing](#module-2-data-exploration--preprocessing)
   - [Initial Data Assessment & Quality Report](#initial-data-assessment--quality-report)
   - [Exploratory Data Analysis (EDA) Highlights](#exploratory-data-analysis-eda-highlights)
   - [Data Cleaning & Transformation Decisions](#data-cleaning--transformation-decisions)
   - [Before-and-After Validation](#before-and-after-validation)
6. [Generated Deliverables & Outputs](#generated-deliverables--outputs)
7. [Installation & Execution Guide](#installation--execution-guide)
8. [Next Steps (Module 3 Roadmap)](#next-steps-module-3-roadmap)

---

## Project Overview

With thousands of TED Talks spanning science, technology, education, design, business, and humanities, users face significant discovery challenges. This project builds a machine-learning recommender system combining:
- **Content-Based Filtering:** Recommending talks based on transcript semantics, tags, speaker topics, and emotional ratings.
- **Collaborative Filtering:** Modeling user viewing patterns and community engagement dynamics.

This phase completes the foundational data engineering lifecycle: gathering, verifying, exploring, auditing, and cleaning the corpus.

---

## Repository Structure

```text
TED Talks Recommendation System/
├── data/
│   ├── ted_main.csv               # Original raw TED Talk metadata (2,550 records, 17 columns)
│   ├── transcripts.csv            # Original raw transcripts (2,467 records, 2 columns)
│   ├── ted_main_cleaned.csv       # Cleaned & feature-engineered metadata (2,550 records, 27 columns)
│   └── transcripts_cleaned.csv    # Cleaned, deduplicated transcripts (2,464 records, 7 columns)
├── notebooks/
│   ├── 01_data_collection.ipynb   # Module 1: Ingestion, schema audit, key analysis & licensing
│   └── 02_eda_preprocessing.ipynb # Module 2: In-depth EDA charts, data cleaning & validation
├── reports/
│   ├── data_quality_report.csv    # Granular 19-column quality audit (nulls, blanks, padding, issues)
│   └── eda/                       # 11 Publication-grade visualization charts (PNG, 300 DPI)
│       ├── 01_top_tags.png
│       ├── 02_duration_distribution.png
│       ├── 03_views_distribution.png
│       ├── 04_ratings_distribution.png
│       ├── 05_missing_values.png
│       ├── 06_transcript_length_distribution.png
│       ├── 07_top_speakers.png
│       ├── 08_top_occupations.png
│       ├── 09_views_vs_comments.png
│       ├── 10_talks_by_year.png
│       └── 11_transcript_coverage.png
├── src/
│   └── data_processing.py         # Production-grade, reusable cleaning & EDA pipeline script
├── requirements.txt               # Pinned Python package dependencies
└── README.md                      # Project documentation and reproduction guide
```

---

## Technology Stack

- **Language:** Python 3.13+
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Interactive Notebooks:** Jupyter Notebook, nbformat, nbconvert
- **Text & Preprocessing:** Regex (`re`), AST (`ast.literal_eval`), Scikit-learn

---

## Module 1: Data Collection & Dataset Relationship Analysis

### Dataset Attribution & Licensing

| Dimension | Details |
| :--- | :--- |
| **Dataset Source** | Kaggle TED Talks Dataset (`rounakbanik/ted-talks`), scraped from the official TED website ([ted.com](https://www.ted.com)). |
| **Curator / Attribution** | Rounak Banik; TED Conferences LLC. |
| **Licensing** | Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)) and TED Terms of Use for non-commercial educational use. |
| **Temporal Span** | Talks filmed between May 1972 and August 2017; published on ted.com through September 21, 2017. |

### Dataset Schema & Field Descriptions

#### 1. Metadata (`data/ted_main.csv` - 2,550 Rows, 17 Columns)
- `comments` (*int64*): Number of public first-level comments on ted.com.
- `description` (*object*): Concise textual synopsis of the talk.
- `duration` (*int64*): Talk duration in seconds.
- `event` (*object*): TED conference or partner event (e.g., `TED2006`, `TEDGlobal 2012`, `TEDx`).
- `film_date` (*int64*): Unix epoch timestamp of when the talk was filmed.
- `languages` (*int64*): Number of subtitle translations available.
- `main_speaker` (*object*): Name of the primary speaker.
- `name` (*object*): Formal title combining speaker and talk title (e.g., `Ken Robinson: Do schools kill creativity?`).
- `num_speaker` (*int64*): Number of speakers participating in the talk.
- `published_date` (*int64*): Unix epoch timestamp of when the talk was published to ted.com.
- `ratings` (*object*): JSON-encoded list of dictionaries recording audience emotional response votes.
- `related_talks` (*object*): JSON-encoded list of recommended related talks.
- `speaker_occupation` (*object*): Professional background/title of the speaker (6 missing values).
- `tags` (*object*): String-encoded Python list of topic keywords.
- `title` (*object*): Official title of the talk.
- `url` (*object*): Web URL to the talk page on ted.com (contains trailing `\n` in raw data).
- `views` (*int64*): Cumulative view count on ted.com.

#### 2. Transcripts (`data/transcripts.csv` - 2,467 Rows, 2 Columns)
- `transcript` (*object*): Full English verbatim spoken transcript.
- `url` (*object*): Talk page URL on ted.com (contains trailing `\n` in raw data).

### Candidate Key & Linking Analysis
1. **Candidate Keys Evaluated:**
   - `name`: Unique in metadata (2,550), but does not exist in `transcripts.csv`.
   - `title`: Unique in metadata (2,550), but does not exist in `transcripts.csv`.
   - `url`: Available in both datasets.
2. **Formatting Anomaly Discovered:**
   - Every single `url` in both `ted_main.csv` and `transcripts.csv` contained a trailing newline character (`\n`) and whitespace.
   - Once stripped (`url.str.strip()`), URLs formed an exact 1:1 join key.
3. **Duplicate Detection in Transcripts:**
   - Raw `transcripts.csv` had 2,467 rows but only 2,464 unique URLs.
   - Exactly 3 URLs were duplicated (rows 1107–1109 and 1114–1116):
     - `https://www.ted.com/talks/brene_brown_listening_to_shame`
     - `https://www.ted.com/talks/jonathan_haidt_humanity_s_stairway_to_self_transcendence`
     - `https://www.ted.com/talks/rob_reid_the_8_billion_ipod`
   - Verifying word-for-word string equality confirmed these were 100% identical exact duplicate rows, which were safely deduplicated.
4. **Dataset Overlap:**
   - **Common Talks:** 2,464 talks (96.63% of metadata, 100% of unique transcripts).
   - **Metadata Only:** 86 talks (3.37% of metadata).
   - **Transcripts Only:** 0 talks (100% transcript coverage against metadata).

### Missing Transcripts Investigation
Why are 86 talks in `ted_main.csv` missing from `transcripts.csv`?
A domain audit revealed titles such as:
- *"A dance of 'Symbiosis'"* (dance/performance)
- *"A string quartet plays 'Blue Room'"* (instrumental live music)
- *"'Woza'"* and *"'M'Bifo'"* (guitar performance)
- *"Piano jazz that rocks"* (jazz piano)
- *"An electrifying organ performance"* (instrumental organ)

These 86 talks represent **musical performances, dances, and artistic demonstrations**. Because they contain no spoken dialogue, TED never created transcripts for them. This is authentic domain behavior rather than data corruption.

### Dataset Limitations
1. **Static Temporal Cutoff:** Data is limited to talks published through September 2017.
2. **Absence of User-Level Interaction Logs:** The dataset contains aggregate community ratings, but no individual user watch histories or per-user rating matrices. Collaborative filtering in subsequent modules must leverage synthesized interaction matrices or rating-vector user profiles.
3. **86 Non-Verbal Talks:** Recommender features requiring transcript text must implement graceful fallbacks to `title`, `description`, and `tags` for performance talks.

---

## Module 2: Data Exploration & Preprocessing

### Initial Data Assessment & Quality Report
An automated assessment generated [`reports/data_quality_report.csv`](reports/data_quality_report.csv). Summary findings:
- **Completeness:** `ted_main.csv` has 43,344 total values with only 6 nulls (99.986% complete).
- **Missing Data:** Only `speaker_occupation` contained nulls (6 records / 0.235%).
- **Syntax / Messy Data:** 2,550 URLs in `ted_main` and 2,467 URLs in `transcripts` had trailing newlines (`\n`). 53 descriptions and 42 speaker occupations had leading/trailing whitespace padding.
- **Complex Structures:** `tags` is a Python string-encoded list (`"['tag1', 'tag2']"`). `ratings` is a string-encoded list of dicts.

### Exploratory Data Analysis (EDA) Highlights
All 11 visualizations are saved at 300 DPI in [`reports/eda/`](reports/eda/):

1. **`01_top_tags.png` (Top 20 Tags):** `technology` (727), `science` (567), `global issues` (501), `culture` (486), and `design` (418) dominate the corpus, perfectly reflecting the TED acronym (*Technology, Entertainment, Design*).
2. **`02_duration_distribution.png` (Durations):** Distinct unimodal distribution centered at the historical 18-minute TED benchmark ($1,080\text{ seconds}$). Median duration: 14.1 minutes (848 seconds); Mean: 13.8 minutes.
3. **`03_views_distribution.png` (Views):** Heavily right-skewed Pareto distribution. Median views: $1.12\text{M}$; maximum: $47.23\text{M}$ (Sir Ken Robinson's *"Do schools kill creativity?"*). Log-transformation achieves a Gaussian distribution.
4. **`04_ratings_distribution.png` (Audience Ratings):** The 14 rating categories capture $7.44\text{ million}$ aggregate audience votes. Top emotions: `Inspiring` ($1.35\text{M}$), `Informative` ($887\text{k}$), `Fascinating` ($801\text{k}$), and `Persuasive` ($561\text{k}$).
5. **`05_missing_values.png` (Data Completeness):** Visualizes the 6 missing `speaker_occupation` values and 86 unmatched non-spoken talks against a 100% complete core schema.
6. **`06_transcript_length_distribution.png` (Transcript Word Counts):** Median transcript contains $2,028\text{ words}$ ($11,539\text{ characters}$); mean: $2,040\text{ words}$. Short outliers (<50 words) represent performances with minimal stage notes.
7. **`07_top_speakers.png` (Top 15 Speakers):** Hans Rosling leads with 9 talks, followed by Juan Enriquez (7), Rives (6), Marco Tempest (6), and Nicholas Negroponte / Bill Gates (5 each).
8. **`08_top_occupations.png` (Top 15 Occupations):** Writers (45), Designers (34), Artists (34), Journalists (33), Entrepreneurs (31), Architects (30), and Inventors (27).
9. **`09_views_vs_comments.png` (Engagement Correlation):** Moderate-to-strong positive correlation (Pearson $r = 0.53$, log-log scale) between view counts and user comments.
10. **`10_talks_by_year.png` (Publication Trend):** Steady exponential growth from 2006 (50 talks) peaking between 2011–2016 at ~250 talks/year.
11. **`11_transcript_coverage.png` (Donut Chart):** 96.6% of talks have complete spoken transcripts; 3.4% are non-spoken musical/artistic presentations.

### Data Cleaning & Transformation Decisions

| Pipeline Step | Cleaning Action | Technical Rationale |
| :--- | :--- | :--- |
| **URL Normalization** | Stripped `\n` and whitespace | Resolves join failures between metadata and transcript records. |
| **Whitespace Trimming** | Stripped all object columns (`title`, `main_speaker`, `speaker_occupation`, `description`) | Cleans formatting padding without altering lexical content. |
| **Missing Occupation** | Imputed 6 nulls with `'Unknown'` | Preserves 100% of talk metadata without fabricating professional titles. |
| **Transcript Deduplication** | Dropped 3 confirmed exact duplicate rows | Guarantees exactly one unique transcript per talk URL (2,464 rows retained). |
| **Tag List Parsing** | Parsed via `ast.literal_eval` into clean lowercased tokens; engineered `tag_count` and `clean_tags` | Converts raw strings into structured lists and space-separated strings for NLP vectorizers. |
| **Rating Feature Engineering** | Parsed rating dictionaries to extract `total_ratings` and `top_rating` | Enables affective / sentiment-based recommendation signals. |
| **Datetime Normalization** | Converted Unix epochs to `film_datetime`, `published_datetime`, and `published_year` | Enables temporal filtering and recency weighting. |
| **Duration in Minutes** | Derived `duration_minutes = round(duration / 60, 2)` | Standardizes duration for human readability and length-based filtering. |
| **Transcript Text Cleaning** | Removed stage directions like `(Applause)` / `(Laughter)`, lowercased, normalized punctuation and whitespace | Cleans non-lexical acoustic noise while preserving the full verbatim transcript in `transcript`. |
| **Coverage Flagging** | Added `has_transcript` binary column ($1$ if transcript available, $0$ otherwise) | Allows downstream recommendation models to route features dynamically. |

### Before-and-After Validation

| Metric / Dimension | `ted_main` (Raw) | `ted_main_cleaned` | `transcripts` (Raw) | `transcripts_cleaned` |
| :--- | :---: | :---: | :---: | :---: |
| **Total Rows** | 2,550 | 2,550 | 2,467 | 2,464 |
| **Total Columns** | 17 | 27 | 2 | 7 |
| **Total Missing Values** | 6 | **0** | 0 | **0** |
| **Duplicate Rows** | 0 | **0** | 3 | **0** |
| **Records Retained** | — | **2,550 (100%)** | — | **2,464 (99.88%)** |
| **Records Removed** | — | **0** | — | **3 (Exact Duplicates)** |

---

## Generated Deliverables & Outputs

1. **Cleaned Datasets:**
   - [`data/ted_main_cleaned.csv`](data/ted_main_cleaned.csv) (2,550 rows, 27 columns, 7.95 MB)
   - [`data/transcripts_cleaned.csv`](data/transcripts_cleaned.csv) (2,464 rows, 7 columns, 55.76 MB)
2. **Quality & Audit Reports:**
   - [`reports/data_quality_report.csv`](reports/data_quality_report.csv) (Granular audit across 19 columns)
3. **Exploratory Visualizations:**
   - [`reports/eda/`](reports/eda/) (11 publication-grade PNG charts)
4. **Interactive Jupyter Notebooks:**
   - [`notebooks/01_data_collection.ipynb`](notebooks/01_data_collection.ipynb) (Fully executed, 24 cells)
   - [`notebooks/02_eda_preprocessing.ipynb`](notebooks/02_eda_preprocessing.ipynb) (Fully executed, 44 cells)
5. **Reusable Python Pipeline:**
   - [`src/data_processing.py`](src/data_processing.py) (Modular, tested end-to-end cleaning and EDA script)
6. **Dependencies:**
   - [`requirements.txt`](requirements.txt)

---

## Installation & Execution Guide

### 1. Environment Setup
Clone the repository and install required packages:
```bash
# Clone repository (if applicable) or navigate to project folder
cd "e:\infosys\TED Talks Recommendation System"

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full Processing Pipeline
To regenerate cleaned CSVs, quality reports, and EDA figures in a single command:
```bash
python src/data_processing.py
```

### 3. Launch the Interactive Notebooks
Start Jupyter Notebook to inspect the executed workflows:
```bash
jupyter notebook
```
Navigate to:
- `notebooks/01_data_collection.ipynb`
- `notebooks/02_eda_preprocessing.ipynb`

Both notebooks are pre-executed with all cell outputs, tables, and charts visible immediately.

---

## Next Steps (Module 3 Roadmap)

Per the Infosys Springboard requirements, subsequent modules will implement:
- **Module 3: Feature Engineering**
  - Text tokenization and lemmatization (NLTK / spaCy).
  - TF-IDF feature extraction on cleaned transcripts, tags, and talk descriptions.
  - Aggregation of metadata features (duration, emotional ratings, publication recency).
- **Module 4: Recommendation Engine Development**
  - Content-based cosine similarity matrix computation.
  - User profile modeling and simulated user-item interaction matrix.
  - Hybrid recommendation algorithm combining content semantics and rating distributions.
- **Module 5: Evaluation & Validation**
  - Precision@K, Recall@K, MAP (Mean Average Precision), and NDCG metrics.
- **Module 6: Streamlit Web Application Deployment**
  - Interactive web interface for real-time talk discovery, tag filtering, and recommendation exploration.
