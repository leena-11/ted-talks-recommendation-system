# Personalized TED Talks Recommendation System Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Infosys Springboard](https://img.shields.io/badge/Infosys%20Springboard-Virtual%20Internship%207.0-brightgreen.svg)](https://infyspringboard.onwingspan.com/)
[![Milestone 1](https://img.shields.io/badge/Milestone%201-Completed%20%26%20Verified-brightgreen.svg)]()
[![Milestone 2--4](https://img.shields.io/badge/Milestones%202--4-Planned%20(Pending)-orange.svg)]()

An industry-aligned Machine Learning recommendation system project developed under the **Infosys Springboard Virtual Internship 7.0**. The system analyzes TED Talk metadata, audience affective ratings, and full spoken transcripts to deliver personalized recommendations.

---

## Project Status & Milestone Tracking

| Milestone | Stage / Modules | Status | Deliverables / Notes |
| :--- | :--- | :---: | :--- |
| **Milestone 1** | **Module 1: Data Collection**<br>**Module 2: Data Exploration & Preprocessing** | **COMPLETED & VERIFIED** | Validated datasets, automated audit reports, 11 EDA charts, clean CSVs, and executed notebooks |
| **Milestone 2** | **Module 3: Feature Engineering**<br>**Module 4: Content-Based Recommendation** | *PLANNED* | Text tokenization, TF-IDF / embeddings, metadata feature vectors, cosine similarity |
| **Milestone 3** | **Module 5: Collaborative / Hybrid Engine**<br>**Module 6: Model Evaluation (Precision, Recall, NDCG)** | *PLANNED* | Simulated user-interaction matrix, hybrid scoring, rank evaluation |
| **Milestone 4** | **Module 7: Interactive Streamlit Web Application** | *PLANNED* | Responsive UI for talk discovery, tag filtering, and recommendation serving |

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Technology Stack](#technology-stack)
4. [Module 1: Data Collection & Dataset Relationship Analysis](#module-1-data-collection--dataset-relationship-analysis)
   - [Data Sources & Licensing](#data-sources--licensing)
   - [Schema & Field Descriptions](#schema--field-descriptions)
   - [Candidate Key & Linking Analysis](#candidate-key--linking-analysis)
   - [Investigation of Missing Transcripts](#investigation-of-missing-transcripts)
   - [Investigation of User Interaction Data](#investigation-of-user-interaction-data)
5. [Module 2: Data Exploration & Preprocessing](#module-2-data-exploration--preprocessing)
   - [Initial Data Assessment & Quality Report](#initial-data-assessment--quality-report)
   - [Exploratory Data Analysis (EDA) Highlights](#exploratory-data-analysis-eda-highlights)
   - [Data Cleaning & Transformation Decisions](#data-cleaning--transformation-decisions)
   - [Before-and-After Validation Report](#before-and-after-validation-report)
6. [Generated Deliverables & Outputs](#generated-deliverables--outputs)
7. [Installation & Execution Guide](#installation--execution-guide)
8. [Next Steps (Milestone 2 Roadmap)](#next-steps-milestone-2-roadmap)

---

## Project Overview

With thousands of TED Talks spanning science, technology, education, design, business, and humanities, users encounter substantial information overload. This project builds a machine-learning recommendation system combining:
- **Content-Based Filtering:** Recommending talks based on transcript semantics, tags, speaker domain, and audience emotional ratings.
- **Collaborative / Hybrid Filtering:** Modeling user viewing patterns and community engagement dynamics.

Milestone 1 establishes the complete data engineering lifecycle: gathering, verifying, exploring, auditing, cleaning, and engineering foundational features.

---

## Repository Structure

```text
TED Talks Recommendation System/
├── data/
│   ├── ted_main.csv               # Original raw TED Talk metadata (2,550 records, 17 columns)
│   ├── transcripts.csv            # Original raw transcripts (2,467 records, 2 columns)
│   ├── ted_main_cleaned.csv       # Cleaned & feature-engineered metadata (2,550 records, 28 columns)
│   └── transcripts_cleaned.csv    # Cleaned, deduplicated transcripts (2,464 records, 7 columns)
├── notebooks/
│   ├── 01_data_collection.ipynb   # Module 1: Pre-executed ingestion, schema audit & key analysis
│   └── 02_eda_preprocessing.ipynb # Module 2: Pre-executed EDA charts, data cleaning & validation
├── reports/
│   ├── data_quality_report.csv    # Granular 19-column quality audit (nulls, blanks, padding, issues)
│   ├── before_after_validation.csv# Before-and-after cleaning delta metrics and rationale
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
├── DATA_SOURCES.md                # Comprehensive data source documentation, licensing, & provenance
├── requirements.txt               # Pinned Python package dependencies
├── .gitignore                     # Git rules ignoring virtual environments, caches, and secrets
└── README.md                      # Project documentation and execution guide
```

---

## Technology Stack

- **Language:** Python 3.13+
- **Data Manipulation:** Pandas, NumPy
- **Visualizations:** Matplotlib, Seaborn
- **Notebooks:** Jupyter Notebook, nbformat, nbconvert
- **Text & Transformation:** Regex (`re`), AST (`ast.literal_eval`), Scikit-learn
- **Version Control:** Git, GitHub

---

## Module 1: Data Collection & Dataset Relationship Analysis

### Data Sources & Licensing
Detailed documentation is provided in [`DATA_SOURCES.md`](DATA_SOURCES.md). Summary:
- **Source:** Kaggle TED Talks Dataset (`rounakbanik/ted-talks`), scraped from official [TED.com](https://www.ted.com) talk pages.
- **Curator & Attribution:** Rounak Banik; TED Conferences LLC.
- **Licensing:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)) and TED.com Terms of Use for non-commercial educational use.
- **Temporal Coverage:** Talks filmed between May 1972 and August 2017; published on ted.com through September 21, 2017.

### Schema & Field Descriptions
- **`data/ted_main.csv` (2,550 rows, 17 columns):**
  `comments`, `description`, `duration`, `event`, `film_date`, `languages`, `main_speaker`, `name`, `num_speaker`, `published_date`, `ratings`, `related_talks`, `speaker_occupation`, `tags`, `title`, `url`, `views`.
- **`data/transcripts.csv` (2,467 rows, 2 columns):**
  `transcript`, `url`.

### Candidate Key & Linking Analysis
1. **Candidate Keys Evaluated:**
   - `name`: Unique in metadata (2,550), but does not exist in `transcripts.csv`.
   - `title`: Unique in metadata (2,550), but does not exist in `transcripts.csv`.
   - `url`: Available in both datasets and serves as the linking primary/foreign key.
2. **Formatting Anomaly Discovered & Fixed:**
   - Every `url` in both raw files contained a trailing newline (`\n`) and whitespace padding. Stripping whitespace (`df['url'].str.strip()`) restored clean 1:1 joinability.
3. **Exact Duplicates in Transcripts:**
   - Identified 3 duplicate talk URLs in `transcripts.csv` (rows 1107–1109 vs 1114–1116).
   - Comparing verbatim text confirmed they were 100% identical duplicates. Dropping duplicates retained **2,464 unique talk transcripts**.
4. **Dataset Overlap:**
   - **Common Talks:** 2,464 talks (96.63% of metadata, 100% of unique transcripts).
   - **Metadata Only:** 86 talks (3.37% of metadata).
   - **Transcripts Only:** 0 talks (100% transcript coverage against metadata).

### Investigation of Missing Transcripts
Why are 86 talks in `ted_main.csv` missing from `transcripts.csv`?
A domain audit revealed titles such as:
- *"A dance of 'Symbiosis'"* (Choreography/dance)
- *"A string quartet plays 'Blue Room'"* (Live cello/string quartet)
- *"'Woza'"* and *"'M'Bifo'"* (Live African guitar)
- *"Piano jazz that rocks"* (Instrumental jazz piano)
- *"An electrifying organ performance"* (Organ showcase)

**Domain Finding:** These 86 talks are **instrumental musical performances, choreographies, and non-verbal artistic showcases**. Because there was no spoken speech, official TED transcripts were naturally never created.
This represents authentic domain behavior rather than data corruption. Recommendation pipelines in later modules will use `title`, `description`, and `clean_tags` for content modeling on these non-spoken items.

### Investigation of User Interaction Data
In accordance with Milestone 1 specifications:
- **Finding:** The public TED dataset **does not contain individual user IDs, per-user watch histories, timestamps of user views, or user-level rating interaction logs**.
- **Available Interaction Signals:**
  1. `views`: Cumulative community view counts per talk.
  2. `comments`: Total comment count per talk.
  3. `ratings`: Aggregate community vote tallies across 14 emotional dimensions (e.g., *Inspiring*, *Informative*, *Fascinating*).
- **Project Decision:** Genuine per-user interaction logs **do not exist** in the public archive. In subsequent collaborative filtering modules, this limitation will be accommodated using simulated user interaction matrices or affinity vectors derived from aggregate rating profiles, rather than inventing false user histories.

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
| **Categorical Encoding** | Mapped `top_rating` into `top_rating_code` (0 to 13) | Provides a clean numerical baseline for data preparation without discarding readable strings. |
| **Datetime Normalization** | Converted Unix epochs to `film_datetime`, `published_datetime`, and `published_year` | Enables temporal filtering and recency weighting. |
| **Duration in Minutes** | Derived `duration_minutes = round(duration / 60, 2)` | Standardizes duration for human readability and length-based filtering. |
| **Transcript Text Cleaning** | Removed stage directions like `(Applause)` / `(Laughter)`, lowercased, normalized punctuation and whitespace | Cleans non-lexical acoustic noise while preserving the full verbatim transcript in `transcript`. |
| **Coverage Flagging** | Added `has_transcript` binary column ($1$ if transcript available, $0$ otherwise) | Allows downstream recommendation models to route features dynamically. |

### Before-and-After Validation Report
The automated audit table is saved to [`reports/before_after_validation.csv`](reports/before_after_validation.csv):

| Dataset | Metric | Raw Value | Cleaned Value | Delta | Rationale & Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`ted_main.csv`** | Total Records (Rows) | 2,550 | 2,550 | 0 | 100% of metadata records preserved (0 deleted) |
| **`ted_main.csv`** | Total Features (Columns) | 17 | 28 | +11 | Engineered `clean_tags`, `tag_count`, `total_ratings`, `top_rating`, `top_rating_code`, `duration_minutes`, datetimes |
| **`ted_main.csv`** | Missing Values Count | 6 | **0** | -6 | Imputed 6 missing `speaker_occupation` rows with `'Unknown'` |
| **`ted_main.csv`** | Duplicate Rows | 0 | **0** | 0 | Zero duplicate records in metadata |
| **`ted_main.csv`** | Whitespace-Padded URLs | 2,550 | **0** | -2,550 | Stripped trailing newline (`\n`) and whitespace from all URLs |
| **`transcripts.csv`** | Total Records (Rows) | 2,467 | 2,464 | -3 | Removed 3 confirmed exact duplicate rows; 2,464 unique talks retained |
| **`transcripts.csv`** | Total Features (Columns) | 2 | 7 | +5 | Engineered `clean_transcript`, `char_count`, `word_count`, `clean_char_count`, `clean_word_count` |
| **`transcripts.csv`** | Missing Values Count (Transcript) | 0 | **0** | 0 | Zero missing verbatim transcripts |
| **`transcripts.csv`** | Duplicate Rows | 3 | **0** | -3 | Exact duplicate transcript rows dropped (1 unique row per talk URL) |
| **`transcripts.csv`** | Whitespace-Padded URLs | 2,467 | **0** | -2,467 | Stripped trailing newline (`\n`) and whitespace from all URLs |

*\*Note on non-spoken performances: 8 musical/dance performance talks originally contained only acoustic notes (e.g. `(Music)(Applause)`). When stage notes are stripped, `clean_transcript` correctly reflects 0 words (`""`), while the full verbatim text is preserved in `transcript`.*

---

## Generated Deliverables & Outputs

1. **Cleaned Datasets:**
   - [`data/ted_main_cleaned.csv`](data/ted_main_cleaned.csv) (2,550 rows, 28 columns, 7.97 MB)
   - [`data/transcripts_cleaned.csv`](data/transcripts_cleaned.csv) (2,464 rows, 7 columns, 55.76 MB)
2. **Quality & Validation Reports:**
   - [`reports/data_quality_report.csv`](reports/data_quality_report.csv) (Granular audit across 19 columns)
   - [`reports/before_after_validation.csv`](reports/before_after_validation.csv) (Before/after validation delta matrix)
3. **Exploratory Visualizations:**
   - [`reports/eda/`](reports/eda/) (11 publication-grade PNG charts at 300 DPI)
4. **Interactive Jupyter Notebooks:**
   - [`notebooks/01_data_collection.ipynb`](notebooks/01_data_collection.ipynb) (Fully executed, 24 cells)
   - [`notebooks/02_eda_preprocessing.ipynb`](notebooks/02_eda_preprocessing.ipynb) (Fully executed, 44 cells)
5. **Reusable Python Pipeline:**
   - [`src/data_processing.py`](src/data_processing.py) (Modular, tested end-to-end cleaning and EDA script)
6. **Documentation & Dependencies:**
   - [`DATA_SOURCES.md`](DATA_SOURCES.md) (Data sources, schemas, licensing, and limitations)
   - [`requirements.txt`](requirements.txt) (Pinned package dependencies)
   - [`.gitignore`](.gitignore) (Ignoring virtual environments, caches, and credentials)

---

## Installation & Execution Guide

### 1. Environment Setup
Clone the repository and install required packages:
```bash
# Clone repository
git clone https://github.com/leena-11/ted-talks-recommendation-system.git
cd "ted-talks-recommendation-system"

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
Open:
- `notebooks/01_data_collection.ipynb`
- `notebooks/02_eda_preprocessing.ipynb`

Both notebooks are pre-executed with all cell outputs, tables, and charts visible immediately.

---

## Next Steps (Milestone 2 Roadmap)

Per the Infosys Springboard roadmap, subsequent milestones will implement:
- **Milestone 2: Feature Engineering & Content-Based Recommendation Engine**
  - Text tokenization and lemmatization (NLTK / spaCy).
  - TF-IDF feature extraction on cleaned transcripts, tags, and talk descriptions.
  - Cosine similarity matrix computation and content-based recommendation ranking.
- **Milestone 3: Collaborative Filtering & Model Evaluation**
  - Simulated user interaction modeling.
  - Hybrid recommendation algorithm combining content semantics and rating distributions.
  - Quantitative evaluation using Precision@K, Recall@K, MAP, and NDCG metrics.
- **Milestone 4: Interactive Streamlit Web Application**
  - Interactive UI for real-time talk discovery, tag filtering, and recommendation exploration.
