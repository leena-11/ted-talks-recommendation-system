# Data Sources and Attribution
## Personalized TED Talks Recommendation System Using Machine Learning
**Infosys Springboard Virtual Internship 7.0 — Milestone 1 (Module 1 & Module 2)**

---

## 1. Overview
This document records the provenance, licensing, field schemas, collection methodology, and known limitations of the datasets utilized in the **Personalized TED Talks Recommendation System**.

Two core datasets are employed in Milestone 1:
1. **Talk Metadata (`data/ted_main.csv`)**
2. **Talk Transcripts (`data/transcripts.csv`)**

---

## 2. Dataset Attribution & Provenance

| Attribute | Details |
| :--- | :--- |
| **Dataset Title** | TED Talks Dataset (Talk Metadata and Transcripts) |
| **Original Curator** | Rounak Banik |
| **Primary Platform** | [Kaggle: TED Talks Dataset](https://www.kaggle.com/datasets/rounakbanik/ted-talks) |
| **Original Source** | Official TED Talks Archive ([ted.com](https://www.ted.com)) |
| **Collection Methodology** | Web scraping using Python (BeautifulSoup and requests) targeting public talk pages on ted.com. |
| **Temporal Coverage** | Talks filmed between May 14, 1972 and August 27, 2017; published on ted.com between June 27, 2006 and September 21, 2017. |
| **Licensing** | Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)) & [TED.com Terms of Use](https://www.ted.com/about/our-organization/our-policies-terms/ted-talks-usage-policy) for non-commercial educational use. |
| **Attribution Requirement** | TED Conferences LLC and Rounak Banik must be credited for data extraction and curation. |

---

## 3. Dataset Specifications & Schemas

### 3.1 Metadata Dataset: `data/ted_main.csv`
- **Total Records:** 2,550 talks
- **Total Columns:** 17 columns
- **File Size:** ~7.58 MB (raw CSV)

| Field Name | Data Type | Non-Null Count | Description & Constraints |
| :--- | :--- | :--- | :--- |
| `comments` | `int64` | 2,550 | Total number of first-level comments posted on ted.com (range: 2 to 6,404). |
| `description` | `object` | 2,550 | Short synopsis of the talk contents provided by TED. |
| `duration` | `int64` | 2,550 | Talk duration in seconds (range: 135 to 5,256 seconds; median: 848 seconds / 14.1 minutes). |
| `event` | `object` | 2,550 | Name of the TED event where the talk took place (e.g. `TED2006`, `TEDGlobal 2012`, `TEDx`). |
| `film_date` | `int64` | 2,550 | Unix epoch timestamp (seconds) when the talk was recorded. |
| `languages` | `int64` | 2,550 | Count of available subtitle language translations (range: 0 to 72). |
| `main_speaker` | `object` | 2,550 | First speaker listed for the talk. |
| `name` | `object` | 2,550 | Combined string of speaker and talk title (e.g., `Ken Robinson: Do schools kill creativity?`). |
| `num_speaker` | `int64` | 2,550 | Number of speakers featured (range: 1 to 5). |
| `published_date` | `int64` | 2,550 | Unix epoch timestamp (seconds) when the talk was published on ted.com. |
| `ratings` | `object` | 2,550 | String-encoded Python list of dictionaries with 14 affective vote counts (e.g., `Funny`, `Inspiring`, `Informative`). |
| `related_talks` | `object` | 2,550 | String-encoded list of dictionaries containing related talk IDs and titles recommended by TED. |
| `speaker_occupation` | `object` | 2,544 | Professional title/background of the primary speaker (6 missing values: 0.235%). |
| `tags` | `object` | 2,550 | String-encoded Python list of category tags assigned by TED curators. |
| `title` | `object` | 2,550 | Official title of the talk. |
| `url` | `object` | 2,550 | Official canonical URL to the talk page on ted.com (contained trailing `\n` in raw file). |
| `views` | `int64` | 2,550 | Cumulative view count on ted.com (range: 50,443 to 47,227,110). |

---

### 3.2 Transcripts Dataset: `data/transcripts.csv`
- **Total Records:** 2,467 rows (2,464 unique talks after deduplication)
- **Total Columns:** 2 columns
- **File Size:** ~28.53 MB (raw CSV)

| Field Name | Data Type | Non-Null Count | Description & Constraints |
| :--- | :--- | :--- | :--- |
| `transcript` | `object` | 2,467 | Verbatim English transcript of the talk, including audience and acoustic cues like `(Applause)` or `(Laughter)`. |
| `url` | `object` | 2,467 | Canonical talk page URL on ted.com (contained trailing `\n` in raw file). |

---

## 4. Key Relationships & Matching Analysis

### 4.1 Candidate Key Evaluation
- `name` is unique in `ted_main.csv` (2,550 unique values), but is not present in `transcripts.csv`.
- `title` is unique in `ted_main.csv` (2,550 unique values), but is not present in `transcripts.csv`.
- `url` is present in both datasets and serves as the **linking foreign key**.

### 4.2 Formatting Anomaly Resolved
Every single `url` entry in both raw datasets ended with a trailing newline character (`\n`) and trailing whitespace padding:
```python
# Raw representation
"https://www.ted.com/talks/ken_robinson_says_schools_kill_creativity\n"

# Normalized representation
"https://www.ted.com/talks/ken_robinson_says_schools_kill_creativity"
```
Trimming whitespace across all records (`df['url'].str.strip()`) resolved the discrepancy and enabled exact 1-to-1 matching.

### 4.3 Deduplication of Transcripts
In raw `transcripts.csv`, **3 URLs were duplicated** (rows 1107–1109 vs 1114–1116):
- `https://www.ted.com/talks/brene_brown_listening_to_shame`
- `https://www.ted.com/talks/jonathan_haidt_humanity_s_stairway_to_self_transcendence`
- `https://www.ted.com/talks/rob_reid_the_8_billion_ipod`

Comparison confirmed that the transcript text between duplicate pairs was 100% identical. The duplicate rows were dropped, leaving **2,464 unique talk transcripts**.

### 4.4 Overlap & Coverage Summary
- **Common Talks (both metadata and transcripts):** 2,464 talks (**96.63%** of metadata)
- **Metadata Only (no transcript):** 86 talks (**3.37%** of metadata)
- **Transcripts Only (no metadata):** 0 talks (**0.00%**)

---

## 5. Domain Investigation: Missing Transcripts
Inspection of the 86 talks lacking transcripts revealed titles and categories such as:
- *"A dance of 'Symbiosis'"* (Choreography/dance)
- *"A string quartet plays 'Blue Room'"* (Live cello/string quartet)
- *"'Woza'"* and *"'M'Bifo'"* (Live guitar performance)
- *"Piano jazz that rocks"* (Instrumental piano)
- *"An electrifying organ performance"* (Organ showcase)

**Domain Finding:** These 86 talks are **instrumental musical performances, choreographies, and non-verbal artistic showcases**. Because there was no spoken speech, official TED transcripts were naturally never created.
This is genuine domain behavior rather than data corruption. Recommendation pipelines in later modules will use `title`, `description`, and `clean_tags` for content modeling on these non-spoken items.

---

## 6. Investigation of User-Item Interaction Data

In accordance with Milestone 1 specifications:
- **Finding:** The public TED dataset **does not contain individual user IDs, per-user watch histories, timestamps of user views, or user-level rating interaction logs**.
- **Available Interaction Signals:**
  1. `views`: Cumulative community view counts per talk.
  2. `comments`: Total comment count per talk.
  3. `ratings`: Aggregate community vote tallies across 14 emotional dimensions (e.g., *Inspiring*, *Informative*, *Fascinating*).
- **Project Decision:** Genuine per-user interaction logs **do not exist** in the public archive. In subsequent collaborative filtering modules, this limitation will be accommodated using simulated user interaction matrices or affinity vectors derived from aggregate rating profiles, rather than inventing false user histories.

---

## 7. Licensing & Compliance Guidelines

1. **Non-Commercial Use:** The dataset is used exclusively for non-commercial educational and academic evaluation as part of the Infosys Springboard Virtual Internship 7.0.
2. **Attribution:** Any publication or presentation derived from this work must cite TED Conferences LLC and Rounak Banik.
3. **Data Integrity:** Original datasets (`data/ted_main.csv` and `data/transcripts.csv`) remain immutable and unmutated in the repository. Cleaned transformations are stored separately as `ted_main_cleaned.csv` and `transcripts_cleaned.csv`.
