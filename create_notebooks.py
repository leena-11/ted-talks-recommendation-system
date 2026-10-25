"""
Script to generate and execute notebooks/01_data_collection.ipynb and notebooks/02_eda_preprocessing.ipynb.
"""

import sys
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbconvert.preprocessors import ExecutePreprocessor


def build_notebook_1(output_path: Path):
    nb = new_notebook()

    # Cell 1: Markdown
    nb.cells.append(new_markdown_cell(
"""# Module 1: Data Collection & Dataset Relationship Analysis
### Personalized TED Talks Recommendation System Using Machine Learning
**Infosys Springboard Virtual Internship 7.0**

---

## 1. Introduction and Objectives
Welcome to **Module 1: Data Collection** of the *Personalized TED Talks Recommendation System*. In this initial module of the machine learning lifecycle, our primary goals are:
1. **Verify Environment and Data Availability:** Inspect the project directory structure and confirm the existence of `data/ted_main.csv` and `data/transcripts.csv`.
2. **Ingest Datasets:** Load both CSV files using Pandas and verify initial records, shapes, and schemas without modifying original files.
3. **Analyze Identifiers & Candidate Keys:** Evaluate common columns and determine whether `url`, `title`, or other features can serve as a dependable primary/foreign key.
4. **Inspect Overlap & Relationships:** Quantify how many talks in the metadata have matching transcripts, diagnose discrepancies (e.g. why 86 talks lack transcripts), and assess merge viability.
5. **Document Source, Attribution, & Licensing:** Record dataset origins, licensing terms (CC BY-NC-SA 4.0), and known real-world limitations.

---
### Technology Stack for Module 1
- **Python 3.13**
- **Pandas** for structured data manipulation
- **NumPy** for numerical verification
- **Pathlib** for cross-platform filesystem handling
"""
    ))

    # Cell 2: Code - Imports & Setup
    nb.cells.append(new_code_cell(
"""import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Configure pandas display options
pd.set_option('display.max_columns', 25)
pd.set_option('display.max_colwidth', 60)
pd.set_option('display.width', 1000)

print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
"""
    ))

    # Cell 3: Markdown
    nb.cells.append(new_markdown_cell(
"""## 2. Dataset Existence Verification
We first verify that both required raw CSV files are present in the `data/` directory. If either file is missing, the code provides clear instructions on where to locate the dataset."""
    ))

    # Cell 4: Code - Verification
    nb.cells.append(new_code_cell(
"""# Define paths
data_dir = Path("../data") if Path("../data").exists() else Path("data")
main_file = data_dir / "ted_main.csv"
trans_file = data_dir / "transcripts.csv"

print(f"Checking data directory: {data_dir.resolve()}")

files_status = {
    "ted_main.csv": main_file.is_file(),
    "transcripts.csv": trans_file.is_file()
}

for filename, exists in files_status.items():
    status_str = "EXISTS" if exists else "MISSING"
    print(f"  [{status_str}] {filename}")

if not all(files_status.values()):
    missing = [f for f, exists in files_status.items() if not exists]
    raise FileNotFoundError(
        f"Missing required file(s): {missing}.\\n"
        f"Please place 'ted_main.csv' and 'transcripts.csv' inside '{data_dir.resolve()}'."
    )
else:
    print("\\nBoth required dataset files are present and verified!")
"""
    ))

    # Cell 5: Markdown
    nb.cells.append(new_markdown_cell(
"""## 3. Data Ingestion & Initial Inspection
We load both datasets into Pandas DataFrames and examine their shapes, columns, and sample records.
- `df_main`: Talk metadata (titles, speakers, tags, view counts, ratings, publication dates, etc.)
- `df_trans`: Spoken transcript text for TED talks"""
    ))

    # Cell 6: Code - Ingestion
    nb.cells.append(new_code_cell(
"""# Load datasets
df_main = pd.read_csv(main_file)
df_trans = pd.read_csv(trans_file)

print(f"ted_main.csv shape: {df_main.shape[0]:,} rows, {df_main.shape[1]} columns")
print(f"transcripts.csv shape: {df_trans.shape[0]:,} rows, {df_trans.shape[1]} columns")
"""
    ))

    # Cell 7: Markdown
    nb.cells.append(new_markdown_cell(
"""### 3.1 Metadata Dataset (`ted_main.csv`) First 5 Rows & Schema"""
    ))

    # Cell 8: Code - Preview df_main
    nb.cells.append(new_code_cell(
"""# Display first 5 rows of ted_main
df_main.head()
"""
    ))

    # Cell 9: Code - Info df_main
    nb.cells.append(new_code_cell(
"""# Display column data types and non-null counts for ted_main
df_main.info()
"""
    ))

    # Cell 10: Markdown
    nb.cells.append(new_markdown_cell(
"""### 3.2 Transcripts Dataset (`transcripts.csv`) First 5 Rows & Schema"""
    ))

    # Cell 11: Code - Preview df_trans
    nb.cells.append(new_code_cell(
"""# Display first 5 rows of transcripts
df_trans.head()
"""
    ))

    # Cell 12: Code - Info df_trans
    nb.cells.append(new_code_cell(
"""# Display column data types and non-null counts for transcripts
df_trans.info()
"""
    ))

    # Cell 13: Markdown
    nb.cells.append(new_markdown_cell(
"""## 4. Identifier Analysis & Candidate Key Evaluation
To join talk metadata with transcript text, we need a reliable common identifier.
Let us examine:
1. Common column names across both datasets.
2. Uniqueness of candidate keys (`url`, `name`, `title`).
3. Whitespace / newline formatting quirks."""
    ))

    # Cell 14: Code - Identifier Analysis
    nb.cells.append(new_code_cell(
"""# Identify intersecting column names
common_columns = list(set(df_main.columns).intersection(set(df_trans.columns)))
print(f"Common columns between datasets: {common_columns}")

# Inspect uniqueness in metadata
print(f"Total records in ted_main: {len(df_main):,}")
print(f"Unique 'url' values in ted_main: {df_main['url'].nunique():,}")
print(f"Unique 'title' values in ted_main: {df_main['title'].nunique():,}")
print(f"Unique 'name' values in ted_main: {df_main['name'].nunique():,}")

# Inspect uniqueness in transcripts
print(f"\\nTotal records in transcripts: {len(df_trans):,}")
print(f"Unique 'url' values in transcripts: {df_trans['url'].nunique():,}")
"""
    ))

    # Cell 15: Markdown
    nb.cells.append(new_markdown_cell(
"""### 4.1 Diagnosing URL Whitespace and Duplicate Transcripts
Notice that `transcripts.csv` has 2,467 rows but only 2,464 unique URLs (3 duplicate URLs).
Furthermore, let us check whether the URLs contain whitespace or newline (`\\n`) characters."""
    ))

    # Cell 16: Code - URL formatting & duplicates
    nb.cells.append(new_code_cell(
"""# Check sample URL character representation
sample_main_url = df_main['url'].iloc[0]
sample_trans_url = df_trans['url'].iloc[0]
print("Sample ted_main URL representation:", repr(sample_main_url))
print("Sample transcripts URL representation:", repr(sample_trans_url))

# Count trailing newlines
main_has_newline = df_main['url'].str.endswith('\\n').sum()
trans_has_newline = df_trans['url'].str.endswith('\\n').sum()
print(f"ted_main URLs ending with '\\\\n': {main_has_newline} of {len(df_main)}")
print(f"transcripts URLs ending with '\\\\n': {trans_has_newline} of {len(df_trans)}")

# Inspect the duplicate URL records in transcripts.csv
clean_trans_urls = df_trans['url'].str.strip()
dup_mask = clean_trans_urls.duplicated(keep=False)
dup_records = df_trans[dup_mask].copy()
dup_records['clean_url'] = clean_trans_urls[dup_mask]
print(f"\\nDuplicate transcript rows found: {len(dup_records)}")
print(dup_records[['clean_url']].drop_duplicates())
"""
    ))

    # Cell 17: Code - Check identical duplicate transcripts
    nb.cells.append(new_code_cell(
"""# Verify whether duplicate transcript rows are 100% identical
for url, group in dup_records.groupby('clean_url'):
    t0 = group['transcript'].iloc[0]
    t1 = group['transcript'].iloc[1]
    is_identical = (t0 == t1)
    print(f"URL: {url.split('/')[-1]}")
    print(f"  Length: {len(t0):,} characters | Identical text: {is_identical}")
"""
    ))

    # Cell 18: Markdown
    nb.cells.append(new_markdown_cell(
"""## 5. Dataset Overlap & Relationship Inspection
With normalized URLs (stripped of trailing `\\n`), let us analyze the exact overlap between `ted_main.csv` and `transcripts.csv`."""
    ))

    # Cell 19: Code - Set intersection
    nb.cells.append(new_code_cell(
"""main_urls = set(df_main['url'].str.strip())
trans_urls = set(df_trans['url'].str.strip())

common_urls = main_urls.intersection(trans_urls)
main_only_urls = main_urls - trans_urls
trans_only_urls = trans_urls - main_urls

print(f"Total unique talks in ted_main: {len(main_urls):,}")
print(f"Total unique talks in transcripts: {len(trans_urls):,}")
print(f"Common talks present in both datasets: {len(common_urls):,} ({len(common_urls)/len(main_urls):.2%})")
print(f"Talks in ted_main without transcripts: {len(main_only_urls):,} ({len(main_only_urls)/len(main_urls):.2%})")
print(f"Transcripts without metadata: {len(trans_only_urls):,} ({len(trans_only_urls)/len(trans_urls):.2%})")
"""
    ))

    # Cell 20: Markdown
    nb.cells.append(new_markdown_cell(
"""### 5.1 Deep-Dive: Why do 86 talks lack transcripts?
Let us inspect sample talks from `ted_main.csv` that do not have matching transcripts in `transcripts.csv`."""
    ))

    # Cell 21: Code - Inspect missing transcripts
    nb.cells.append(new_code_cell(
"""missing_transcripts_df = df_main[df_main['url'].str.strip().isin(main_only_urls)]
print("Sample talks without transcripts:")
missing_transcripts_df[['title', 'main_speaker', 'speaker_occupation', 'tags']].head(10)
"""
    ))

    # Cell 22: Markdown
    nb.cells.append(new_markdown_cell(
"""**Domain Finding:**
Notice the titles and tags of talks without transcripts:
- *"A dance of 'Symbiosis'"* (dance, performance)
- *"A string quartet plays 'Blue Room'"* (live music, cello)
- *"'Woza'"* (African guitar music)
- *"Piano jazz that rocks"* (instrumental piano)
- *"An electrifying organ performance"* (organ performance)

These 86 talks are **instrumental musical performances, choreographies, and artistic demonstrations**. Because there was no spoken speech, official TED transcripts were naturally not created. This is an authentic domain attribute rather than corrupt or missing data."""
    ))

    # Cell 23: Markdown
    nb.cells.append(new_markdown_cell(
"""## 6. Dataset Documentation, Source, Licensing, & Limitations

| Attribute | Details |
| :--- | :--- |
| **Dataset Source** | Kaggle TED Talks Dataset (`rounakbanik/ted-talks`), scraped from official TED website ([ted.com](https://www.ted.com)). |
| **Curator / Attribution** | Rounak Banik; TED Conferences LLC. |
| **Licensing** | Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) & TED Terms of Use for non-commercial educational use. |
| **Temporal Coverage** | TED Talks filmed from May 1972 through August 2017, and published on ted.com through September 21, 2017. |
| **Metadata Features (`ted_main.csv`)** | 2,550 talks, 17 columns: `comments`, `description`, `duration`, `event`, `film_date`, `languages`, `main_speaker`, `name`, `num_speaker`, `published_date`, `ratings`, `related_talks`, `speaker_occupation`, `tags`, `title`, `url`, `views`. |
| **Transcript Features (`transcripts.csv`)** | 2,467 rows (2,464 unique talks after deduplication), 2 columns: `transcript`, `url`. |
| **Reliable Linking Key** | `url` (after stripping trailing `\\n` newline characters). |
| **Known Limitations** | 1. **Static Snapshot:** Data is capped up to September 2017.<br>2. **No Per-User Interactions:** Lacks individual user watch histories or per-user rating profiles (ratings are aggregate community votes), requiring collaborative filtering to use alternative or simulated interaction matrices.<br>3. **86 Non-Spoken Talks:** Instrumental music, dance, and audio-visual art talks lack transcript text, so content recommendations for these items must rely on `title`, `description`, and `tags`. |
"""
    ))

    # Cell 24: Markdown
    nb.cells.append(new_markdown_cell(
"""## 7. Summary & Next Steps
- **Data Collection (Module 1) is complete.**
- Both datasets have been safely loaded, audited, and mapped via `url`.
- Trailing `\\n` formatting and 3 duplicate transcript rows were diagnosed.
- Proceed to **Module 2 (`02_eda_preprocessing.ipynb`)** for exploratory data analysis, data quality reporting, and end-to-end cleaning."""
    ))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Created notebook: {output_path}")


def build_notebook_2(output_path: Path):
    nb = new_notebook()

    # Cell 1: Markdown
    nb.cells.append(new_markdown_cell(
"""# Module 2: Data Exploration and Data Preprocessing
### Personalized TED Talks Recommendation System Using Machine Learning
**Infosys Springboard Virtual Internship 7.0**

---

## 1. Introduction and Objectives
In **Module 2**, we explore the TED Talks datasets, evaluate data quality, generate rich visualizations, and clean the data for downstream feature engineering and recommendation models.

**Key Objectives:**
1. **Initial Data Assessment:** Audit shapes, data types, descriptive statistics, missing values, duplicates, and blank/padded strings.
2. **Data Quality Documentation:** Export an automated Data Quality Report (`reports/data_quality_report.csv`).
3. **Exploratory Data Analysis (EDA):** Generate and save 11 publication-grade visualizations using Matplotlib and Seaborn.
4. **Data Cleaning & Wrangling:**
   - Preserve raw immutable datasets.
   - Strip leading/trailing whitespace and trailing `\\n` from URLs and text fields.
   - Impute missing `speaker_occupation` (fill with `'Unknown'`).
   - Deduplicate `transcripts.csv` (remove 3 exact duplicate rows).
   - Safely parse string-encoded Python lists (`tags`) and lists of dicts (`ratings`).
   - Clean transcript text (remove stage directions like `(Laughter)` / `(Applause)`, lowercase, normalize whitespace).
   - Engineer useful features (`clean_transcript`, `word_count`, `duration_minutes`, `film_year`, `published_year`, `top_rating`).
5. **Before-and-After Validation:** Verify shape deltas, missing value resolutions, and record retention.
6. **Persist Cleaned Datasets:** Save `data/ted_main_cleaned.csv` and `data/transcripts_cleaned.csv`.
"""
    ))

    # Cell 2: Code - Imports & Setup
    nb.cells.append(new_code_cell(
"""import os
import sys
import ast
import re
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication-quality plot styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "figure.dpi": 150,
})

PRIMARY_COLOR = "#E62B1E"    # TED Red
SECONDARY_COLOR = "#1D3557"  # Deep Navy
ACCENT_COLOR = "#457B9D"     # Slate Blue

# Configure paths and sys.path
project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

data_dir = project_root / "data"
reports_dir = project_root / "reports"
eda_dir = reports_dir / "eda"
eda_dir.mkdir(parents=True, exist_ok=True)

print(f"Project root: {project_root.resolve()}")
print(f"Data dir: {data_dir.resolve()}")
print("Environment setup completed successfully.")
"""
    ))

    # Cell 3: Markdown
    nb.cells.append(new_markdown_cell(
"""## 2. Ingestion of Raw Datasets"""
    ))

    # Cell 4: Code - Load
    nb.cells.append(new_code_cell(
"""df_main_raw = pd.read_csv(data_dir / "ted_main.csv")
df_trans_raw = pd.read_csv(data_dir / "transcripts.csv")

print(f"Raw ted_main shape: {df_main_raw.shape}")
print(f"Raw transcripts shape: {df_trans_raw.shape}")
"""
    ))

    # Cell 5: Markdown
    nb.cells.append(new_markdown_cell(
"""## 3. Initial Data Assessment & Data Quality Audit
Let us evaluate missing values, data types, descriptive statistics, and whitespace issues across both datasets."""
    ))

    # Cell 6: Code - Missing values
    nb.cells.append(new_code_cell(
"""print("=== Missing Values in ted_main.csv ===")
main_nulls = df_main_raw.isnull().sum()
main_nulls_pct = (df_main_raw.isnull().sum() / len(df_main_raw)) * 100
main_missing_df = pd.DataFrame({"Missing_Count": main_nulls, "Missing_Pct (%)": main_nulls_pct.round(3)})
display(main_missing_df[main_missing_df["Missing_Count"] > 0])

print("\\n=== Missing Values in transcripts.csv ===")
trans_nulls = df_trans_raw.isnull().sum()
trans_nulls_pct = (df_trans_raw.isnull().sum() / len(df_trans_raw)) * 100
trans_missing_df = pd.DataFrame({"Missing_Count": trans_nulls, "Missing_Pct (%)": trans_nulls_pct.round(3)})
display(trans_missing_df)
"""
    ))

    # Cell 7: Code - Descriptive statistics
    nb.cells.append(new_code_cell(
"""print("=== Numerical Descriptive Statistics (ted_main.csv) ===")
df_main_raw.describe().round(2)
"""
    ))

    # Cell 8: Code - Whitespace and padding audit
    nb.cells.append(new_code_cell(
"""print("=== Whitespace / Newline Padding Audit ===")
padding_report = []
for col in df_main_raw.select_dtypes(include="object").columns:
    s = df_main_raw[col].dropna().astype(str)
    blank_cnt = (s.str.strip() == "").sum()
    padded_cnt = (s != s.str.strip()).sum()
    padding_report.append({"Dataset": "ted_main", "Column": col, "Blank_Strings": blank_cnt, "Padded_Records": padded_cnt})

for col in df_trans_raw.select_dtypes(include="object").columns:
    s = df_trans_raw[col].dropna().astype(str)
    blank_cnt = (s.str.strip() == "").sum()
    padded_cnt = (s != s.str.strip()).sum()
    padding_report.append({"Dataset": "transcripts", "Column": col, "Blank_Strings": blank_cnt, "Padded_Records": padded_cnt})

display(pd.DataFrame(padding_report))
"""
    ))

    # Cell 9: Markdown
    nb.cells.append(new_markdown_cell(
"""### 3.1 Generate Automated Data Quality Report
We compile a structured audit report summarizing every column's type, non-null count, missing count, missing percentage, blank strings, whitespace padding, unique values, and diagnostic notes."""
    ))

    # Cell 10: Code - Data quality report
    nb.cells.append(new_code_cell(
"""# Import data quality report function from src or compute directly
from src.data_processing import generate_data_quality_report

quality_report_csv = reports_dir / "data_quality_report.csv"
quality_df = generate_data_quality_report(df_main_raw, df_trans_raw, quality_report_csv)
print(f"Data quality report saved to: {quality_report_csv.resolve()}")
quality_df
"""
    ))

    # Cell 11: Markdown
    nb.cells.append(new_markdown_cell(
"""## 4. Exploratory Data Analysis (EDA)
We now generate and analyze 11 comprehensive visualizations representing the distributions, correlations, and characteristics of TED Talks."""
    ))

    # Cell 12: Markdown & Code - Chart 1: Top Tags
    nb.cells.append(new_markdown_cell(
"""### 4.1 Most Frequently Occurring TED Talk Tags
TED talks are categorized by subject tags. Let us extract and rank the top 20 most frequent tags."""
    ))
    nb.cells.append(new_code_cell(
"""tag_counts = Counter()
for t_str in df_main_raw["tags"]:
    try:
        t_list = ast.literal_eval(t_str)
        for t in t_list:
            tag_counts[str(t).strip().lower()] += 1
    except Exception:
        pass

top_tags = tag_counts.most_common(20)
tags_df = pd.DataFrame(top_tags, columns=["Tag", "Frequency"])

plt.figure(figsize=(10, 6))
bars = plt.barh(tags_df["Tag"][::-1], tags_df["Frequency"][::-1], color=PRIMARY_COLOR, alpha=0.9)
plt.title("Top 20 Most Frequent TED Talk Tags", fontsize=14, weight="bold", pad=12)
plt.xlabel("Frequency (Number of Talks)")
plt.ylabel("Tag Name")
plt.xlim(0, max(tags_df["Frequency"]) * 1.12)

for bar in bars:
    w = bar.get_width()
    plt.text(w + 12, bar.get_y() + bar.get_height() / 2, f"{int(w):,}", va="center", ha="left", fontsize=9)

plt.tight_layout()
plt.savefig(eda_dir / "01_top_tags.png", bbox_inches="tight")
plt.show()
"""
    ))

    # Cell 13: Markdown & Code - Chart 2: Duration Distribution
    nb.cells.append(new_markdown_cell(
"""### 4.2 Distribution of Talk Durations
The core TED format historically limits talks to 18 minutes (1,080 seconds). Let us inspect the duration distribution in minutes."""
    ))
    nb.cells.append(new_code_cell(
"""durations_min = df_main_raw["duration"] / 60.0
median_dur = durations_min.median()
mean_dur = durations_min.mean()

plt.figure(figsize=(10, 5.5))
sns.histplot(durations_min, kde=True, bins=40, color=SECONDARY_COLOR, edgecolor="white", alpha=0.85)
plt.axvline(18.0, color=PRIMARY_COLOR, linestyle="--", linewidth=2, label="TED 18-min Target")
plt.axvline(median_dur, color="#2A9D8F", linestyle="-.", linewidth=2, label=f"Median ({median_dur:.1f} min)")
plt.axvline(mean_dur, color="#E76F51", linestyle=":", linewidth=2, label=f"Mean ({mean_dur:.1f} min)")

plt.title("Distribution of TED Talk Durations (Minutes)", fontsize=14, weight="bold", pad=12)
plt.xlabel("Duration (Minutes)")
plt.ylabel("Number of Talks")
plt.xlim(0, 45)
plt.legend(frameon=True, facecolor="white")

plt.tight_layout()
plt.savefig(eda_dir / "02_duration_distribution.png", bbox_inches="tight")
plt.show()
"""
    ))

    # Cell 14: Markdown & Code - Chart 3: Views Distribution
    nb.cells.append(new_markdown_cell(
"""### 4.3 Distribution of Talk Views
TED talk view counts span from ~50,000 to over 47 million. We examine both linear (in millions) and logarithmic scales."""
    ))
    nb.cells.append(new_code_cell(
"""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

views_m = df_main_raw["views"] / 1e6
sns.histplot(views_m, bins=45, color=ACCENT_COLOR, kde=True, ax=ax1, edgecolor="white")
ax1.set_title("Distribution of Views (Linear Scale in Millions)", weight="bold")
ax1.set_xlabel("Views (Millions)")
ax1.set_ylabel("Talk Count")
ax1.axvline(views_m.median(), color=PRIMARY_COLOR, linestyle="--", label=f"Median: {views_m.median():.2f}M")
ax1.legend()

sns.histplot(df_main_raw["views"], bins=45, log_scale=True, color=SECONDARY_COLOR, kde=True, ax=ax2, edgecolor="white")
ax2.set_title("Distribution of Views (Logarithmic Scale)", weight="bold")
ax2.set_xlabel("Views (Log Scale)")
ax2.set_ylabel("Talk Count")

plt.tight_layout()
plt.savefig(eda_dir / "03_views_distribution.png", bbox_inches="tight")
plt.show()
"""
    ))

    # Cell 15: Markdown & Code - Chart 4: Ratings Distribution
    nb.cells.append(new_markdown_cell(
"""### 4.4 Aggregate Distribution of Ratings
Audience members rate TED talks across 14 emotional dimensions (e.g. *Inspiring*, *Informative*, *Fascinating*)."""
    ))
    nb.cells.append(new_code_cell(
"""rating_totals = Counter()
for r_str in df_main_raw["ratings"]:
    try:
        r_list = ast.literal_eval(r_str)
        for item in r_list:
            if isinstance(item, dict):
                rating_totals[item.get("name", "Unknown")] += item.get("count", 0)
    except Exception:
        pass

ratings_df = pd.DataFrame(rating_totals.most_common(), columns=["Rating", "Total_Votes"])

plt.figure(figsize=(10, 6))
bars = plt.barh(ratings_df["Rating"][::-1], ratings_df["Total_Votes"][::-1] / 1000, color="#2A9D8F", alpha=0.9)
plt.title("Aggregate TED Talk Audience Ratings (Thousands of Votes)", fontsize=14, weight="bold", pad=12)
plt.xlabel("Total Votes (in Thousands)")
plt.ylabel("Rating Category")
plt.xlim(0, (max(ratings_df["Total_Votes"]) / 1000) * 1.15)

for bar in bars:
    w = bar.get_width()
    plt.text(w + 15, bar.get_y() + bar.get_height() / 2, f"{int(w):,}k", va="center", ha="left", fontsize=9)

plt.tight_layout()
plt.savefig(eda_dir / "04_ratings_distribution.png", bbox_inches="tight")
plt.show()
"""
    ))

    # Cell 16: Markdown & Code - Chart 5: Missing Values
    nb.cells.append(new_markdown_cell(
"""### 4.5 Data Completeness & Missing Values Audit"""
    ))
    nb.cells.append(new_code_cell(
"""missing_dict = {
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
bar_colors = ["#E76F51" if c > 0 else "#2A9D8F" for c in missing_df["Missing_Count"]]
bars = plt.bar(missing_df["Field"], missing_df["Missing_Count"], color=bar_colors, width=0.6)
plt.title("Data Completeness and Missing Values Audit", fontsize=14, weight="bold", pad=12)
plt.xlabel("Field Evaluated")
plt.ylabel("Missing Count (Rows)")
plt.xticks(rotation=30, ha="right")
plt.ylim(0, 100)

for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, h + 2, f"{int(h)}", ha="center", va="bottom", fontsize=10, weight="bold")

plt.tight_layout()
plt.savefig(eda_dir / "05_missing_values.png", bbox_inches="tight")
plt.show()
"""
    ))

    # Cell 17: Markdown & Code - Chart 6: Transcript Lengths
    nb.cells.append(new_markdown_cell(
"""### 4.6 Transcript Word Count Distribution
We examine the word count distribution across all transcripts."""
    ))
    nb.cells.append(new_code_cell(
"""word_counts = df_trans_raw["transcript"].fillna("").astype(str).apply(lambda x: len(x.split()))
median_words = word_counts.median()
mean_words = word_counts.mean()

plt.figure(figsize=(10, 5.5))
sns.histplot(word_counts, kde=True, bins=40, color="#264653", edgecolor="white", alpha=0.85)
plt.axvline(median_words, color=PRIMARY_COLOR, linestyle="--", linewidth=2, label=f"Median ({int(median_words):,} words)")
plt.axvline(mean_words, color="#E76F51", linestyle="-.", linewidth=2, label=f"Mean ({int(mean_words):,} words)")

plt.title("Distribution of Transcript Word Counts", fontsize=14, weight="bold", pad=12)
plt.xlabel("Word Count per Talk")
plt.ylabel("Number of Talks")
plt.legend(frameon=True, facecolor="white")

plt.tight_layout()
plt.savefig(eda_dir / "06_transcript_length_distribution.png", bbox_inches="tight")
plt.show()
"""
    ))

    # Cell 18: Markdown & Code - Chart 7: Top Speakers
    nb.cells.append(new_markdown_cell(
"""### 4.7 Top 15 Most Frequent TED Talk Speakers"""
    ))
    nb.cells.append(new_code_cell(
"""top_speakers = df_main_raw["main_speaker"].str.strip().value_counts().head(15).reset_index()
top_speakers.columns = ["Speaker", "Talk_Count"]

plt.figure(figsize=(10, 5.5))
bars = plt.barh(top_speakers["Speaker"][::-1], top_speakers["Talk_Count"][::-1], color=SECONDARY_COLOR, alpha=0.9)
plt.title("Top 15 Most Frequent TED Talk Speakers", fontsize=14, weight="bold", pad=12)
plt.xlabel("Number of Talks")
plt.ylabel("Speaker Name")
plt.xlim(0, max(top_speakers["Talk_Count"]) + 1.5)

for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.15, bar.get_y() + bar.get_height() / 2, f"{int(w)}", va="center", ha="left", fontsize=9)

plt.tight_layout()
plt.savefig(eda_dir / "07_top_speakers.png", bbox_inches="tight")
plt.show()
"""
    ))

    # Cell 19: Markdown & Code - Chart 8: Top Occupations
    nb.cells.append(new_markdown_cell(
"""### 4.8 Top 15 Speaker Occupations"""
    ))
    nb.cells.append(new_code_cell(
"""top_occ = df_main_raw["speaker_occupation"].dropna().str.strip().value_counts().head(15).reset_index()
top_occ.columns = ["Occupation", "Count"]

plt.figure(figsize=(10, 5.5))
bars = plt.barh(top_occ["Occupation"][::-1], top_occ["Count"][::-1], color=ACCENT_COLOR, alpha=0.9)
plt.title("Top 15 TED Speaker Occupations", fontsize=14, weight="bold", pad=12)
plt.xlabel("Number of Speakers")
plt.ylabel("Occupation")
plt.xlim(0, max(top_occ["Count"]) + 6)

for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.8, bar.get_y() + bar.get_height() / 2, f"{int(w)}", va="center", ha="left", fontsize=9)

plt.tight_layout()
plt.savefig(eda_dir / "08_top_occupations.png", bbox_inches="tight")
plt.show()
"""
    ))

    # Cell 20: Markdown & Code - Chart 9: Views vs Comments
    nb.cells.append(new_markdown_cell(
"""### 4.9 Views vs. Comments Correlation"""
    ))
    nb.cells.append(new_code_cell(
"""plt.figure(figsize=(9, 6))
corr = df_main_raw[["views", "comments"]].corr().iloc[0, 1]

sns.regplot(
    x="views", y="comments", data=df_main_raw,
    scatter_kws={"alpha": 0.45, "color": SECONDARY_COLOR, "s": 25},
    line_kws={"color": PRIMARY_COLOR, "linewidth": 2}
)
plt.xscale("log")
plt.yscale("log")
plt.title(f"Views vs. Comments Correlation (Pearson r = {corr:.2f})", fontsize=14, weight="bold", pad=12)
plt.xlabel("Views (Log Scale)")
plt.ylabel("Comments (Log Scale)")

plt.tight_layout()
plt.savefig(eda_dir / "09_views_vs_comments.png", bbox_inches="tight")
plt.show()
"""
    ))

    # Cell 21: Markdown & Code - Chart 10: Publication Trends
    nb.cells.append(new_markdown_cell(
"""### 4.10 Annual Publication Trend (2006 - 2017)"""
    ))
    nb.cells.append(new_code_cell(
"""pub_years = pd.to_datetime(df_main_raw["published_date"], unit="s").dt.year.value_counts().sort_index().reset_index()
pub_years.columns = ["Year", "Talk_Count"]

plt.figure(figsize=(10, 5))
plt.plot(pub_years["Year"], pub_years["Talk_Count"], marker="o", linewidth=2.5, color=PRIMARY_COLOR, markersize=7)
plt.fill_between(pub_years["Year"], pub_years["Talk_Count"], color=PRIMARY_COLOR, alpha=0.15)
plt.title("TED Talks Published by Year (2006 - 2017)", fontsize=14, weight="bold", pad=12)
plt.xlabel("Year Published")
plt.ylabel("Number of Talks")
plt.xticks(pub_years["Year"])

for _, row in pub_years.iterrows():
    plt.text(row["Year"], row["Talk_Count"] + 7, f"{int(row['Talk_Count'])}", ha="center", fontsize=9, weight="bold")

plt.ylim(0, max(pub_years["Talk_Count"]) * 1.18)
plt.tight_layout()
plt.savefig(eda_dir / "10_talks_by_year.png", bbox_inches="tight")
plt.show()
"""
    ))

    # Cell 22: Markdown & Code - Chart 11: Transcript Coverage
    nb.cells.append(new_markdown_cell(
"""### 4.11 Transcript Coverage Analysis (Donut Chart)"""
    ))
    nb.cells.append(new_code_cell(
"""clean_main_urls = set(df_main_raw["url"].str.strip())
clean_trans_urls = set(df_trans_raw["url"].str.strip())
common = len(clean_main_urls.intersection(clean_trans_urls))
unmatched = len(clean_main_urls - clean_trans_urls)

labels = [
    f"Spoken Talks with Transcripts\\n({common:,} talks - {common/len(clean_main_urls):.1%})",
    f"Music / Dance / Art Demonstrations\\n({unmatched:,} talks - {unmatched/len(clean_main_urls):.1%})"
]
sizes = [common, unmatched]
colors = ["#2A9D8F", "#E76F51"]

fig, ax = plt.subplots(figsize=(7, 6))
wedges, texts = ax.pie(
    sizes, labels=labels, colors=colors, startangle=140,
    wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
    textprops=dict(fontsize=10, weight="bold")
)
ax.set_title("Transcript Coverage in TED Dataset", fontsize=14, weight="bold", pad=14)

plt.tight_layout()
plt.savefig(eda_dir / "11_transcript_coverage.png", bbox_inches="tight")
plt.show()
"""
    ))

    # Cell 23: Markdown
    nb.cells.append(new_markdown_cell(
"""## 5. Data Cleaning & Preprocessing Pipeline
We now apply our production-grade cleaning transformations via functions defined in `src/data_processing.py`:
1. Stripping trailing `\\n` and whitespace from URLs and text fields.
2. Imputing missing `speaker_occupation` with `'Unknown'`.
3. Deduplicating `transcripts.csv` by removing the 3 exact duplicate rows.
4. Parsing tags into clean token lists and strings.
5. Extracting `total_ratings` and `top_rating` from JSON-like rating dicts.
6. Cleaning transcript text (removing stage directions, lowercase, normalizing whitespace).
7. Flagging `has_transcript`."""
    ))

    # Cell 24: Code - Execute Cleaning
    nb.cells.append(new_code_cell(
"""from src.data_processing import (
    preprocess_ted_main,
    preprocess_transcripts,
    validate_cleaning
)

# Common clean URLs set
common_urls = set(df_trans_raw["url"].str.strip())

# Clean DataFrames
df_main_clean = preprocess_ted_main(df_main_raw, common_urls=common_urls)
df_trans_clean = preprocess_transcripts(df_trans_raw)

print(f"Cleaned ted_main shape: {df_main_clean.shape}")
print(f"Cleaned transcripts shape: {df_trans_clean.shape}")
"""
    ))

    # Cell 25: Markdown
    nb.cells.append(new_markdown_cell(
"""### 5.1 Preview Cleaned Metadata (`ted_main_cleaned.csv`)"""
    ))
    nb.cells.append(new_code_cell(
"""df_main_clean[["title", "main_speaker", "speaker_occupation", "clean_tags", "top_rating", "duration_minutes", "has_transcript"]].head()
"""
    ))

    # Cell 26: Markdown
    nb.cells.append(new_markdown_cell(
"""### 5.2 Preview Cleaned Transcripts (`transcripts_cleaned.csv`)"""
    ))
    nb.cells.append(new_code_cell(
"""df_trans_clean[["url", "word_count", "clean_word_count", "clean_transcript"]].head()
"""
    ))

    # Cell 27: Markdown
    nb.cells.append(new_markdown_cell(
"""## 6. Before-and-After Validation
We formally validate the before-and-after states to verify data integrity."""
    ))

    # Cell 28: Code - Validation
    nb.cells.append(new_code_cell(
"""val = validate_cleaning(df_main_raw, df_main_clean, df_trans_raw, df_trans_clean)

val_df = pd.DataFrame([
    {
        "Dataset": "ted_main",
        "Raw_Shape": str(val["metadata"]["raw_shape"]),
        "Clean_Shape": str(val["metadata"]["clean_shape"]),
        "Raw_Nulls": val["metadata"]["raw_nulls"],
        "Clean_Nulls": val["metadata"]["clean_nulls"],
        "Records_Retained": val["metadata"]["records_retained"],
        "Duplicates_Removed": val["metadata"]["records_removed"]
    },
    {
        "Dataset": "transcripts",
        "Raw_Shape": str(val["transcripts"]["raw_shape"]),
        "Clean_Shape": str(val["transcripts"]["clean_shape"]),
        "Raw_Nulls": val["transcripts"]["raw_nulls"],
        "Clean_Nulls": val["transcripts"]["clean_nulls"],
        "Records_Retained": val["transcripts"]["records_retained"],
        "Duplicates_Removed": val["transcripts"]["records_removed"]
    }
])
display(val_df)

print("\\n=== Data Cleaning Decisions & Rationale ===")
display(pd.DataFrame(val["cleaning_decisions"]))
"""
    ))

    # Cell 29: Markdown
    nb.cells.append(new_markdown_cell(
"""## 7. Persisting Cleaned Datasets"""
    ))

    # Cell 30: Code - Save outputs
    nb.cells.append(new_code_cell(
"""clean_main_csv = data_dir / "ted_main_cleaned.csv"
clean_trans_csv = data_dir / "transcripts_cleaned.csv"

df_main_clean.to_csv(clean_main_csv, index=False)
df_trans_clean.to_csv(clean_trans_csv, index=False)

print(f"Saved: {clean_main_csv.resolve()} ({clean_main_csv.stat().st_size:,} bytes)")
print(f"Saved: {clean_trans_csv.resolve()} ({clean_trans_csv.stat().st_size:,} bytes)")
"""
    ))

    # Cell 31: Markdown
    nb.cells.append(new_markdown_cell(
"""## 8. Summary & Next Steps
- **Module 2 is complete!**
- The datasets have been thoroughly explored, visualized, and cleaned.
- Cleaned files `data/ted_main_cleaned.csv` and `data/transcripts_cleaned.csv` are saved and ready for **Module 3 (Feature Engineering)**.
"""
    ))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Created notebook: {output_path}")


def execute_notebook(nb_path: Path):
    print(f"Executing notebook: {nb_path}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(nb_path.parent)}})

    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Execution successful and saved: {nb_path}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    nb1_path = base_dir / "notebooks" / "01_data_collection.ipynb"
    nb2_path = base_dir / "notebooks" / "02_eda_preprocessing.ipynb"

    build_notebook_1(nb1_path)
    build_notebook_2(nb2_path)

    execute_notebook(nb1_path)
    execute_notebook(nb2_path)
