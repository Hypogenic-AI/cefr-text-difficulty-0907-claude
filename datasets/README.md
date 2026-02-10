# Datasets for CEFR Text Difficulty Research

This directory contains (or provides instructions for downloading) key datasets used in CEFR text difficulty modeling research.

## Quick Setup

```bash
# Activate the project virtual environment
source /workspaces/cefr-text-difficulty-0907-claude/.venv/bin/activate

# Install required libraries
pip install datasets pandas
```

---

## 1. CEFR-SP (English Sentence-Level CEFR Annotations)

**Source:** Arase, Uchida & Kajiwara (2022) -- EMNLP 2022
**HuggingFace:** [UniversalCEFR/cefr_sp_en](https://huggingface.co/datasets/UniversalCEFR/cefr_sp_en)
**Local path:** `datasets/cefr_sp_en/`

The CEFR-SP dataset provides English sentences annotated with CEFR levels (A1--C2) at the sentence level. It is one of the most widely used benchmarks for sentence-level CEFR difficulty classification.

### Download via Python

```python
from datasets import load_dataset
import pandas as pd

# Load from HuggingFace
dataset = load_dataset("UniversalCEFR/cefr_sp_en")

# Save each split to CSV
for split_name in dataset:
    df = dataset[split_name].to_pandas()
    df.to_csv(f"datasets/cefr_sp_en/{split_name}.csv", index=False)
    print(f"Saved {split_name}: {len(df)} rows")
```

---

## 2. Kaggle CEFR-Levelled English Texts

**Source:** A. Montgomerie (Kaggle)
**URL:** https://www.kaggle.com/datasets/amontgomerie/cefr-levelled-english-texts
**Local path:** `datasets/kaggle_cefr_texts/`

A collection of English texts labeled with CEFR levels, useful for document-level CEFR classification.

### Download Instructions

**Option A: Using the Kaggle API**

```bash
# Install Kaggle CLI (if not already installed)
pip install kaggle

# Ensure your Kaggle API key is at ~/.kaggle/kaggle.json
# Download the dataset
kaggle datasets download -d amontgomerie/cefr-levelled-english-texts -p datasets/kaggle_cefr_texts/
unzip datasets/kaggle_cefr_texts/cefr-levelled-english-texts.zip -d datasets/kaggle_cefr_texts/
```

**Option B: Manual Download**

1. Visit https://www.kaggle.com/datasets/amontgomerie/cefr-levelled-english-texts
2. Click "Download" (requires a free Kaggle account)
3. Extract the ZIP file into `datasets/kaggle_cefr_texts/`

---

## 3. UniversalCEFR (Multilingual CEFR Benchmark)

**Source:** Imperial et al. (2025) -- EMNLP 2025
**HuggingFace:** [UniversalCEFR](https://huggingface.co/UniversalCEFR)
**Local path:** `datasets/universalcefr/`

A large-scale multilingual CEFR benchmark covering many languages. Useful for cross-lingual experiments.

### Download via Python

```python
from datasets import load_dataset
import pandas as pd

# Load the multilingual dataset (check HuggingFace for available configs)
dataset = load_dataset("UniversalCEFR/universalcefr")

# Save each split to CSV
for split_name in dataset:
    df = dataset[split_name].to_pandas()
    df.to_csv(f"datasets/universalcefr/{split_name}.csv", index=False)
    print(f"Saved {split_name}: {len(df)} rows")
```

---

## 4. ReadMe++ (Multilingual Multi-Domain Readability)

**Source:** Naous et al. (2023)
**HuggingFace:** Search for "ReadMe++" on HuggingFace Hub
**Local path:** `datasets/readme_plus/`

A multilingual, multi-domain readability assessment benchmark.

### Download via Python

```python
from datasets import load_dataset

# Check HuggingFace for the exact dataset identifier
dataset = load_dataset("tareknaous/readmepp")

for split_name in dataset:
    df = dataset[split_name].to_pandas()
    df.to_csv(f"datasets/readme_plus/{split_name}.csv", index=False)
```

---

## 5. Cambridge Readability Dataset (OneStopEnglish)

**Source:** Xia, Kochmar & Briscoe (2016)
**GitHub:** https://github.com/nishkalavallabhi/OneStopEnglishCorpus
**Local path:** `datasets/onestopenglish/`

A corpus of texts at three readability levels (elementary, intermediate, advanced) derived from the OneStopEnglish news website.

### Download

```bash
git clone https://github.com/nishkalavallabhi/OneStopEnglishCorpus.git datasets/onestopenglish/
```

---

## Dataset Summary Table

| Dataset | Language(s) | Granularity | CEFR Levels | Size | Source |
|---------|-------------|-------------|-------------|------|--------|
| CEFR-SP | English | Sentence | A1--C2 | ~17k sentences | HuggingFace |
| Kaggle CEFR Texts | English | Document | A1--C2 | ~1.1k texts | Kaggle |
| UniversalCEFR | Multilingual | Varies | A1--C2 | Large | HuggingFace |
| ReadMe++ | Multilingual | Document | Varies | Large | HuggingFace |
| OneStopEnglish | English | Document | 3 levels | ~189 articles | GitHub |

---

## Directory Structure

```
datasets/
  README.md            # This file
  .gitignore           # Excludes data files from git
  cefr_sp_en/          # CEFR-SP English sentence-level data
    train.csv
    validation.csv
    test.csv
  kaggle_cefr_texts/   # Kaggle CEFR-levelled English texts
  universalcefr/       # UniversalCEFR multilingual data
  readme_plus/         # ReadMe++ multilingual readability
  onestopenglish/      # OneStopEnglish corpus
```

## Notes

- Large data files (CSV, JSONL, Parquet, etc.) are excluded from version control via `.gitignore`.
- Always check the license and citation requirements for each dataset before use.
- Some datasets require authentication (e.g., Kaggle API key).
