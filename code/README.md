# Code Repositories for CEFR Text Difficulty Research

This directory contains cloned repositories and references to key code resources for CEFR-based text difficulty modeling.

---

## CEFR Classification

### CEFR-SP (Sentence-level CEFR Prediction)
- **Repository:** https://github.com/yukiar/CEFR-SP
- **Paper:** Arase, Uchida & Kajiwara (2022), EMNLP 2022
- **Description:** Official code for CEFR-Based Sentence Difficulty Annotation and Assessment. Includes the CEFR-SP dataset, annotation guidelines, and baseline models for sentence-level CEFR classification.
- **Local path:** `code/CEFR-SP/`

```bash
git clone https://github.com/yukiar/CEFR-SP.git code/CEFR-SP
```

### UniversalCEFRScoring
- **Repository:** https://github.com/nishkalavallabhi/UniversalCEFRScoring
- **Paper:** Vajjala & Rama (2018), BEA Workshop
- **Description:** Code for experiments on universal CEFR classification across multiple languages. Includes feature extraction and classification pipelines.
- **Local path:** `code/UniversalCEFRScoring/`

```bash
git clone https://github.com/nishkalavallabhi/UniversalCEFRScoring.git code/UniversalCEFRScoring
```

### UniversalCEFR (Imperial et al., 2025)
- **Repository:** https://github.com/Joeyetinghan/UniversalCEFR (check HuggingFace for latest)
- **Paper:** Imperial et al. (2025), EMNLP 2025
- **Description:** Large-scale multilingual CEFR benchmark with evaluation code and pre-trained models.

### From Tarzan to Tolkien (CEFR-level LLM Control)
- **Repository:** https://github.com/Ukraine-Davinci/Tarzan-to-Tolkien (check paper for latest URL)
- **Paper:** Malik et al. (2024), ACL Findings 2024
- **Description:** Methods for controlling the language proficiency level of LLM outputs to match target CEFR levels.

---

## Feature Extraction Tools

### LFTK (Linguistic Feature Toolkit)
- **Repository:** https://github.com/brucewlee/lftk
- **Description:** A comprehensive Python library for extracting linguistic features from text. Supports 200+ features across lexical, syntactic, discourse, and readability dimensions. Useful for building feature-based CEFR classifiers.
- **Local path:** `code/lftk/`

```bash
git clone https://github.com/brucewlee/lftk.git code/lftk
```

### spaCy
- **Repository:** https://github.com/explosion/spaCy
- **Description:** Industrial-strength NLP library. Provides tokenization, POS tagging, dependency parsing, and NER, which are foundational for many linguistic feature extraction pipelines used in CEFR research.
- **Install:** `pip install spacy`

### Stanza (Stanford NLP)
- **Repository:** https://github.com/stanfordnlp/stanza
- **Description:** Multilingual NLP toolkit from Stanford. Provides tokenization, POS, lemmatization, dependency parsing for 60+ languages. Important for multilingual CEFR experiments.
- **Install:** `pip install stanza`

---

## Surprisal / Language Model Features

### surprisal
- **Repository:** https://github.com/aalto-speech/surprisal (or pip install)
- **Description:** Python library for computing token-level surprisal values from language models. Surprisal is a key psycholinguistic feature used in readability and difficulty prediction.
- **Install:** `pip install surprisal`

### minicons
- **Repository:** https://github.com/kanishkamisra/minicons
- **Description:** A library for computing behavioral and representational analyses of transformer language models. Useful for extracting token-level log-probabilities and surprisal scores.
- **Install:** `pip install minicons`

### Hugging Face Transformers
- **Repository:** https://github.com/huggingface/transformers
- **Description:** The standard library for working with pre-trained transformer models (BERT, GPT-2, etc.). Used extensively for fine-tuning CEFR classifiers and computing LM-based features.
- **Install:** `pip install transformers`

---

## Utility Libraries

### datasets (Hugging Face)
- **Repository:** https://github.com/huggingface/datasets
- **Description:** Library for accessing and processing datasets from the HuggingFace Hub. Used to load CEFR-SP, UniversalCEFR, and other benchmark datasets.
- **Install:** `pip install datasets`

### scikit-learn
- **Repository:** https://github.com/scikit-learn/scikit-learn
- **Description:** Machine learning library for Python. Used for training traditional ML classifiers (SVM, Random Forest, Logistic Regression) that serve as baselines in CEFR research.
- **Install:** `pip install scikit-learn`

### OneStopEnglishCorpus
- **Repository:** https://github.com/nishkalavallabhi/OneStopEnglishCorpus
- **Description:** Corpus of news articles at three reading levels (elementary, intermediate, advanced). Often used as a readability benchmark in conjunction with CEFR studies.

```bash
git clone https://github.com/nishkalavallabhi/OneStopEnglishCorpus.git code/OneStopEnglishCorpus
```

---

## Cloned Repositories Summary

| Repository | Category | Local Path | Status |
|------------|----------|------------|--------|
| CEFR-SP | CEFR Classification | `code/CEFR-SP/` | Cloned |
| lftk | Feature Extraction | `code/lftk/` | Cloned |
| UniversalCEFRScoring | CEFR Classification | `code/UniversalCEFRScoring/` | Cloned |

## Quick Clone All

```bash
# Clone all key repositories at once
git clone https://github.com/yukiar/CEFR-SP.git code/CEFR-SP
git clone https://github.com/brucewlee/lftk.git code/lftk
git clone https://github.com/nishkalavallabhi/UniversalCEFRScoring.git code/UniversalCEFRScoring
```
