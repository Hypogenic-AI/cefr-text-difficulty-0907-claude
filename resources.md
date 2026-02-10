# Research Resources: Modeling Text Difficulty Using CEFR Levels

> Catalog of all research resources (papers, datasets, code, tools) collected for this project.
> Last updated: 2026-02-10

---

## Papers (18 downloaded)

All papers are stored in the `papers/` directory. Chunked page-level PDFs are available in `papers/pages/`.

### 1. arase2022_cefr_sp.pdf

- **Citation:** Arase, Y., Uchida, S., & Kajiwara, T. (2022). CEFR-Based Sentence Difficulty Annotation and Assessment. In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP 2022)*.
- **Key Contribution:** Introduces the CEFR-SP corpus of 17,676 English sentences annotated with CEFR levels by education professionals, and proposes a BERT-based sentence-level difficulty assessment model achieving 84.5% macro-F1.
- **Relevance to Hypothesis:** Provides the primary English sentence-level CEFR dataset and a strong neural baseline. Demonstrates that fine-tuned BERT can reliably predict CEFR difficulty at the sentence level, supporting the viability of transformer-based approaches.

### 2. xia2016_readability_l2.pdf

- **Citation:** Xia, M., Kochmar, E., & Briscoe, T. (2016). Text Readability Assessment for Second Language Learners. In *Proceedings of the 11th Workshop on Innovative Use of NLP for Building Educational Applications (BEA@NAACL 2016)*.
- **Key Contribution:** Develops readability models specifically for L2 English learners using Cambridge exam texts, demonstrating that domain adaptation from L1 readability models to L2 contexts significantly improves performance.
- **Relevance to Hypothesis:** Highlights the distinction between L1 and L2 text difficulty and the importance of using CEFR-aligned learner data rather than native-speaker readability corpora. Validates that linguistic features transfer across readability frameworks.

### 3. vajjala2018_universal_cefr.pdf

- **Citation:** Vajjala, S., & Rama, T. (2018). Experiments with Universal CEFR Classification. In *Proceedings of the 13th Workshop on Innovative Use of NLP for Building Educational Applications (BEA@NAACL 2018)*.
- **Key Contribution:** Demonstrates cross-lingual CEFR classification across German, Czech, and Italian using the MERLIN corpus with Universal Dependencies features, showing that language-independent morphosyntactic features enable cross-lingual transfer.
- **Relevance to Hypothesis:** Provides evidence that CEFR levels are to some degree language-universal and that UDPipe-extracted features can generalize across languages, supporting the design of multilingual difficulty models.

### 4. khallaf2021_arabic_cefr.pdf

- **Citation:** Khallaf, N., & Sharoff, S. (2021). Automatic Difficulty Classification of Arabic Text. In *Proceedings of the Sixth Arabic Natural Language Processing Workshop (WANLP@EACL 2021)*.
- **Key Contribution:** Compares handcrafted linguistic features against BERT representations for Arabic CEFR difficulty classification, finding that BERT achieves competitive performance even without language-specific feature engineering.
- **Relevance to Hypothesis:** Demonstrates CEFR modeling in a morphologically rich, non-Latin-script language. Supports the claim that pretrained language models can capture difficulty-related information without explicit linguistic feature engineering.

### 5. lagutina2023_cefr_ml_bert.pdf

- **Citation:** Lagutina, N., Lagutina, K., Boychuk, E., & Vorontsov, I. (2023). Stylometric Features vs BERT for CEFR Level Classification. In *Proceedings of the 2023 Conference on Artificial Intelligence and Natural Language (AINL 2023)*.
- **Key Contribution:** Systematically compares stylometric (handcrafted) features against BERT embeddings for CEFR classification, analyzing which feature families contribute most to difficulty prediction.
- **Relevance to Hypothesis:** Directly addresses the core question of whether traditional linguistic features or neural representations are more effective for CEFR classification, providing empirical evidence for hybrid approaches.

### 6. arnold2018_cefr_metrics.pdf

- **Citation:** Arnold, T., Ballier, N., Gaillat, T., & Lissargue, P. (2018). A Corpus-Based Study of Linguistic Complexity Metrics in EFCAMDAT. In *Proceedings of the Workshop on Linguistic Complexity and Natural Language Processing*.
- **Key Contribution:** Applies a battery of linguistic complexity metrics (lexical richness, syntactic complexity, discourse coherence) to the EFCAMDAT corpus of 41,000+ learner texts, mapping metric behavior across CEFR levels.
- **Relevance to Hypothesis:** Provides empirical grounding for which complexity metrics actually discriminate between CEFR levels in real learner data, informing feature selection for difficulty models.

### 7. imperial2025_universalcefr.pdf

- **Citation:** Imperial, J. M., Barayan, A., Stodden, R., Wilkens, R., et al. (2025). UniversalCEFR: Enabling Open Multilingual Research on Language Proficiency Assessment. In *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)*.
- **Key Contribution:** Consolidates 26 existing corpora into UniversalCEFR, the largest open multilingual CEFR dataset with 505,807 texts across 13 languages and 4 scripts. Provides benchmark experiments with linguistic features, fine-tuned LLMs, and descriptor-based prompting.
- **Relevance to Hypothesis:** Serves as the primary large-scale benchmark dataset. Benchmark results establish strong baselines for both feature-based and neural approaches across multiple languages, directly informing experimental design.

### 8. pilan2016_readable_read.pdf

- **Citation:** Pilan, I., Vajjala, S., & Volodina, E. (2016). A Readable Read: Automatic Assessment of Language Learning Materials Based on Linguistic Complexity. In *International Journal of Computational Linguistics and Applications, 7(1)*.
- **Key Contribution:** Develops CEFR-level classification for Swedish language learning texts using 867 COCTAILL corpus texts, achieving 73.4% accuracy with a rich set of linguistic complexity features including dependency-based measures.
- **Relevance to Hypothesis:** Demonstrates that fine-grained linguistic complexity features (especially syntactic dependency measures) are predictive of CEFR levels for non-English languages, supporting the generalizability of the feature-based approach.

### 9. santucci2020_text_complexity.pdf

- **Citation:** Santucci, V., Ferretti, S., Santarelli, F., & Spina, S. (2020). Automatic Classification of Text Complexity for Italian Learners. In *Proceedings of the Seventh Italian Conference on Computational Linguistics (CLiC-it 2020)*.
- **Key Contribution:** Applies text complexity analysis to Italian CEFR-labeled texts, developing classifiers that combine readability indices with morphosyntactic features specific to Italian.
- **Relevance to Hypothesis:** Extends CEFR difficulty modeling to Italian, showing that language-specific morphosyntactic features complement universal readability metrics and improve classification accuracy.

### 10. naous2023_readme_plus.pdf

- **Citation:** Naous, T., Ryan, M. J., Lavrouk, A., Chandra, M., & Xu, W. (2023). ReadMe++: Benchmarking Multilingual Language Models for Multi-Domain Readability Assessment. In *Findings of the Association for Computational Linguistics: EACL 2024*.
- **Key Contribution:** Introduces ReadMe++, a multilingual multi-domain readability benchmark of 9,757 sentences across 5 languages (Arabic, English, French, Hindi, Russian) with CEFR-aligned annotations and a rank-and-rate annotation methodology.
- **Relevance to Hypothesis:** Provides a carefully annotated multilingual readability benchmark with domain diversity. The cross-lingual and cross-domain evaluation protocols are directly applicable to testing the generalization of CEFR difficulty models.

### 11. fujinuma2021_joint_readability.pdf

- **Citation:** Fujinuma, Y., & Hagiwara, M. (2021). Graph-Based Joint Word-Document Readability Assessment. In *Findings of the Association for Computational Linguistics: EMNLP 2021*.
- **Key Contribution:** Proposes a Graph Convolutional Network (GCN) architecture for jointly estimating word-level and document-level readability, treating the problem as label propagation over a word-document bipartite graph.
- **Relevance to Hypothesis:** Introduces a novel architectural approach that connects word-level difficulty to document-level CEFR classification through graph-based reasoning, offering an alternative to purely sequential neural models.

### 12. malik2024_tarzan_tolkien.pdf

- **Citation:** Malik, A., Mayhew, S., Piech, C., & Bicknell, K. (2024). From Tarzan to Tolkien: Controlling the Language Proficiency Level of LLMs for Content Generation. In *Findings of the Association for Computational Linguistics: ACL 2024*.
- **Key Contribution:** Develops CaLM (CEFR-Aligned Language Model) using PPO fine-tuning of LLaMA2-7B and Mistral-7B to generate text at specified CEFR levels, demonstrating that LLMs can be trained to control output difficulty.
- **Relevance to Hypothesis:** While focused on generation rather than classification, this work validates that CEFR levels are learnable by large language models and that difficulty can be disentangled as a controllable attribute, informing both the modeling and evaluation aspects of difficulty prediction.

### 13. francois2009_ffl_readability.pdf

- **Citation:** Francois, T., & Fairon, C. (2012). An AI Readability Formula for French as a Foreign Language. In *Proceedings of the Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning (EMNLP-CoNLL 2012)*.
- **Key Contribution:** Develops the first readability formula specifically for French as a Foreign Language (FFL), incorporating statistical language model perplexity alongside traditional readability features to predict CEFR levels.
- **Relevance to Hypothesis:** Demonstrates that language model perplexity is a powerful predictor of CEFR difficulty, providing early evidence for using neural LM-derived features in difficulty classification. The FFL-specific approach highlights the importance of learner-oriented models.

### 14. jamet2024_french_difficulty.pdf

- **Citation:** Jamet, L., Bibal, A., Francois, T., & Lavergne, T. (2024). Assessing French Difficulty with LLMs. In *Proceedings of the 2024 Conference on Language Resources and Evaluation (LREC-COLING 2024)*.
- **Key Contribution:** Evaluates the zero-shot and few-shot ability of modern LLMs (GPT-3.5, GPT-4, LLaMA) to assess French text difficulty on the CEFR scale, comparing against established feature-based models.
- **Relevance to Hypothesis:** Provides direct evidence on the effectiveness of prompting-based LLM approaches for CEFR classification, informing whether zero-shot LLM predictions can complement or replace trained classifiers.

### 15. kogan2025_ace_cefr.pdf

- **Citation:** Kogan, V., Efimov, P., & Glazkova, A. (2025). Ace-CEFR: Assessing CEFR Level of Conversational English. Preprint / Conference paper, 2025.
- **Key Contribution:** Extends CEFR difficulty assessment to conversational/spoken English text, addressing the gap between written text readability and spoken interaction difficulty.
- **Relevance to Hypothesis:** Broadens the scope of CEFR difficulty modeling beyond written texts to conversational data, testing whether the same feature frameworks and models generalize to spoken language transcripts.

### 16. imperial2023_flesch_fumble.pdf

- **Citation:** Imperial, J. M., Madanagopal, H., & Ong, E. (2023). Flesch or Fumble? Evaluating Readability Standard Alignment of Instruction-Tuned Language Models. In *Proceedings of the Workshop on Text Simplification, Accessibility, and Readability (TSAR@RANLP 2023)*.
- **Key Contribution:** Evaluates how well instruction-tuned LLMs (ChatGPT, Alpaca, Vicuna) align with established readability standards, finding significant inconsistencies in LLM self-assessment of text difficulty.
- **Relevance to Hypothesis:** Provides cautionary evidence about using LLMs as readability judges. Highlights the need for calibrated evaluation when using LLM-based approaches for CEFR classification and the gap between LLM confidence and actual readability alignment.

### 17. vajjala2012_sla_readability.pdf

- **Citation:** Vajjala, S., & Meurers, D. (2012). On Improving the Accuracy of Readability Classification using Insights from Second Language Acquisition. In *Proceedings of the Seventh Workshop on Building Educational Applications Using NLP (BEA@NAACL 2012)*.
- **Key Contribution:** Integrates Second Language Acquisition (SLA) research into readability feature design, incorporating psycholinguistic word properties (age of acquisition, imageability, familiarity) alongside traditional readability measures.
- **Relevance to Hypothesis:** Foundational work showing that SLA-informed features significantly improve readability classification over traditional formulas. Motivates the inclusion of psycholinguistic and learner-oriented features in CEFR difficulty models.

### 18. pitler2008_readability.pdf

- **Citation:** Pitler, E., & Nenkova, A. (2008). Revisiting Readability: A Unified Framework for Predicting Text Quality. In *Proceedings of the 2008 Conference on Empirical Methods in Natural Language Processing (EMNLP 2008)*.
- **Key Contribution:** Proposes a unified readability framework combining discourse relations, lexical features (language model scores), and syntactic features, showing that discourse-level features are significant predictors of readability.
- **Relevance to Hypothesis:** Establishes the importance of discourse-level features (RST relations, entity coherence) for readability prediction. Provides motivation for including discourse-based features alongside lexical and syntactic measures in CEFR models.

---

## Datasets

### Primary CEFR-Labeled Datasets

| # | Name | Size | Languages | CEFR Labels | Access Method | URL |
|---|------|------|-----------|-------------|---------------|-----|
| 1 | **UniversalCEFR** | 505,807 texts | 13 languages (English, German, Italian, Czech, French, Spanish, Portuguese, Swedish, Estonian, Slovene, Arabic, Hindi, Korean) | Yes (A1-C2) | HuggingFace Hub | [HuggingFace: UniversalCEFR/datasets](https://huggingface.co/UniversalCEFR) |
| 2 | **CEFR-SP** | 17,676 sentences | English | Yes (A1-C2) | GitHub + HuggingFace | [GitHub: yukiar/CEFR-SP](https://github.com/yukiar/CEFR-SP) / [HuggingFace: UniversalCEFR/cefr_sp_en](https://huggingface.co/datasets/UniversalCEFR/cefr_sp_en) |
| 3 | **MERLIN Corpus** | 2,286 texts | German, Italian, Czech | Yes (A1-C1) | Web download | [merlin-platform.eu](https://merlin-platform.eu) |
| 4 | **Cambridge English Exams** | 331 documents | English (L2 learner texts) | Yes (by exam level: KET/PET/FCE/CAE/CPE) | Web download | [cl.cam.ac.uk/~mx223/cedata.html](https://www.cl.cam.ac.uk/~mx223/cedata.html) |
| 5 | **EFCAMDAT** | 41,000+ texts | English (learner texts) | Yes (mapped to CEFR via course level) | Access form required | [efcamdat.info](https://corpus.mml.cam.ac.uk/efcamdat2/) |
| 6 | **COCTAILL** | 867 texts | Swedish | Yes (A1-C1) | Swedish Language Bank (Sprakbanken) | [Sprakbanken](https://spraakbanken.gu.se/en/resources/coctaill) |
| 7 | **ReadMe++** | 9,757 sentences | Arabic, English, French, Hindi, Russian | Yes (1-6 CEFR-aligned scale) | GitHub | [GitHub: tareknaous/readme](https://github.com/tareknaous/readme) |
| 8 | **Kaggle CEFR English Texts** | ~1,500 texts | English | Yes (A1-C2) | Kaggle download (CC0) | [Kaggle: cefr-levelled-english-texts](https://www.kaggle.com/datasets/amontgomerie/cefr-levelled-english-texts) |
| 9 | **BEA-2019 W&I+LOCNESS** | 3,350 texts | English | Yes (A, B, C levels) | HuggingFace Hub | [HuggingFace: bea2019st/wi_locness](https://huggingface.co/datasets/bea2019st/wi_locness) |
| 10 | **English CEFR Word Dataset** | 18,995 words | English | Yes (word-level A1-C2) | HuggingFace Hub (Apache 2.0) | [HuggingFace: Alex123321/english_cefr_dataset](https://huggingface.co/datasets/Alex123321/english_cefr_dataset) |
| 11 | **CEFR Mixed 60K** | 60,000 token-level annotations | English | Yes (token-level CEFR) | HuggingFace Hub | [HuggingFace: DioBot2000/CEFR_MIXED_dataset_60000](https://huggingface.co/datasets/DioBot2000/CEFR_MIXED_dataset_60000) |
| 12 | **Spanish Readability CAES** | 31,149 texts | Spanish | Yes (CEFR-aligned) | HuggingFace Hub (CC-BY-4.0) | [HuggingFace: somosnlp-hackathon-2022/readability-es-caes](https://huggingface.co/datasets/somosnlp-hackathon-2022/readability-es-caes) |

### Readability Datasets (Non-CEFR)

| # | Name | Size | Languages | Labels | Access Method | URL |
|---|------|------|-----------|--------|---------------|-----|
| 13 | **CommonLit Readability** | ~3,000 excerpts | English | Continuous readability scores (not CEFR) | HuggingFace Hub | [HuggingFace: casey-martin/CommonLit-Ease-of-Readability](https://huggingface.co/datasets/casey-martin/CommonLit-Ease-of-Readability) |
| 14 | **OneStopEnglishCorpus** | 189 articles x 3 levels | English | 3-level readability (Elementary / Intermediate / Advanced) | GitHub | [GitHub: nishkalavallabhi/OneStopEnglishCorpus](https://github.com/nishkalavallabhi/OneStopEnglishCorpus) |

### Word-Level CEFR Resources

| # | Name | Description | URL |
|---|------|-------------|-----|
| 15 | **Words-CEFR-Dataset** | English words mapped to CEFR levels based on CEFR-J, with lemmas, POS tags, and Google N-gram frequency data | [GitHub: Maximax67/Words-CEFR-Dataset](https://github.com/Maximax67/Words-CEFR-Dataset) |
| 16 | **English Profile Vocabulary** | Official CEFR word lists from the English Profile project | [englishprofile.org](https://www.englishprofile.org/wordlists/evp) |

---

## Code Repositories

### Primary Tools and Libraries

#### 1. LFTK (Linguistic Feature Toolkit)
- **URL:** [github.com/brucewlee/lftk](https://github.com/brucewlee/lftk)
- **Stars:** 149
- **Language:** Python
- **Framework:** spaCy
- **Paper:** "LFTK: Handcrafted Features in Computational Linguistics" (BEA@ACL 2023)
- **Key Features:** 200+ handcrafted linguistic features; readability formulas (Flesch-Kincaid, SMOG, Gunning Fog, Coleman-Liau, ARI); type-token ratio and lexical variation; age-of-acquisition features; SubtlexUS word frequency; POS counts and tree structure features; entity grid and local coherence; LDA-based semantic richness
- **Status:** Active successor to LingFeat
- **Relevance:** The most comprehensive feature extraction toolkit for CEFR difficulty modeling. Covers all major linguistic branches needed for handcrafted feature baselines.

#### 2. LingFeat
- **URL:** [github.com/brucewlee/lingfeat](https://github.com/brucewlee/lingfeat)
- **Stars:** 132
- **Language:** Python
- **Framework:** spaCy
- **Paper:** "Pushing on Text Readability Assessment: A Transformer Meets Handcrafted Linguistic Features" (EMNLP 2021)
- **Key Features:** 255 features across 5 linguistic branches (Advanced Semantic, Discourse, Syntactic, Lexico-Semantic, Shallow Traditional)
- **Status:** Archived (August 2025); superseded by LFTK
- **Relevance:** Historical predecessor with slightly broader feature set. Use LFTK for new projects.

#### 3. TextDescriptives
- **URL:** [github.com/HLasse/TextDescriptives](https://github.com/HLasse/TextDescriptives)
- **Stars:** 359
- **Language:** Python (86.3%)
- **Framework:** spaCy v3 pipeline component
- **Paper:** "TextDescriptives: A Python package for calculating a large variety of metrics from text" (2023)
- **Key Features:** Readability indices (Flesch, SMOG, Gunning Fog, Coleman-Liau, ARI, LIX, RIX); descriptive statistics (token/sentence length, syllable counts); dependency distance metrics; POS proportions; semantic coherence; information theory metrics; text quality assessment; multilingual via spaCy models; web demo available
- **Status:** Actively maintained (v2.8.4, December 2024)
- **Relevance:** Best modern spaCy-integrated tool for readability and linguistic metrics. Dependency distance is a strong syntactic complexity proxy. Multilingual support via spaCy models.

#### 4. CEFR-SP
- **URL:** [github.com/yukiar/CEFR-SP](https://github.com/yukiar/CEFR-SP)
- **Stars:** 56
- **Language:** Python (98.5%)
- **Key Features:** 17K CEFR-annotated English sentences; sentence-level CEFR assessment model; criterial feature analysis between adjacent CEFR levels; macro-F1 of 84.5%
- **Status:** Stable (2022-2024)
- **Relevance:** Provides both the primary English CEFR sentence dataset and a baseline assessment model. Criterial feature analysis reveals which features distinguish adjacent CEFR levels.

#### 5. ReadMe++ (readmepp)
- **URL:** [github.com/tareknaous/readme](https://github.com/tareknaous/readme)
- **Stars:** 12
- **Language:** Python
- **Framework:** HuggingFace Transformers, PyTorch
- **Key Features:** 9,757 CEFR-annotated sentences across 5 languages; pip-installable (`pip install readmepp`); fine-tuned BERT models on HuggingFace; rank-and-rate annotation methodology; cross-lingual transfer evaluation
- **Status:** Stable (November 2023)
- **Relevance:** Ready-to-use CEFR readability prediction via pip install. Multilingual and multi-domain coverage for generalization studies.

#### 6. UniversalCEFR
- **URL:** [github.com/UniversalCEFR](https://github.com/UniversalCEFR) / [universalcefr.github.io](https://universalcefr.github.io/)
- **HuggingFace:** [huggingface.co/UniversalCEFR](https://huggingface.co/UniversalCEFR)
- **Key Features:** Benchmark code for linguistic features, fine-tuned LLMs, and descriptor-based prompting experiments; standardized JSON format with 8 metadata fields per instance; 505,807 texts across 13 languages
- **Status:** Published at EMNLP 2025
- **Relevance:** The definitive multilingual CEFR benchmark. Benchmark code provides reproducible baselines.

#### 7. surprisal
- **URL:** [github.com/aalok-sathe/surprisal](https://github.com/aalok-sathe/surprisal)
- **Language:** Python
- **Framework:** HuggingFace Transformers, KenLM
- **Key Features:** Unified API for surprisal computation from causal LMs (GPT-2, LLaMA), Petals distributed models, and KenLM n-gram models; pip installable with optional extras
- **Status:** Actively maintained
- **Relevance:** Enables computing language model surprisal as a text difficulty feature. Surprisal correlates with processing difficulty and serves as a powerful feature for CEFR classification.

#### 8. textstat
- **URL:** [github.com/textstat/textstat](https://github.com/textstat/textstat)
- **Language:** Python (standalone, no NLP dependency)
- **Key Features:** Flesch Reading Ease; Flesch-Kincaid Grade; SMOG Index; Coleman-Liau Index; Automated Readability Index; Dale-Chall Readability Score; Gunning Fog Index; Linsear Write Formula; `text_standard` consensus grade level; multilingual syllable support
- **Status:** Actively maintained
- **Relevance:** Lightweight, dependency-free baseline readability computation. Good for quick feature extraction without spaCy overhead.

### CEFR Classification Projects

#### 9. CEFR-English-Level-Predictor
- **URL:** [github.com/AMontgomerie/CEFR-English-Level-Predictor](https://github.com/AMontgomerie/CEFR-English-Level-Predictor)
- **Language:** Python
- **Framework:** scikit-learn, HuggingFace Transformers, Streamlit
- **Key Features:** Full CEFR classification pipeline; XGBoost vs SVC vs Random Forest vs Decision Tree vs fine-tuned BERT/DeBERTa comparison; XGBoost achieves ~71% accuracy, outperforming transformers on 1,500 example texts; Streamlit web app for inference; Docker deployment support
- **Relevance:** Complete reference implementation. The finding that XGBoost outperforms transformers on small data is informative for modeling decisions on limited datasets.

#### 10. UniversalCEFRScoring
- **URL:** [github.com/nishkalavallabhi/UniversalCEFRScoring](https://github.com/nishkalavallabhi/UniversalCEFRScoring)
- **Language:** Python
- **Key Features:** Cross-lingual CEFR classification experiments for German, Italian, Czech using UDPipe-extracted features
- **Relevance:** Reference implementation for the Vajjala & Rama (2018) cross-lingual CEFR experiments. Demonstrates feature extraction with Universal Dependencies.

#### 11. textcomplexity
- **URL:** [github.com/tsproisl/textcomplexity](https://github.com/tsproisl/textcomplexity)
- **Language:** Python
- **Key Features:** Lexical variability, evenness, and rarity measures; syntactic complexity measures; linguistic and stylistic complexity quantification
- **Relevance:** Provides complementary complexity measures not available in LFTK or TextDescriptives, particularly lexical rarity metrics.

#### 12. cefrpy
- **URL:** [github.com/Maximax67/cefrpy](https://github.com/Maximax67/cefrpy)
- **Language:** Python
- **Key Features:** Lightweight word-level CEFR analysis; spaCy integration; maps individual words to CEFR levels
- **Relevance:** Useful for computing word-level CEFR statistics (percentage of C1+ words, average word CEFR level) as input features for text-level classifiers.

#### 13. diff_joint_estimate
- **URL:** [github.com/akkikiki/diff_joint_estimate](https://github.com/akkikiki/diff_joint_estimate)
- **Language:** Python
- **Key Features:** GCN-based joint word-document readability estimation; implementation of the Fujinuma & Hagiwara (2021) graph-based approach
- **Relevance:** Reference implementation for the novel graph-based joint estimation approach that links word difficulty to document difficulty.

### Additional Related Repositories

| Name | URL | Description |
|------|-----|-------------|
| **OneStopEnglishCorpus** | [github.com/nishkalavallabhi/OneStopEnglishCorpus](https://github.com/nishkalavallabhi/OneStopEnglishCorpus) | Three-level readability corpus with sentence alignments |
| **LingX** | [github.com/ContentSide/LingX](https://github.com/ContentSide/LingX) | Psycholinguistic complexity metrics (IDT, DLT) using Stanza |
| **neural-complexity** | [github.com/vansky/neural-complexity](https://github.com/vansky/neural-complexity) | Neural LM for incremental processing complexity (surprisal, entropy) |
| **Linguistic-Features-for-Readability** | [github.com/TovlyDeutsch/Linguistic-Features-for-Readability](https://github.com/TovlyDeutsch/Linguistic-Features-for-Readability) | Feature code from Deutsch, Jasbi & Shieber (2020) |
| **sent_cefr** | [github.com/IldikoPilan/sent_cefr](https://github.com/IldikoPilan/sent_cefr) | Small corpus of sentences with CEFR-level annotations |
| **TRUNAJOD 2.0** | [github.com/dpalmasan/TRUNAJOD2.0](https://github.com/dpalmasan/TRUNAJOD2.0) | Text complexity library (spaCy); parse tree similarity, semantic coherence, emotion lexicon; primarily Spanish |

---

## Feature Extraction Tools

The following summarizes the main tools available for extracting linguistic features relevant to CEFR difficulty modeling, organized by feature category.

### Readability Formulas

Traditional readability indices provide quick baseline features but are designed for L1 English and may not directly reflect L2 difficulty.

| Tool | Features | Notes |
|------|----------|-------|
| **textstat** | Flesch Reading Ease, Flesch-Kincaid Grade, SMOG, Gunning Fog, Coleman-Liau, ARI, Dale-Chall, Linsear Write | Standalone, no NLP pipeline needed |
| **TextDescriptives** | Flesch, SMOG, Gunning Fog, Coleman-Liau, ARI, LIX, RIX | spaCy v3 integration, includes European indices (LIX, RIX) |
| **LFTK** | Flesch-Kincaid, SMOG, Gunning Fog, Coleman-Liau, ARI | Part of the broader 200+ feature set |

### Lexical Features

Word-level features capture vocabulary difficulty, frequency, and diversity.

| Tool | Features | Notes |
|------|----------|-------|
| **LFTK** | Type-token ratio, word frequency (SubtlexUS), age-of-acquisition, lexical variation scores, difficult word counts | Most comprehensive lexical feature coverage |
| **TextDescriptives** | Token/sentence length statistics, syllable counts | Good for basic descriptive statistics |
| **cefrpy** | Per-word CEFR level mapping | Enables CEFR-specific vocabulary difficulty metrics |
| **textcomplexity** | Lexical variability, evenness, rarity | Specialized lexical complexity measures |

### Syntactic Features

Syntactic complexity features capture grammatical sophistication.

| Tool | Features | Notes |
|------|----------|-------|
| **LFTK** | POS counts, phrasal counts, tree structure features, parse tree depth | Constituency and dependency features |
| **TextDescriptives** | Dependency distance metrics | Strong proxy for syntactic complexity |
| **LingX** | Integration-cost Dependency Theory (IDT), Dependency Locality Theory (DLT) metrics | Psycholinguistically motivated syntactic complexity |
| **textcomplexity** | Syntactic complexity measures | Additional syntactic metrics |

### Discourse and Coherence Features

Discourse-level features capture text organization and coherence.

| Tool | Features | Notes |
|------|----------|-------|
| **LFTK / LingFeat** | Entity grid, local coherence, LDA-based semantic richness | Most comprehensive discourse feature set |
| **TextDescriptives** | Semantic coherence between sentences | Embedding-based inter-sentence coherence |
| **TRUNAJOD** | Parse tree similarity, semantic coherence, synonym overlap | Primarily designed for Spanish |

### Language Model Features

Neural language model-derived features capture statistical text properties.

| Tool | Features | Notes |
|------|----------|-------|
| **surprisal** | Per-word surprisal, sentence-level perplexity | Supports GPT-2, LLaMA, KenLM |
| **neural-complexity** | Surprisal, entropy, entropy reduction | Focused on incremental processing complexity |
| **HuggingFace Transformers** | Perplexity, token probabilities, embeddings | Direct access to any pretrained LM |

### Recommended Feature Extraction Pipeline

For a comprehensive CEFR difficulty modeling pipeline, the recommended combination is:

1. **LFTK** for the core set of 200+ handcrafted linguistic features (lexical, syntactic, discourse, readability)
2. **TextDescriptives** for spaCy-integrated dependency distance and coherence metrics
3. **cefrpy** for word-level CEFR statistics as additional features
4. **surprisal** for language model perplexity/surprisal features
5. **HuggingFace Transformers** for fine-tuning BERT/DeBERTa-based classifiers

---

## Download Scripts

### UniversalCEFR (Primary Large-Scale Dataset)

```python
from datasets import load_dataset

# Load the full UniversalCEFR collection
# Individual language subsets are available as configs
universal_cefr = load_dataset("UniversalCEFR/datasets")

# Example: load English subset only
english_data = load_dataset("UniversalCEFR/datasets", "en")

print(f"Number of examples: {len(english_data['train'])}")
print(f"Columns: {english_data['train'].column_names}")
print(f"Sample: {english_data['train'][0]}")
```

### CEFR-SP (English Sentences)

```python
from datasets import load_dataset

# Load CEFR-SP from HuggingFace
cefr_sp = load_dataset("UniversalCEFR/cefr_sp_en")

print(f"Train size: {len(cefr_sp['train'])}")
print(f"Columns: {cefr_sp['train'].column_names}")
print(f"Sample: {cefr_sp['train'][0]}")

# Alternatively, clone from GitHub for the original format:
# git clone https://github.com/yukiar/CEFR-SP.git
```

### ReadMe++ (Multilingual Readability)

```python
# Option 1: pip install for direct model inference
# pip install readmepp

# Option 2: Load the dataset
from datasets import load_dataset

readme_pp = load_dataset("tareknaous/readme")
print(f"Languages available: {readme_pp.keys()}")
```

### BEA-2019 W&I+LOCNESS

```python
from datasets import load_dataset

bea2019 = load_dataset("bea2019st/wi_locness")

print(f"Splits: {bea2019.keys()}")
print(f"Train size: {len(bea2019['train'])}")
print(f"Sample: {bea2019['train'][0]}")
```

### English CEFR Word Dataset

```python
from datasets import load_dataset

# Word-level CEFR annotations (Apache 2.0 license)
word_cefr = load_dataset("Alex123321/english_cefr_dataset")

print(f"Number of words: {len(word_cefr['train'])}")
print(f"Sample: {word_cefr['train'][0]}")
```

### CEFR Mixed 60K (Token-Level)

```python
from datasets import load_dataset

cefr_mixed = load_dataset("DioBot2000/CEFR_MIXED_dataset_60000")

print(f"Number of annotations: {len(cefr_mixed['train'])}")
print(f"Columns: {cefr_mixed['train'].column_names}")
```

### Spanish Readability CAES

```python
from datasets import load_dataset

# CC-BY-4.0 license
spanish_cefr = load_dataset("somosnlp-hackathon-2022/readability-es-caes")

print(f"Size: {len(spanish_cefr['train'])}")
print(f"Columns: {spanish_cefr['train'].column_names}")
```

### CommonLit Readability (Non-CEFR Baseline)

```python
from datasets import load_dataset

commonlit = load_dataset("casey-martin/CommonLit-Ease-of-Readability")

print(f"Size: {len(commonlit['train'])}")
print(f"Columns: {commonlit['train'].column_names}")
```

### Kaggle CEFR English Texts

```python
# Requires kaggle CLI: pip install kaggle
# Ensure ~/.kaggle/kaggle.json is configured

import subprocess
subprocess.run([
    "kaggle", "datasets", "download",
    "-d", "amontgomerie/cefr-levelled-english-texts",
    "-p", "./datasets/kaggle_cefr/"
], check=True)

# Unzip
import zipfile
with zipfile.ZipFile("./datasets/kaggle_cefr/cefr-levelled-english-texts.zip", "r") as z:
    z.extractall("./datasets/kaggle_cefr/")
```

### OneStopEnglishCorpus

```python
# Clone from GitHub
import subprocess
subprocess.run([
    "git", "clone",
    "https://github.com/nishkalavallabhi/OneStopEnglishCorpus.git",
    "./datasets/OneStopEnglishCorpus/"
], check=True)
```

### Batch Download All HuggingFace Datasets

```python
"""
Batch download script for all HuggingFace-hosted CEFR datasets.
Saves each dataset to the datasets/ directory in Arrow format.
"""
from datasets import load_dataset
import os

SAVE_DIR = "./datasets"
os.makedirs(SAVE_DIR, exist_ok=True)

HF_DATASETS = {
    "universal_cefr": "UniversalCEFR/datasets",
    "cefr_sp_en": "UniversalCEFR/cefr_sp_en",
    "bea2019_wi_locness": "bea2019st/wi_locness",
    "english_cefr_words": "Alex123321/english_cefr_dataset",
    "cefr_mixed_60k": "DioBot2000/CEFR_MIXED_dataset_60000",
    "spanish_readability_caes": "somosnlp-hackathon-2022/readability-es-caes",
    "commonlit_readability": "casey-martin/CommonLit-Ease-of-Readability",
}

for name, hf_path in HF_DATASETS.items():
    print(f"\nDownloading {name} from {hf_path}...")
    try:
        ds = load_dataset(hf_path)
        save_path = os.path.join(SAVE_DIR, name)
        ds.save_to_disk(save_path)
        print(f"  Saved to {save_path}")
        for split_name, split_data in ds.items():
            print(f"  {split_name}: {len(split_data)} examples")
    except Exception as e:
        print(f"  ERROR downloading {name}: {e}")

print("\nAll downloads complete.")
```

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Papers downloaded | 18 |
| CEFR-labeled datasets | 12 |
| Non-CEFR readability datasets | 2 |
| Word-level CEFR resources | 2 |
| Code repositories (primary) | 13 |
| Code repositories (additional) | 6 |
| Feature extraction tool categories | 5 (readability, lexical, syntactic, discourse, LM-based) |
