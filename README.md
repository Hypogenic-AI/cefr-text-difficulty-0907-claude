# Modeling Text Difficulty Using CEFR Levels

An interpretable feature analysis of sentence-level CEFR difficulty, comparing handcrafted linguistic features against fine-tuned BERT.

## Key Results

| Model | Macro F1 | Accuracy |
|-------|----------|----------|
| Majority baseline | 0.083 | — |
| Logistic Regression (41 features) | 0.390 | 0.449 |
| Random Forest (41 features) | 0.436 | 0.534 |
| XGBoost (41 features) | 0.435 | 0.539 |
| **BERT (fine-tuned)** | **0.524** | **0.644** |

- 60% of BERT's predictions explained by interpretable features (probing R² = 0.601)
- GPT-2 surprisal provides the largest unique feature contribution (F1 drop = 0.023 when removed)
- BERT's advantage concentrates on boundary levels (A2, C1, C2)

See [REPORT.md](REPORT.md) for the full analysis.

## Project Structure

```
├── REPORT.md                 # Full research report with results
├── README.md                 # This file
├── planning.md               # Research plan and hypothesis decomposition
├── pyproject.toml             # Python dependencies
├── datasets/
│   └── cefr_sp_en/
│       └── train.csv          # CEFR-SP English dataset (10,004 sentences)
├── src/
│   ├── config.py              # Configuration constants
│   ├── data_loader.py         # Data loading and CV splits
│   ├── feature_extraction.py  # 41-feature extraction pipeline
│   └── run_experiments.py     # Main experiment runner
├── results/
│   ├── experiment_results.json # All results in structured JSON
│   ├── features.csv           # Cached extracted features
│   ├── feature_groups.json    # Feature group definitions
│   ├── feature_correlations.csv # Spearman correlations with CEFR
│   └── feature_importance.csv # XGBoost feature importance
└── figures/
    ├── class_distribution.png
    ├── feature_correlations_top20.png
    ├── top_features_by_level.png
    ├── xgboost_feature_importance.png
    ├── model_comparison.png
    ├── ablation_individual_groups.png
    ├── per_class_f1_comparison.png
    └── confusion_matrices_comparison.png
```

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn scipy xgboost textstat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate
pip install spacy
python -m spacy download en_core_web_sm
```

## Running Experiments

```bash
# Step 1: Extract features (cached after first run)
cd src && python feature_extraction.py

# Step 2: Run all experiments (EDA, classifiers, BERT, diagnostics)
cd src && CUDA_VISIBLE_DEVICES=0 python run_experiments.py
```

Total runtime: ~18 minutes on NVIDIA RTX A6000.

## Feature Groups

| Group | Features | Description |
|-------|----------|-------------|
| Readability (7) | ARI, Flesch-Kincaid, etc. | Traditional readability formulas |
| Lexical (11) | TTR, word length, syllables | Vocabulary complexity measures |
| Syntactic (16) | Tree depth, POS ratios | Dependency parse complexity |
| Surprisal (7) | GPT-2 per-token surprisal | Neural language model predictability |

## Dataset

[CEFR-SP](https://github.com/Yusuke196/cefr-sp) (Arase et al., 2022): 10,004 English sentences annotated with CEFR levels A1–C2.

## Citation

If you use this work, please cite:

```
Arase, Y., Uchida, S., & Kajiwara, T. (2022). CEFR-Based Sentence-Profile
Dataset for English Learners. LREC 2022.
```
