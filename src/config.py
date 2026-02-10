"""Configuration for the CEFR text difficulty experiments."""
import os

SEED = 42
N_FOLDS = 5
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "cefr_sp_en", "train.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
CEFR_TO_INT = {level: i for i, level in enumerate(CEFR_LEVELS)}
INT_TO_CEFR = {i: level for i, level in enumerate(CEFR_LEVELS)}

# BERT training config
BERT_MODEL_NAME = "bert-base-uncased"
BERT_MAX_LEN = 128
BERT_BATCH_SIZE = 64
BERT_LR = 2e-5
BERT_EPOCHS = 5
BERT_WARMUP_RATIO = 0.1

# GPU
DEVICE = "cuda:0"
