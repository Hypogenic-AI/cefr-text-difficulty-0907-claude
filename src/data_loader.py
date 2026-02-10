"""Load and prepare the CEFR-SP dataset."""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from config import DATA_PATH, CEFR_TO_INT, SEED, N_FOLDS


def load_data():
    """Load CEFR-SP dataset and return DataFrame with numeric labels."""
    df = pd.read_csv(DATA_PATH)
    df = df[["text", "cefr_level"]].dropna().reset_index(drop=True)
    df["label"] = df["cefr_level"].map(CEFR_TO_INT)
    print(f"Loaded {len(df)} sentences")
    print(f"Label distribution:\n{df['cefr_level'].value_counts().sort_index()}")
    return df


def get_cv_splits(df):
    """Return list of (train_idx, test_idx) for stratified k-fold CV."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(skf.split(df, df["label"]))


if __name__ == "__main__":
    df = load_data()
    splits = get_cv_splits(df)
    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"Fold {i}: train={len(train_idx)}, test={len(test_idx)}")
