"""
Configuration and constants for the training pipeline.
"""

import os
from pathlib import Path

# ============================================================
# Directories
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Input/Output File Naming
# ============================================================

# feature_engineering/pipeline.py outputs this file
FEATURE_ENGINEERED_DATA = DATA_DIR / "features_engineered.jsonl"

# Exported model and metadata
MODEL_FILEPATH = MODELS_DIR / "model_production.joblib"
MODEL_METADATA_FILEPATH = MODELS_DIR / "model_metadata.json"

# ============================================================
# Model Selection Metrics
# ============================================================

# Primary metric for model selection (higher is better)
PRIMARY_METRIC = "roc_auc"

# All metrics to evaluate and store
EVAL_METRICS = [
    "roc_auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
]

# Metrics where higher is better
HIGHER_IS_BETTER = {
    "roc_auc": True,
    "accuracy": True,
    "precision": True,
    "recall": True,
    "f1": True,
}

# ============================================================
# Train-Test Split
# ============================================================
TEST_SIZE = 0.2
RANDOM_STATE = 42
STRATIFY = True

# ============================================================
# Model Hyperparameters
# ============================================================

# Ensemble voting classifier configuration
ENSEMBLE_CONFIG = {
    "voting": "soft",
    "estimators": [
        {
            "name": "decision_tree",
            "class": "sklearn.tree.DecisionTreeClassifier",
            "params": {"random_state": 42},
        },
        {
            "name": "logistic_regression",
            "class": "sklearn.linear_model.LogisticRegression",
            "params": {"max_iter": 1000},
        },
        {
            "name": "random_forest",
            "class": "sklearn.ensemble.RandomForestClassifier",
            "params": {"n_estimators": 200, "random_state": 42},
        },
        {
            "name": "svm",
            "class": "sklearn.svm.SVC",
            "params": {"kernel": "linear", "probability": True, "random_state": 42},
        },
    ],
}
