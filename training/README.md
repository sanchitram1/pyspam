# Training Module

Productionalized training pipeline for the spam detection model. This module orchestrates feature engineering and model training to produce a production-ready classifier.

## Overview

The training module is responsible for:

1. **Loading feature-engineered data** from JSONL files
2. **Preprocessing features** using scikit-learn pipelines (OneHotEncoding for categorical features, StandardScaling for numerical features)
3. **Training an ensemble model** combining four diverse classifiers
4. **Evaluating performance** using multiple metrics (accuracy, precision, recall, F1, ROC-AUC)
5. **Saving the model and metadata** for deployment

## Architecture

```
Raw Data (JSONL)
     ↓
[feature_engineering/pipeline.py] 
     ↓ (outputs: features_engineered.jsonl)
[training/train.py] 
     ↓ (outputs: model_production.joblib, model_metadata.json)
[Docker/Cloud Deployment]
```

## Directory Structure

```
training/
├── __init__.py              # Module initialization
├── config.py                # Centralized configuration
├── train.py                 # Model training script
├── pipeline.py              # Orchestration script for full workflow
├── ensemble_model.py        # Legacy model development notebook (for reference)
├── README.md                # This file
```

## Configuration (`training/config.py`)

All pipeline parameters are centralized in a single configuration file:

### Paths
- `PROJECT_ROOT`: Root directory of the project
- `DATA_DIR`: Directory containing input data (default: `data/`)
- `MODELS_DIR`: Directory for saving trained models (default: `models/`)
- `FEATURE_ENGINEERED_DATA`: Path to feature-engineered JSONL input
- `MODEL_FILEPATH`: Path to save trained model
- `MODEL_METADATA_FILEPATH`: Path to save model metadata

### Model Evaluation Metrics
- `PRIMARY_METRIC`: "roc_auc" (primary metric for model selection)
- `EVAL_METRICS`: List of metrics to compute and store
  - roc_auc
  - accuracy
  - precision
  - recall
  - f1

### Train-Test Split
- `TEST_SIZE`: 0.2 (20% test set)
- `RANDOM_STATE`: 42 (for reproducibility)
- `STRATIFY`: True (stratified by target class)

### Ensemble Configuration
The model uses a VotingClassifier with four base estimators:

1. **Decision Tree Classifier** (random_state=42)
2. **Logistic Regression** (max_iter=1000)
3. **Random Forest Classifier** (200 estimators, random_state=42)
4. **SVM with Linear Kernel** (probability=True, random_state=42)

Voting strategy: "soft" (probability-based voting)

## Core Components

### `train.py`

Main training script that implements the following functions:

#### `load_training_data(input_path: str | Path) -> tuple[pd.DataFrame, pd.Series]`
Loads feature-engineered JSONL data and prepares features (X) and target (y).
- Removes non-predictive columns: "pkg_name", "is_spam"
- Returns tuple of (X_features, y_target)

#### `build_pipeline(X_train: pd.DataFrame) -> Pipeline`
Constructs a scikit-learn Pipeline with preprocessing and model stages.
- **Preprocessor**: ColumnTransformer with OneHotEncoder for categorical features
- **Scaler**: StandardScaler for numerical normalization
- **Classifier**: VotingClassifier ensemble with 4 base models

#### `evaluate_model(y_true, y_pred, y_pred_proba=None) -> dict`
Computes evaluation metrics on test set.
- Returns dictionary with: accuracy, precision, recall, f1, (and roc_auc if proba provided)

#### `save_model_with_metadata(pipeline, metrics, output_dir, model_filepath=None, metadata_filepath=None) -> dict`
Saves trained model and metadata to disk.
- Saves model as joblib binary
- Saves metadata as JSON including: timestamp, metrics, model_type, test_size, random_state, primary_metric

#### `train_model(input_path: str | Path, output_dir: str | Path) -> tuple[Pipeline, dict]`
Main training pipeline orchestrator.
1. Loads training data
2. Performs stratified train-test split
3. Builds and trains ensemble pipeline
4. Evaluates on test set
5. Saves model and metadata

### `pipeline.py`

Orchestration script for the complete workflow:

#### `run_feature_engineering(input_path: str | Path, output_path: str | Path) -> bool`
Executes the feature engineering pipeline as a subprocess.
- Calls `python -m feature_engineering.pipeline`
- Returns True on success, False on failure

#### `run_training(input_path: str | Path, output_dir: str | Path) -> bool`
Executes the model training as a subprocess.
- Calls `python -m training.train`
- Returns True on success, False on failure

#### `main()`
Full end-to-end pipeline orchestration.
- Requires: `--raw-data` argument (path to raw data JSONL)
- Optional: `--features-output` and `--models-output` paths
- Exits with code 1 if any step fails

### `config.py`

Centralized configuration module. Contains all constants, paths, and hyperparameters needed by the training pipeline. Modify this file to:
- Change model hyperparameters
- Adjust train-test split ratio
- Modify evaluation metrics
- Update file paths

## Usage

### Option 1: Full Pipeline (Feature Engineering + Training)

Run feature engineering and training sequentially:

```bash
python -m training.pipeline \
  --raw-data data/raw_data.jsonl \
  --features-output data/features_engineered.jsonl \
  --models-output models/
```

With default paths:
```bash
python -m training.pipeline --raw-data data/raw_data.jsonl
```

### Option 2: Training Only (if features already exist)

If you've already run feature engineering:

```bash
python -m training.train \
  --input data/features_engineered.jsonl \
  --output models/
```

With default paths:
```bash
python -m training.train
```

### Option 3: Python API

Import and use the training functions directly:

```python
from training.train import train_model

# Train with custom paths
pipeline, metrics = train_model(
    input_path="data/features_engineered.jsonl",
    output_dir="models/"
)

# Access trained model
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)
```

Or use the full pipeline:

```python
from training.pipeline import run_feature_engineering, run_training

# Feature engineering
if run_feature_engineering("data/raw.jsonl", "data/features.jsonl"):
    # Model training
    if run_training("data/features.jsonl", "models/"):
        print("Pipeline completed successfully!")
```

## Output Files

The training pipeline produces two files:

### `model_production.joblib`

A serialized scikit-learn Pipeline containing:
- **Preprocessor**: ColumnTransformer with OneHotEncoder
- **Scaler**: StandardScaler
- **Classifier**: VotingClassifier with 4 base models

**Usage in production:**
```python
import joblib

model = joblib.load("models/model_production.joblib")

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)
```

### `model_metadata.json`

Metadata about the trained model:
```json
{
  "timestamp": "2025-12-15T10:30:00.123456",
  "metrics": {
    "accuracy": 0.95,
    "precision": 0.92,
    "recall": 0.88,
    "f1": 0.90,
    "roc_auc": 0.96
  },
  "model_type": "VotingClassifier",
  "test_size": 0.2,
  "random_state": 42,
  "primary_metric": "roc_auc"
}
```

## Model Architecture

### Ensemble Strategy

The VotingClassifier uses **soft voting** where predictions are made based on the averaged probability estimates of the four base models:

1. **Decision Tree**: Fast, interpretable, captures non-linear patterns
2. **Logistic Regression**: Fast, stable, provides probability calibration
3. **Random Forest**: Robust, handles feature interactions, reduces overfitting
4. **SVM (Linear)**: Effective in high-dimensional spaces, good generalization

The combination balances:
- Speed (Decision Tree, Logistic Regression)
- Robustness (Random Forest)
- Generalization (SVM)

### Data Preprocessing

1. **Categorical Features**: OneHotEncoder (drop='first' to avoid multicollinearity)
   - handle_unknown='ignore' for new categories at test time
   - sparse_output=False for compatibility

2. **Numerical Features**: StandardScaler
   - Zero mean, unit variance
   - Applied after one-hot encoding to all features

## Training Details

### Data Split
- **Train set**: 80% (stratified by target)
- **Test set**: 20% (stratified by target)
- **Stratification**: Ensures class balance in both splits

### Model Evaluation
- **Train-test split**: Stratified 80/20 split with random_state=42
- **Metrics computed on test set**:
  - **Accuracy**: Overall correct predictions
  - **Precision**: True positives / (True positives + False positives)
  - **Recall**: True positives / (True positives + False negatives)
  - **F1**: Harmonic mean of precision and recall
  - **ROC-AUC**: Area under the ROC curve (primary metric)

## Integration with Docker

The Dockerfile includes the trained model artifacts:

```dockerfile
# Copy trained model
COPY models/ ./models/

# In the API:
import joblib
model = joblib.load("models/model_production.joblib")
```

The API can then use the model to make predictions on new packages.

## Testing

Comprehensive test suite achieves 100% code coverage:

```bash
# Run training module tests
python -m pytest tests/test_training_*.py -v --cov=training

# Run with coverage report
python -m pytest tests/test_training_*.py --cov=training --cov-report=html
```

Test files:
- `tests/test_training_config.py` (27 tests)
- `tests/test_training_train.py` (28 tests)
- `tests/test_training_pipeline.py` (22 tests)

Tests cover:
- Configuration loading and validation
- Data loading and preprocessing
- Pipeline construction
- Model training and evaluation
- File I/O (saving/loading models and metadata)
- Pipeline orchestration and error handling
- Command-line arguments and entry points

## Logging

The training module uses Python's logging module with INFO level by default:

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Loading training data...")
logger.info(f"Features: {X.shape[1]}, Target distribution: {y.value_counts()}")
logger.info("Training complete")
```

Logs show:
- Data loading progress and statistics
- Pipeline construction details
- Training progress
- Test set evaluation metrics
- File save locations

## Future Improvements

- [ ] **Hyperparameter Tuning**: GridSearchCV/RandomizedSearchCV for optimal hyperparameters
- [ ] **Model Versioning**: Timestamp-based or git-hash-based versioning
- [ ] **Automatic Model Promotion**: Promote models based on metric thresholds
- [ ] **Cross-Validation**: K-fold cross-validation for more robust evaluation
- [ ] **Feature Importance**: Extract and log feature importance from tree-based models
- [ ] **Model Registry**: Track all trained models with their metrics
- [ ] **Retraining Pipeline**: Automated retraining on new data
- [ ] **Performance Monitoring**: Track model performance drift in production

## Dependencies

Key dependencies (defined in `pyproject.toml`):
- `pandas>=2.3.3`: Data manipulation
- `scikit-learn==1.7.1`: Machine learning models and preprocessing
- `joblib>=1.5.2`: Model serialization
- `pytest>=9.0.2` (dev): Testing framework
- `pytest-cov>=7.0.0` (dev): Coverage reporting

## Author Notes

The training module was productionalized from a Jupyter notebook (`ensemble_model.py`) to enable:
- Reproducible, version-controlled training
- Integration with feature engineering pipeline
- Containerized deployment via Docker
- Easy retraining with new data
- Comprehensive test coverage
