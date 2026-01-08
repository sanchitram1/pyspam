# Training Pipeline

Productionalized training workflow for the spam detection model.

## Architecture

```
Raw Data (JSONL)
      ↓
[feature_engineering/pipeline.py] → features_engineered.jsonl
      ↓
[training/train.py] → model_production.joblib + model_metadata.json
      ↓
[Docker build] → Push to Google Cloud Run
```

## Configuration

All pipeline parameters are centralized in `training/config.py`:
- **Input/Output paths**: Standardized filenames and directories
- **Model selection metrics**: Primary metric and evaluation metrics
- **Train-test split parameters**: Test size, random state, stratification
- **Ensemble configuration**: Model choices and hyperparameters

## Usage

### Option 1: Full Pipeline (Feature Engineering + Training)

```bash
python -m training.pipeline \
  --raw-data /path/to/raw_data.jsonl \
  --features-output data/features_engineered.jsonl \
  --models-output models/
```

### Option 2: Training Only (if features already exist)

```bash
python -m training.train \
  --input data/features_engineered.jsonl \
  --output models/
```

### Option 3: Use Defaults

```bash
# Full pipeline with default paths
python -m training.pipeline --raw-data data/raw_data.jsonl

# Training only with default paths
python -m training.train
```

## Output

The training script generates two files:

### `model_production.joblib`
The trained sklearn Pipeline containing:
- OneHotEncoder for categorical features
- StandardScaler for numerical features
- VotingClassifier with 4 base models:
  - Decision Tree
  - Logistic Regression
  - Random Forest (200 estimators)
  - SVM (linear kernel)

### `model_metadata.json`
Metadata about the model:
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

## Model Selection

Models are evaluated on test set metrics (see `training/config.py`). The primary metric is **ROC-AUC**. 

To select a different model:
1. Train with different hyperparameters by editing `ENSEMBLE_CONFIG` in `config.py`
2. Compare metrics in `model_metadata.json`
3. Manually move/promote the best model if needed (for now, human-in-the-loop)

## Integration with Docker

The Dockerfile includes the trained model artifacts:
```dockerfile
COPY models/ ./models/
```

The API loads the model:
```python
import joblib
model = joblib.load("models/model_production.joblib")
```

## Next Steps

- [ ] Implement hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
- [ ] Add model versioning (timestamp-based or git-hash)
- [ ] Automate model promotion based on metrics thresholds
- [ ] Add cross-validation for more robust evaluation
- [ ] Cache preprocessor fit state for consistency
