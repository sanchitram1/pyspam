#!/usr/bin/env python3
"""Test script to load and test joblib models with data from jsonl file."""

import json
from pathlib import Path

import joblib
import pandas as pd


def main():
    # Path to the data file
    jsonl_file = Path("data/20251207-bq-results-with-mid-pkgs.jsonl")

    # Read the first row
    with open(jsonl_file) as f:
        first_row = json.loads(f.readline())

    print("First row data loaded:")
    print(json.dumps(first_row, indent=2))
    print("\n" + "=" * 80 + "\n")

    # Convert to DataFrame for models (drop pkg_name and is_spam like in training)
    df = pd.DataFrame([first_row])
    drop_cols = ["pkg_name", "is_spam"]
    X_raw = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    print(f"Raw input shape: {X_raw.shape}")
    print(f"Columns: {list(X_raw.columns[:5])}...")
    print("\n" + "=" * 80 + "\n")

    # Load and apply preprocessor
    print("Loading preprocessor...")
    preprocessor = joblib.load("fitted_preprocessor.joblib")
    X_preprocessed = preprocessor.transform(X_raw)
    print(f"After preprocessing: shape {X_preprocessed.shape}")

    # Load and apply scaler
    print("Loading scaler...")
    scaler = joblib.load("fitted_scaler.joblib")
    X_scaled = scaler.transform(X_preprocessed)
    print(f"After scaling: shape {X_scaled.shape}")
    
    # FIXME: Hack to add constant term
    # The training pipeline adds a constant column after scaling (statsmodels.tools.tools.add_constant),
    # but this step wasn't captured in the saved joblib models. We manually add it here.
    import numpy as np
    X_with_constant = np.column_stack([np.ones((X_scaled.shape[0], 1)), X_scaled])
    print(f"After adding constant: shape {X_with_constant.shape}")
    print("\n" + "=" * 80 + "\n")

    # List of classifier models to test
    classifier_models = [
        "dtc_spam_classifier.joblib",
        "log_reg_spam_classifier.joblib",
        "rf_spam_classifier.joblib",
        "svm_linear_spam_classifier.joblib",
    ]

    for model_name in classifier_models:
        model_path = Path(model_name)
        if not model_path.exists():
            print(f"[SKIP] {model_name} - file not found")
            continue

        try:
            model = joblib.load(model_path)
            print(f"[LOAD OK] {model_name}")
            print(f"  Type: {type(model).__name__}")

            if hasattr(model, "classes_"):
                print(f"  Classes: {model.classes_}")
            if hasattr(model, "n_features_in_"):
                print(f"  Features expected: {model.n_features_in_}")

            # Predict
            try:
                prediction = model.predict(X_with_constant)
                print(f"  Prediction: {prediction[0]} (0=not spam, 1=spam)")

                # Get probability if available
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_with_constant)
                    print(f"  Probability [not spam, spam]: {proba[0]}")
            except Exception as e:
                print(f"  Predict failed: {type(e).__name__}: {e}")

        except Exception as e:
            print(f"[LOAD FAIL] {model_name}")
            print(f"  Error: {type(e).__name__}: {e}")

        print()


if __name__ == "__main__":
    main()
