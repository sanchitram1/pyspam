import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.bq import fetch_package_metadata
from feature_engineering.pipeline import transform_single_package

app = FastAPI(title="PySpam API", description="API for PySpam")


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "PySpam API is running"}


@app.get("/scan/{package_name}")
def scan_package(package_name: str):
    # 1. Fetch Raw Data (BigQuery)
    print(f"***** Fetching {package_name}...")
    raw_data = fetch_package_metadata(package_name)

    if not raw_data:
        raise HTTPException(
            status_code=404, detail="Package not found in PyPI public dataset"
        )

    # 2. Transform Features (Pipeline)
    # TODO: it's broken...lol
    print("Running feature engineering...")
    try:
        # Pass the dict directly to our new adapter
        features_json = transform_single_package(raw_data)
        features = pd.DataFrame.from_dict([features_json])
    except Exception as e:
        print(f"Pipeline Error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Feature engineering failed: {str(e)}"
        )

    # TODO 3. Predict
    model = joblib.load("models/ensemble.joblib")

    try:
        prediction = model.predict_proba(features)[0][1]
    except Exception as e:
        print(f"******* Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # For now, let's just return the features to prove the pipeline works
    # TODO: the return should include prediction, AND features, AND raw_data

    return {
        "package": package_name,
        "features_generated": len(features_json),
        "sample_feature": features_json.get("n_name_len"),  # Just to check logic
        "full_features": features_json,
        "prediction": prediction,
    }
