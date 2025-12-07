import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import your helper modules
from services.bigquery import get_package_data

# Assuming pipeline.py has a function called 'clean_and_transform'
from feature_engineering.pipeline import main as pipeline_main

app = FastAPI(title="PySpam Scanner API")

# TODO: this will error
# the idea is that it loads the model from the model/ directory
# where should this directory even be?
# what should the model name be?
# anyway, it also coordinates the bigquery fetch
# and then the running of pipeline.
# we'll see how quick that is to run live
# it needs the model/random_forest_v1.pkl file
# finally, how should we return the result?

# --- 1. LOAD MODEL AT STARTUP ---
# Loading models is slow, so we do it once when the server starts,
# not every time a user makes a request.
model = None


@app.on_event("startup")
def load_model():
    global model
    # Load your trained model (ensure scikit-learn versions match!)
    model = joblib.load("model/random_forest_v1.pkl")
    print("✅ Model loaded successfully")


# --- 2. DEFINE RESPONSE FORMAT ---
class ScanResult(BaseModel):
    package_name: str
    is_spam: bool
    spam_probability: float
    risk_factors: list[str]  # For explainability


# --- 3. THE ENDPOINT ---
@app.post("/scan/{package_name}", response_model=ScanResult)
async def scan_package(package_name: str):
    print(f"🔎 Scanning package: {package_name}...")

    # Step A: Get Raw Data from BigQuery
    raw_df = get_package_data(package_name)
    if raw_df is None:
        raise HTTPException(
            status_code=404, detail="Package not found in PyPI BigQuery dataset"
        )

    # Step B: Run the Pipeline (Feature Engineering)
    # This transforms the raw BQ row into the exact X format the model expects
    features_df = pipeline_main(raw_df)

    # Step C: Predict
    # We use predict_proba to get a "Risk Score" (e.g., 0.85), not just True/False
    probability = model.predict_proba(features_df)[0][
        1
    ]  # Probability of Class 1 (Spam)
    is_spam = probability > 0.5

    # Step D: Generate simple heuristics for the user
    # (You can make this fancier later)
    reasons = []
    if probability > 0.5:
        if features_df["n_downloads_30d"].iloc[0] < 50:
            reasons.append("Extremely low popularity")
        if features_df["n_classifiers"].iloc[0] > 10:
            reasons.append("Suspiciously high classifier count")

    return {
        "package_name": package_name,
        "is_spam": bool(is_spam),
        "spam_probability": round(probability, 4),
        "risk_factors": reasons,
    }
