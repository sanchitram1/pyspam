import os
import time

import joblib
import jwt
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.bq import fetch_package_metadata
from feature_engineering.pipeline import transform_single_package

app = FastAPI(title="PySpam API", description="API for PySpam")
security = HTTPBearer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://sanchit-ram.web.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

API_TOKEN_SECRET = os.environ.get("API_TOKEN_SECRET")

if not API_TOKEN_SECRET:
    raise RuntimeError("API_TOKEN_SECRET not set")


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "PySpam API is running"}


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        # check signature AND expiration
        payload = jwt.decode(token, API_TOKEN_SECRET, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            401, detail="Token expired. Go to my portfolio to get a new one"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(401, detail="Invalid token.")


@app.get("/scan/{package_name}")
def scan_package(package_name: str, auth_data: dict = Depends(verify_token)):
    # 1. Fetch Raw Data (BigQuery)
    print(f"Fetching {package_name}...")
    raw_data = fetch_package_metadata(package_name)

    if not raw_data:
        raise HTTPException(
            status_code=404, detail="Package not found in PyPI public dataset"
        )

    # 2. Transform Features (Pipeline)
    print("Running feature engineering...")
    try:
        # Pass the dict directly to our new adapter
        features_json = transform_single_package(raw_data)

        # Make the DF for model.predict() to work
        features = pd.DataFrame.from_dict([features_json])
    except Exception as e:
        print(f"Pipeline Error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Feature engineering failed: {str(e)}"
        )

    # 3. Predict
    # Load the model
    model = joblib.load("models/ensemble.joblib")

    # Try and predict, and throw if it fails
    try:
        prediction = model.predict_proba(features)[0][1]
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return jsonable_encoder(
        {
            "package": package_name,
            "raw_data": raw_data,
            "features": features_json,
            "prediction": prediction,
            "user_type": auth_data.get("sub"),
        }
    )


@app.post("/generate-key")
def generate_key(service_secret: str = None):
    """Generate a JWT token. Public users get 1 hour, service accounts get 30 days"""
    now = time.time()
    
    # Check if this is a service account request
    if service_secret and service_secret == os.environ.get("SERVICE_ACCOUNT_SECRET"):
        exp_time = 86400 * 30  # 30 days
        sub = "mcp-service"
        expires_in = "2592000 seconds (30 days)"
    else:
        exp_time = 3600  # 1 hour
        sub = "portfolio-visitor"
        expires_in = "3600 seconds"
    
    payload = {
        "sub": sub,
        "iat": now,
        "exp": now + exp_time,
    }
    token = jwt.encode(payload, API_TOKEN_SECRET, algorithm="HS256")
    return {
        "token": token,
        "expires_in": expires_in,
        "note": "Include this in the 'Authorization: Bearer <token>' header",
    }
