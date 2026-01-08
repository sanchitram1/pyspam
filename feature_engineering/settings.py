from pathlib import Path

# Use standardized paths that coordinate with training pipeline
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Default input/output paths (can be overridden via CLI args)
INPUT_PATH = str(DATA_DIR / "raw_data.jsonl")
OUTPUT_PATH = str(DATA_DIR / "features_engineered.jsonl")

# Example reference sets (you should replace / extend these)
TOP_LEGIT_PACKAGES = [
    "requests",
    "numpy",
    "pandas",
    "flask",
    "django",
    "scipy",
    "pytest",
    "matplotlib",
    "scikit-learn",
]

BRAND_ALIASES = [
    "google",
    "microsoft",
    "amazon",
    "facebook",
    "meta",
    "apple",
    "paypal",
    "stripe",
    "github",
    "gitlab",
]

TOP_BRAND_PKGS = [
    "django",
    "flask",
    "requests",
    "tensorflow",
    "torch",
    "boto3",
    "azure",
    "google-cloud-storage",
    "stripe",
]

# Numerical replacements
NUM_FILL_VALUES = {
    "t_median_release_gap_days": 0,
    "n_downloads_7d": 0,
    "n_downloads_30d": 0,
    "n_dependents_est": 0,
    "min_dep_lev_to_brand": 20,
}

# Threshold for "typosquatting" closeness
LEV_THRESHOLD = 2

# Threshold for "low download" packages (you can tune this)
LOW_DOWNLOAD_THRESHOLD_30D = 50
