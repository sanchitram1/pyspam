INPUT_PATH = "data/~/Downloads/bq-results-20251207-093855-1765100428299.json"
OUTPUT_PATH = "data/20251207-bq-results-with-mid-pkgs.jsonl"

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
