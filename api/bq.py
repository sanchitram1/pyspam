import argparse
import json
import logging
from datetime import datetime

from google.cloud import bigquery

# Configure logging for standalone use
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize client once
# Ensure you have run `gcloud auth application-default login` locally
try:
    client = bigquery.Client()
except Exception as e:
    logger.warning(
        f"Could not initialize BigQuery client. Check credentials. Error: {e}"
    )
    client = None


def fetch_package_metadata(package_name: str):
    """
    Fetches raw metadata for a specific package from BigQuery.
    Returns a Dictionary (JSON) or None if not found.
    """
    if not client:
        logger.error("BigQuery client is not initialized.")
        return None

    # This query grabs the LATEST version of the package
    query = """
    SELECT
        name,
        version,
        upload_time,
        summary,
        description,
        author,
        author_email,
        maintainer,
        maintainer_email,
        home_page,
        license,
        project_urls,
        keywords,
        classifiers,
        requires_dist,
        requires_python,
        requires_external
    FROM `bigquery-public-data.pypi.distribution_metadata`
    WHERE name = @pkg_name
    ORDER BY upload_time DESC
    LIMIT 1
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("pkg_name", "STRING", package_name)
        ]
    )

    try:
        query_job = client.query(query, job_config=job_config)
        df = query_job.to_dataframe()
    except Exception as e:
        logger.error(f"BigQuery execution failed: {e}")
        return None

    if df.empty:
        return None

    # Convert timestamps to strings so JSON serialization doesn't fail later
    records = df.to_dict(orient="records")
    record = records[0]

    # Helper to serialize datetimes for JSON output
    for key, value in record.items():
        if isinstance(value, datetime):
            record[key] = value.isoformat()

    return record


if __name__ == "__main__":
    # CLI Entry Point
    parser = argparse.ArgumentParser(
        description="Fetch raw PyPI metadata from BigQuery"
    )
    parser.add_argument(
        "package_name", help="The name of the package to search for (e.g. requests)"
    )

    args = parser.parse_args()

    print(f"🔍 Fetching metadata for: {args.package_name}...")
    result = fetch_package_metadata(args.package_name)

    if result:
        print(f"✅ Found package: {result['name']} (v{result['version']})")
        print("-" * 40)
        # Pretty print the JSON output
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"❌ Package '{args.package_name}' not found in BigQuery dataset.")
