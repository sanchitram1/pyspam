from google.cloud import bigquery

# Initialize Client (It looks for GOOGLE_APPLICATION_CREDENTIALS env var)
client = bigquery.Client()

# TODO: this will error
# the idea is that it runs training.sql for one specific package
# and returns that result
# it needs Google BQ credentials


def get_package_data(package_name: str):
    # Load query from file
    with open("sql/fetch_package.sql", "r") as f:
        query = f.read()

    # Configure the query parameter to prevent SQL Injection
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("package_name", "STRING", package_name)
        ]
    )

    # Run the query
    query_job = client.query(query, job_config=job_config)
    result = query_job.to_dataframe()

    if result.empty:
        return None

    return result
