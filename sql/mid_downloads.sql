WITH package_downloads AS (
  SELECT
    file.project AS name,
    COUNT(*) AS download_count
  FROM `bigquery-public-data.pypi.file_downloads`
  WHERE
    -- timestamp filter must be inside the WHERE of the first CTE
    timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  GROUP BY 1
  HAVING download_count BETWEEN 100 AND 1000
),

latest_metadata AS (
  SELECT
    name,
    version,
    upload_time,
    summary,
    description,
    author,
    home_page,
    license
  FROM `bigquery-public-data.pypi.distribution_metadata`
  -- QUALIFY handles the "Latest Version" logic
  QUALIFY ROW_NUMBER() OVER (PARTITION BY name ORDER BY upload_time DESC) = 1
)

SELECT
  m.name,
  m.version,
  d.download_count,
  m.upload_time,
  m.summary,
  m.home_page
FROM latest_metadata m
INNER JOIN package_downloads d
  ON m.name = d.name
WHERE
  -- Safety filter goes here at the end
  m.upload_time < TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 365 DAY)
ORDER BY d.download_count DESC
LIMIT 5000