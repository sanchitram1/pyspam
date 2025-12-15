INSERT INTO `pyspam.ground_truth` (package_name, is_spam)

WITH package_stats AS (
  -- 1. AGGREGATE METADATA
  SELECT
    name,
    MIN(upload_time) as first_release,
    MAX(upload_time) as last_release,
    
    -- We need these for the logic, even if we don't insert them
    ANY_VALUE(home_page) as home_page,
    ARRAY_TO_STRING(ANY_VALUE(project_urls), ' ') as all_urls_string,
    
    COUNT(DISTINCT version) as n_versions,
    -- Check if array has elements to count classifiers
    ARRAY_LENGTH(ANY_VALUE(classifiers)) as n_classifiers, 
    ANY_VALUE(home_page) as home_page

  FROM `bigquery-public-data.pypi.distribution_metadata`
  GROUP BY name
),

recent_downloads AS (
  -- 2. AGGREGATE DOWNLOADS
  SELECT
    file.project as name,
    COUNT(*) as downloads_30d
  FROM `bigquery-public-data.pypi.file_downloads`
  WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  GROUP BY name
)

-- 3. SELECT AND INSERT
SELECT
  stats.name,
  0 as is_spam -- Hardcoding 0 because we are mining BENIGN packages here

FROM package_stats stats
LEFT JOIN recent_downloads dl
  ON stats.name = dl.name

WHERE
  -- Pattern: Low downloads (including 0), old history, recent update
  IFNULL(dl.downloads_30d, 0) < 10
  AND stats.n_versions > 15
  AND stats.name NOT IN (SELECT package_name FROM `pyspam.ground_truth`)

LIMIT 500