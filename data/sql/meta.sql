-- General dump of all pypi.distribution_metadata table
SELECT
  name,
  STRING_AGG(DISTINCT license, ', ') AS licenses,
  COUNT(DISTINCT version) AS version_count,
  COUNT(DISTINCT IF(has_signature IS TRUE, version, NULL)) AS signed_version_count,
  MAX(upload_time) AS last_upload,
  MIN(upload_time) AS first_upload,
  
  -- Get metadata from the latest version
  ARRAY_FIRST(
    ARRAY_AGG(summary ORDER BY upload_time DESC LIMIT 1)
  ) AS latest_summary,
  ARRAY_FIRST(
    ARRAY_AGG(description ORDER BY upload_time DESC LIMIT 1)
  ) AS latest_description,

  -- Get all distinct authors and maintainers
  ARRAY_AGG(DISTINCT author_email IGNORE NULLS) AS distinct_authors,
  ARRAY_AGG(DISTINCT maintainer_email IGNORE NULLS) AS distinct_maintainers,
  
  -- Get all distinct keywords & classifiers
  ARRAY_AGG(DISTINCT keywords IGNORE NULLS) AS distinct_keywords,
  (
    SELECT ARRAY_AGG(DISTINCT k IGNORE NULLS)
    FROM `bigquery-public-data.pypi.distribution_metadata` AS t
    LEFT JOIN UNNEST(t.classifiers) AS k
    WHERE t.name = dm.name
  ) AS distinct_classifiers,
  
  -- Get project_urls array from the latest version
  ARRAY_FIRST(
    ARRAY_AGG(STRUCT(project_urls) ORDER BY upload_time DESC LIMIT 1)
  ).project_urls AS latest_project_urls,
  
  -- Get dependency count from the latest version
  COALESCE(ARRAY_LENGTH(
    ARRAY_FIRST(
      ARRAY_AGG(STRUCT(requires_dist) ORDER BY upload_time DESC LIMIT 1)
    ).requires_dist
  ), 0) AS latest_dependency_count,

  -- Get list of dependencies
  ARRAY_FIRST(
    ARRAY_AGG(STRUCT(requires_dist) ORDER BY upload_time DESC LIMIT 1)
  ).requires_dist AS latest_dependencies
  -- COUNT(fd.timestamp) AS total_downloads,
  -- COUNT(DISTINCT DATE(fd.timestamp)) AS num_download_days
FROM
  `bigquery-public-data.pypi.distribution_metadata` as dm
JOIN 
  (
    SELECT 
      package_name
    FROM 
      `pyspam.deduplicated_spam_names`
    UNION ALL
    SELECT 
      package_name
    FROM 
      `pyspam.non_spam_packages`   
  ) AS sample ON sample.package_name = dm.name
GROUP BY name;