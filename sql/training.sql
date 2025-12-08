WITH all_metadata AS (
  SELECT
    name,
    STRING_AGG(DISTINCT license, ', ') AS licenses,
    STRING_AGG(version) as versions,
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
    ).requires_dist AS latest_dependencies,
    avg(gt.is_spam) as is_spam
  FROM
    `bigquery-public-data.pypi.distribution_metadata` as dm
  JOIN 
    `pyspam.ground_truth` as gt on dm.name = gt.package_name
  GROUP BY name
), 
agg_releases AS (
  -- Compute release/activity features for all PyPI packages
  WITH releases AS (
    SELECT
      name AS pkg_name,
      version,
      TIMESTAMP(upload_time) AS upload_ts
    FROM `bigquery-public-data.pypi.distribution_metadata`
    WHERE upload_time IS NOT NULL
  ),
  per_release AS (
    -- Add ordering and inter-release gaps per package
    SELECT
      pkg_name,
      version,
      upload_ts,
      LAG(upload_ts) OVER (PARTITION BY pkg_name ORDER BY upload_ts) AS prev_upload_ts,
      ROW_NUMBER() OVER (PARTITION BY pkg_name ORDER BY upload_ts DESC) AS rn_desc,
      MAX(upload_ts) OVER (PARTITION BY pkg_name) AS last_upload_ts,
      MIN(upload_ts) OVER (PARTITION BY pkg_name) AS first_upload_ts
    FROM releases
  ),
  gaps AS (
    SELECT
      pkg_name,
      -- gap in days between consecutive releases
      DATE_DIFF(DATE(upload_ts), DATE(prev_upload_ts), DAY) AS gap_days
    FROM per_release
    WHERE prev_upload_ts IS NOT NULL
  ),
  agg AS (
    SELECT
      p.pkg_name,

      -- t_age_first_release_days: days since first release
      DATE_DIFF(CURRENT_DATE(), DATE(MIN(p.first_upload_ts)), DAY) AS t_age_first_release_days,

      -- t_age_last_release_days: days since most recent release
      DATE_DIFF(CURRENT_DATE(), DATE(MAX(p.last_upload_ts)), DAY)  AS t_age_last_release_days,

      -- n_versions: count of distinct versions observed
      COUNT(DISTINCT p.version) AS n_versions,

      -- t_median_release_gap_days: median gap between releases
      -- (use approx quantiles; if only one release, will be NULL)
      -- IFNULL(
      --   (SELECT APPROX_QUANTILES(gap_days, 2)[OFFSET(1)] FROM gaps g WHERE g.pkg_name = p.pkg_name),
      --   NULL
      -- ) AS t_median_release_gap_days,

      -- has_single_release: exactly one version in metadata
      COUNT(DISTINCT p.version) = 1 AS has_single_release,

      -- Last release timestamp (for the final two categorical features)
      MAX(p.last_upload_ts) AS _last_upload_ts
    FROM per_release p
    GROUP BY pkg_name
  ),
  final AS (
    SELECT
      pkg_name,
      t_age_first_release_days,
      t_age_last_release_days,
      n_versions,
      -- t_median_release_gap_days,
      has_single_release,

      -- t_time_of_day_bucket (UTC) based on hour of last release
      CASE
        WHEN EXTRACT(HOUR FROM _last_upload_ts) BETWEEN 0 AND 5   THEN 'night_00_05'
        WHEN EXTRACT(HOUR FROM _last_upload_ts) BETWEEN 6 AND 11  THEN 'morning_06_11'
        WHEN EXTRACT(HOUR FROM _last_upload_ts) BETWEEN 12 AND 17 THEN 'afternoon_12_17'
        ELSE 'evening_18_23'
      END AS t_time_of_day_bucket,

      -- cat_weekday_of_last_release (UTC)
      FORMAT_TIMESTAMP('%A', _last_upload_ts) AS cat_weekday_of_last_release
    FROM agg
    JOIN `pyspam.ground_truth` AS gt ON gt.package_name = agg.pkg_name
  )
  SELECT *
  FROM final
),
urls_and_licenses AS (
  -- Features per project (latest release only), no correlated subqueries
  WITH latest AS (
    SELECT
      name,
      version,
      upload_time,
      home_page,
      project_urls,
      download_url,
      license,
      classifiers,
      -- lowercase, concatenated link surface to simplify detection
      LOWER(CONCAT(IFNULL(home_page, ''), ' ',
                  IFNULL(ARRAY_TO_STRING(project_urls, ','), ''), ' ',
                  IFNULL(download_url, ''))) AS urls_lc,
      ROW_NUMBER() OVER (PARTITION BY name ORDER BY upload_time DESC) AS rn
    FROM `bigquery-public-data.pypi.distribution_metadata`
  )

  SELECT
    name,
    version,
    upload_time,

    -- has_homepage: true if a proper URL is present in home_page
    IF(REGEXP_CONTAINS(LOWER(IFNULL(home_page, '')), r'https?://'), 1, 0) AS has_homepage,

    -- has_repo_url: any known forge host appears anywhere in the declared URLs
    IF(REGEXP_CONTAINS(urls_lc,
        r'(github\.com|gitlab\.com|bitbucket\.org|codeberg\.org|sourceforge\.net|gitee\.com)'),
      1, 0) AS has_repo_url,

    -- cat_repo_host: best-effort classification of the repository host
    CASE
      WHEN REGEXP_CONTAINS(urls_lc, r'github\.com')      THEN 'github'
      WHEN REGEXP_CONTAINS(urls_lc, r'gitlab\.com')      THEN 'gitlab'
      WHEN REGEXP_CONTAINS(urls_lc, r'bitbucket\.org')   THEN 'bitbucket'
      WHEN REGEXP_CONTAINS(urls_lc, r'codeberg\.org')    THEN 'codeberg'
      WHEN REGEXP_CONTAINS(urls_lc, r'sourceforge\.net') THEN 'sourceforge'
      WHEN REGEXP_CONTAINS(urls_lc, r'gitee\.com')       THEN 'gitee'
      ELSE 'unknown'
    END AS cat_repo_host,

    -- has_issue_tracker: common patterns for issue trackers on forges/Jira/etc.
    IF(REGEXP_CONTAINS(urls_lc,
        r'(/issues|/pulls|/merge_requests|youtrack|bugzilla|jira)'),
      1, 0) AS has_issue_tracker,

    -- has_docs_url: readthedocs, docs subdomain, or '/docs' path seen anywhere
    IF(REGEXP_CONTAINS(urls_lc,
        r'(readthedocs|(^|\s)https?://docs\.|/docs(/|$))'),
      1, 0) AS has_docs_url,

    -- has_license: explicit license string OR a license classifier present
    IF(
      (license IS NOT NULL AND TRIM(license) <> '')
      OR REGEXP_CONTAINS(LOWER(ARRAY_TO_STRING(classifiers, ' | ')), r'\blicense\s*::'),
      1, 0
    ) AS has_license

  FROM latest
  JOIN `pyspam.ground_truth` AS gt ON gt.package_name = latest.name
  WHERE rn = 1
)

SELECT
  am.name, 
  am.versions,
  ar.n_versions,
  am.signed_version_count as n_signed_version_count,
  ual.version as latest_version,
  am.first_upload as t_first_release,
  ar.t_age_first_release_days,
  am.last_upload as t_last_release,
  ar.t_age_last_release_days,
  ar.has_single_release,
  ar.t_time_of_day_bucket,
  am.last_upload,
  ual.has_homepage,
  ual.has_repo_url,
  ual.cat_repo_host,
  ual.has_issue_tracker,
  ual.has_docs_url,
  ual.has_license,
  am.latest_project_urls,
  am.licenses,
  am.latest_summary,
  am.latest_description,
  am.distinct_authors,
  am.distinct_maintainers,
  am.distinct_keywords,
  am.distinct_classifiers,
  am.latest_dependency_count as n_latest_dependency_count,
  am.latest_dependencies,
  am.is_spam
FROM 
  all_metadata AS am 
JOIN 
  agg_releases AS ar 
ON 
  am.name = ar.pkg_name
JOIN 
  urls_and_licenses AS ual
ON 
  am.name = ual.name
