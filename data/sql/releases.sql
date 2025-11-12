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
sample as ( 
    SELECT 
      package_name
    FROM 
      `pyspam.deduplicated_spam_names`
    UNION ALL
    SELECT 
      package_name
    FROM 
      `pyspam.non_spam_packages`
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
  JOIN sample ON agg.pkg_name = sample.package_name
)

SELECT *
FROM final
ORDER BY pkg_name;