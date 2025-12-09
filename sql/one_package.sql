-- ============================================================
-- PyPI feature table (SQL-derivable features + raw fields)
-- Restricted to packages in pyspam.ground_truth
-- ============================================================


WITH
-- ------------------------------------------------------------------
-- 0. Limit distribution metadata to labeled packages
-- ------------------------------------------------------------------
labeled_dist AS (
  SELECT dm.*
  FROM `bigquery-public-data.pypi.distribution_metadata` AS dm
  WHERE dm.name = @pkg_name
),

-- ------------------------------------------------------------------
-- A. Package Identity & Naming Features (A.*)
--   Uses only pyspam.ground_truth.package_name
--   NOTE: Levenshtein / TF-IDF-based features to be done offline.
-- ------------------------------------------------------------------
name_features AS (
  SELECT
    @pkg_name AS pkg_name,
    LENGTH(@pkg_name) AS n_name_len,
    REGEXP_CONTAINS(@pkg_name, r'\d') AS has_digit_in_name,
    REGEXP_CONTAINS(@pkg_name, r'[-_]') AS has_dash_or_underscore,
    CASE
      WHEN @pkg_name = LOWER(@pkg_name) THEN 'lower'
      WHEN REGEXP_CONTAINS(
        @pkg_name,
        r'^[a-z]+([A-Z][a-z0-9]*)+$'
      ) THEN 'camel'
      ELSE 'mixed'
    END AS cat_name_case
),


-- ------------------------------------------------------------------
-- 1. Aggregate metadata per project (your original all_metadata)
-- ------------------------------------------------------------------
all_metadata AS (
  SELECT
    dm.name,

    STRING_AGG(DISTINCT dm.license, ', ') AS licenses,
    STRING_AGG(dm.version) AS versions,
    COUNT(DISTINCT dm.version) AS version_count,
    COUNT(DISTINCT IF(dm.has_signature IS TRUE, dm.version, NULL))
      AS signed_version_count,

    MAX(dm.upload_time) AS last_upload,
    MIN(dm.upload_time) AS first_upload,

    ARRAY_FIRST(
      ARRAY_AGG(dm.summary ORDER BY dm.upload_time DESC LIMIT 1)
    ) AS latest_summary,

    ARRAY_FIRST(
      ARRAY_AGG(dm.description ORDER BY dm.upload_time DESC LIMIT 1)
    ) AS latest_description,

    ARRAY_AGG(DISTINCT dm.author_email IGNORE NULLS) AS distinct_authors,
    ARRAY_AGG(DISTINCT dm.maintainer_email IGNORE NULLS)
      AS distinct_maintainers,

    ARRAY_AGG(DISTINCT dm.keywords IGNORE NULLS) AS distinct_keywords,

    (
      SELECT ARRAY_AGG(DISTINCT k IGNORE NULLS)
      FROM `bigquery-public-data.pypi.distribution_metadata` AS t
      LEFT JOIN UNNEST(t.classifiers) AS k
      WHERE t.name = dm.name
    ) AS distinct_classifiers,

    ARRAY_FIRST(
      ARRAY_AGG(STRUCT(dm.project_urls) ORDER BY dm.upload_time DESC LIMIT 1)
    ).project_urls AS latest_project_urls,

    ARRAY_FIRST(
      ARRAY_AGG(STRUCT(dm.requires_dist) ORDER BY dm.upload_time DESC LIMIT 1)
    ).requires_dist AS latest_dependencies,

    COALESCE(
      ARRAY_LENGTH(
        ARRAY_FIRST(
          ARRAY_AGG(STRUCT(dm.requires_dist)
                    ORDER BY dm.upload_time DESC LIMIT 1)
        ).requires_dist
      ),
      0
    ) AS latest_dependency_count

  FROM labeled_dist AS dm
  GROUP BY dm.name
),


-- ------------------------------------------------------------------
-- B. Summary & Description Features (B.*)
-- ------------------------------------------------------------------
text_features AS (
  SELECT
    am.name AS pkg_name,

    -- B.n_summary_len
    LENGTH(IFNULL(am.latest_summary, '')) AS n_summary_len,

    -- B.n_desc_len
    LENGTH(IFNULL(am.latest_description, '')) AS n_desc_len,

    -- B.n_desc_lines
    ARRAY_LENGTH(
      SPLIT(IFNULL(am.latest_description, ''), '\n')
    ) AS n_desc_lines,

    -- B.has_code_block_in_desc
    IF(
      REGEXP_CONTAINS(
        IFNULL(am.latest_description, ''),
        r'```|::\s*\n'
      ),
      TRUE, FALSE
    ) AS has_code_block_in_desc,

    -- B.n_urls_in_desc
    ARRAY_LENGTH(
      REGEXP_EXTRACT_ALL(
        LOWER(IFNULL(am.latest_description, '')),
        r'https?://[^\s)]+'
      )
    ) AS n_urls_in_desc,

    -- B.has_suspicious_kw (adjust regex as needed)
    IF(
      REGEXP_CONTAINS(
        LOWER(IFNULL(am.latest_description, '')),
        r'(bitcoin|mining|keylogger|crack|hack tool|stealer|remote control)'
      ),
      TRUE, FALSE
    ) AS has_suspicious_kw,

    -- B.pct_non_ascii_desc (approximate)
    SAFE_DIVIDE(
      BYTE_LENGTH(IFNULL(am.latest_description, '')) -
      CHAR_LENGTH(IFNULL(am.latest_description, '')),
      GREATEST(
        BYTE_LENGTH(IFNULL(am.latest_description, '')), 1
      )
    ) AS pct_non_ascii_desc

  FROM all_metadata AS am
),

-- ------------------------------------------------------------------
-- C. Release & Activity Features (C.*)
-- ------------------------------------------------------------------
releases AS (
  SELECT
    name AS pkg_name,
    version,
    TIMESTAMP(upload_time) AS upload_ts
  FROM labeled_dist
  WHERE upload_time IS NOT NULL
),
per_release AS (
  SELECT
    r.pkg_name,
    r.version,
    r.upload_ts,
    LAG(r.upload_ts) OVER (
      PARTITION BY r.pkg_name ORDER BY r.upload_ts
    ) AS prev_upload_ts,
    MAX(r.upload_ts) OVER (
      PARTITION BY r.pkg_name
    ) AS last_upload_ts,
    MIN(r.upload_ts) OVER (
      PARTITION BY r.pkg_name
    ) AS first_upload_ts
  FROM releases AS r
),
gaps AS (
  SELECT
    pkg_name,
    DATE_DIFF(DATE(upload_ts), DATE(prev_upload_ts), DAY) AS gap_days
  FROM per_release
  WHERE prev_upload_ts IS NOT NULL
),
gap_stats AS (
  SELECT
    pkg_name,
    APPROX_QUANTILES(gap_days, 2)[OFFSET(1)]
      AS t_median_release_gap_days
  FROM gaps
  GROUP BY pkg_name
),
agg_releases AS (
  SELECT
    p.pkg_name,

    DATE_DIFF(
      CURRENT_DATE(),
      DATE(MIN(p.first_upload_ts)),
      DAY
    ) AS t_age_first_release_days,

    DATE_DIFF(
      CURRENT_DATE(),
      DATE(MAX(p.last_upload_ts)),
      DAY
    ) AS t_age_last_release_days,

    COUNT(DISTINCT p.version) AS n_versions,

    COUNT(DISTINCT p.version) = 1 AS has_single_release,

    MAX(p.last_upload_ts) AS _last_upload_ts
  FROM per_release AS p
  GROUP BY p.pkg_name
),
agg_releases_with_gap AS (
  SELECT
    ar.pkg_name,
    ar.t_age_first_release_days,
    ar.t_age_last_release_days,
    ar.n_versions,
    COALESCE(gs.t_median_release_gap_days, NULL)
      AS t_median_release_gap_days,
    ar.has_single_release,

    CASE
      WHEN EXTRACT(HOUR FROM ar._last_upload_ts) BETWEEN 0 AND 5
        THEN 'night_00_05'
      WHEN EXTRACT(HOUR FROM ar._last_upload_ts) BETWEEN 6 AND 11
        THEN 'morning_06_11'
      WHEN EXTRACT(HOUR FROM ar._last_upload_ts) BETWEEN 12 AND 17
        THEN 'afternoon_12_17'
      ELSE 'evening_18_23'
    END AS t_time_of_day_bucket,

    FORMAT_TIMESTAMP('%A', ar._last_upload_ts)
      AS cat_weekday_of_last_release
  FROM agg_releases AS ar
  LEFT JOIN gap_stats AS gs
    ON ar.pkg_name = gs.pkg_name
),

-- ------------------------------------------------------------------
-- E + F. URLs & License / Classifier Features
-- ------------------------------------------------------------------
urls_and_licenses AS (
  WITH latest AS (
    SELECT
      dm.name,
      dm.version,
      dm.upload_time,
      dm.home_page,
      dm.project_urls,
      dm.download_url,
      dm.license,
      dm.classifiers,
      LOWER(
        CONCAT(
          IFNULL(dm.home_page, ''), ' ',
          IFNULL(ARRAY_TO_STRING(dm.project_urls, ','), ''), ' ',
          IFNULL(dm.download_url, '')
        )
      ) AS urls_lc,
      ROW_NUMBER() OVER (
        PARTITION BY dm.name ORDER BY dm.upload_time DESC
      ) AS rn
    FROM labeled_dist AS dm
  )
  SELECT
    l.name,
    l.version,
    l.upload_time,

    IF(
      REGEXP_CONTAINS(
        LOWER(IFNULL(l.home_page, '')),
        r'https?://'
      ),
      1, 0
    ) AS has_homepage,

    IF(
      REGEXP_CONTAINS(
        l.urls_lc,
        r'(github\.com|gitlab\.com|bitbucket\.org|codeberg\.org|sourceforge\.net|gitee\.com)'
      ),
      1, 0
    ) AS has_repo_url,

    CASE
      WHEN REGEXP_CONTAINS(l.urls_lc, r'github\.com')      THEN 'github'
      WHEN REGEXP_CONTAINS(l.urls_lc, r'gitlab\.com')      THEN 'gitlab'
      WHEN REGEXP_CONTAINS(l.urls_lc, r'bitbucket\.org')   THEN 'bitbucket'
      WHEN REGEXP_CONTAINS(l.urls_lc, r'codeberg\.org')    THEN 'codeberg'
      WHEN REGEXP_CONTAINS(l.urls_lc, r'sourceforge\.net') THEN 'sourceforge'
      WHEN REGEXP_CONTAINS(l.urls_lc, r'gitee\.com')       THEN 'gitee'
      ELSE 'unknown'
    END AS cat_repo_host,

    IF(
      REGEXP_CONTAINS(
        l.urls_lc,
        r'(/issues|/pulls|/merge_requests|youtrack|bugzilla|jira)'
      ),
      1, 0
    ) AS has_issue_tracker,

    IF(
      REGEXP_CONTAINS(
        l.urls_lc,
        r'(readthedocs|(^|\s)https?://docs\.|/docs(/|$))'
      ),
      1, 0
    ) AS has_docs_url,

    IF(
      (l.license IS NOT NULL AND TRIM(l.license) <> '')
      OR REGEXP_CONTAINS(
           LOWER(ARRAY_TO_STRING(l.classifiers, ' | ')),
           r'\blicense\s*::'
         ),
      1, 0
    ) AS has_license,

    ARRAY_LENGTH(l.classifiers) AS n_classifiers,

    CASE
      WHEN REGEXP_CONTAINS(
        LOWER(ARRAY_TO_STRING(l.classifiers, ' | ') || ' ' || IFNULL(l.license, '')),
        r'mit license'
      ) THEN 'MIT'
      WHEN REGEXP_CONTAINS(
        LOWER(ARRAY_TO_STRING(l.classifiers, ' | ') || ' ' || IFNULL(l.license, '')),
        r'gnu general public license|gpl'
      ) THEN 'GPL'
      WHEN REGEXP_CONTAINS(
        LOWER(ARRAY_TO_STRING(l.classifiers, ' | ') || ' ' || IFNULL(l.license, '')),
        r'apache license'
      ) THEN 'Apache'
      WHEN REGEXP_CONTAINS(
        LOWER(ARRAY_TO_STRING(l.classifiers, ' | ') || ' ' || IFNULL(l.license, '')),
        r'bsd license'
      ) THEN 'BSD'
      ELSE 'other_or_unknown'
    END AS cat_license_family,

    IF(
      REGEXP_CONTAINS(
        ARRAY_TO_STRING(l.classifiers, ' | '),
        r'Programming Language :: Python'
      ),
      1, 0
    ) AS has_prog_lang_classifier,

    IF(
      REGEXP_CONTAINS(
        ARRAY_TO_STRING(l.classifiers, ' | '),
        r'Typing ::'
      ),
      1, 0
    ) AS has_typing_classifier

  FROM latest AS l
  WHERE l.rn = 1
),

-- ------------------------------------------------------------------
-- D. Ownership & Maintainer Features
-- ------------------------------------------------------------------
maintainer_emails AS (
  SELECT
    am.name AS pkg_name,
    ARRAY_CONCAT(
      COALESCE(am.distinct_authors, []),
      COALESCE(am.distinct_maintainers, [])
    ) AS all_emails,
    am.distinct_authors
  FROM all_metadata AS am
),
maintainer_stats AS (
  SELECT
    me.pkg_name,
    COUNT(*) AS n_maintainers,
    AVG(
      IF(
        REGEXP_CONTAINS(
          email,
          r'@(gmail\.com|outlook\.com|yahoo\.com|hotmail\.com|protonmail\.com)$'
        ),
        1.0, 0.0
      )
    ) AS pct_free_email_domains,
    MAX(
      IF(
        REGEXP_CONTAINS(
          email,
          r'@(mailinator\.com|10minutemail\.com|guerrillamail\.com)'
        ),
        1, 0
      )
    ) AS has_disposable_email
  FROM maintainer_emails AS me,
       UNNEST(me.all_emails) AS email
  GROUP BY me.pkg_name
),
maintainer_features AS (
  SELECT
    am.name AS pkg_name,
    COALESCE(ms.n_maintainers, 0) AS n_maintainers,
    COALESCE(ms.pct_free_email_domains, 0.0) AS pct_free_email_domains,
    COALESCE(ms.has_disposable_email, 0) AS has_disposable_email,
    IF(ARRAY_LENGTH(am.distinct_authors) = 0, 1, 0) AS has_missing_author
  FROM all_metadata AS am
  LEFT JOIN maintainer_stats AS ms
    ON am.name = ms.pkg_name
),

-- ------------------------------------------------------------------
-- H. Dependency Features (H.*)
-- ------------------------------------------------------------------
dependency_features AS (
  SELECT
    am.name AS pkg_name,
    am.latest_dependency_count AS n_requires,

    IF(
      EXISTS (
        SELECT 1
        FROM UNNEST(COALESCE(am.latest_dependencies, [])) AS dep
        WHERE REGEXP_CONTAINS(LOWER(dep), r'extra\s*==')
      ),
      1,
      0
    ) AS has_extras

  FROM all_metadata AS am
),

-- ------------------------------------------------------------------
-- I. Popularity Features (I.*) from file_downloads
-- ------------------------------------------------------------------
downloads_raw AS (
  SELECT
    fd.file.project AS pkg_name,
    fd.timestamp
  FROM `bigquery-public-data.pypi.file_downloads` AS fd
  WHERE fd.file.project = @pkg_name
    AND DATE(fd.timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
),
downloads_agg AS (
  SELECT
    pkg_name,
    COUNTIF(
      fd.timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
    ) AS n_downloads_7d,
    COUNT(*) AS n_downloads_30d
  FROM downloads_raw AS fd
  GROUP BY pkg_name
),

-- I.n_dependents_est via reverse dependencies
requires_flat AS (
  SELECT
    dm.name AS dependent_pkg,
    REGEXP_EXTRACT(req, r'^([A-Za-z0-9_.-]+)') AS required_pkg
  FROM labeled_dist AS dm,
       UNNEST(dm.requires_dist) AS req
),
dependents_est AS (
  SELECT
    rf.required_pkg AS pkg_name,
    COUNT(DISTINCT rf.dependent_pkg) AS n_dependents_est
  FROM requires_flat AS rf
  WHERE rf.required_pkg IS NOT NULL
  GROUP BY rf.required_pkg
),

-- ------------------------------------------------------------------
-- G. Latest Distribution Features (partial; only n_distributions)
-- ------------------------------------------------------------------
latest_version_per_pkg AS (
  SELECT
    dm.name AS pkg_name,
    ARRAY_FIRST(
      ARRAY_AGG(dm.version ORDER BY dm.upload_time DESC LIMIT 1)
    ) AS latest_version
  FROM labeled_dist AS dm
  GROUP BY dm.name
),
distributions_latest AS (
  SELECT
    lvp.pkg_name,
    lvp.latest_version,
    COUNT(*) AS n_distributions
    -- If `size` exists in distribution_metadata, you can also add:
    -- , SUM(dm.size) AS n_total_bytes
  FROM labeled_dist AS dm
  JOIN latest_version_per_pkg AS lvp
    ON dm.name = lvp.pkg_name
   AND dm.version = lvp.latest_version
  GROUP BY lvp.pkg_name, lvp.latest_version
),

-- ------------------------------------------------------------------
-- FINAL SELECT: join everything by package name
-- ------------------------------------------------------------------
final AS (
  SELECT
    am.name AS pkg_name,

    -- ========== A. Package Identity ==========
    nf.n_name_len,
    nf.has_digit_in_name,
    nf.has_dash_or_underscore,
    nf.cat_name_case,

    -- ========== B. Summary & Description ==========
    tf.n_summary_len,
    tf.n_desc_len,
    tf.n_desc_lines,
    tf.has_code_block_in_desc,
    tf.n_urls_in_desc,
    tf.has_suspicious_kw,
    tf.pct_non_ascii_desc,

    -- ========== C. Release & Activity ==========
    ar.t_age_first_release_days,
    ar.t_age_last_release_days,
    ar.n_versions,
    ar.t_median_release_gap_days,
    ar.has_single_release,
    ar.t_time_of_day_bucket,
    ar.cat_weekday_of_last_release,

    -- ========== D. Ownership & Maintainer ==========
    mf.n_maintainers,
    mf.pct_free_email_domains,
    mf.has_disposable_email,
    mf.has_missing_author,

    -- ========== E. Repository & Link ==========
    ual.has_homepage,
    ual.has_repo_url,
    ual.cat_repo_host,
    ual.has_issue_tracker,
    ual.has_docs_url,

    -- ========== F. License & Classifier ==========
    ual.has_license,
    ual.cat_license_family,
    ual.n_classifiers,
    ual.has_prog_lang_classifier,
    ual.has_typing_classifier,

    -- ========== G. Latest distribution (partial) ==========
    dl.n_distributions,

    -- ========== H. Dependencies ==========
    df.n_requires,
    df.has_extras,

    -- ========== I. Popularity ==========
    da.n_downloads_7d,
    da.n_downloads_30d,
    de.n_dependents_est,

    -- ========== J. Simple rule-based features (SQL-side) ==========
    IF(
      ual.has_repo_url = 0 AND tf.n_desc_len < 200,
      1, 0
    ) AS rule_no_repo_low_desc_len,

    IF(
      tf.has_suspicious_kw = TRUE,
      1, 0
    ) AS rule_suspicious_name_and_dep,

    -- ========== Raw fields for offline additions ==========
    am.licenses,
    am.versions,
    am.first_upload AS t_first_release,
    am.last_upload  AS t_last_release,
    am.latest_summary,
    am.latest_description,
    am.latest_project_urls,
    am.distinct_authors,
    am.distinct_maintainers,
    am.distinct_keywords,
    am.distinct_classifiers,
    am.latest_dependencies,


  FROM all_metadata AS am
  JOIN agg_releases_with_gap AS ar
    ON am.name = ar.pkg_name
  JOIN urls_and_licenses AS ual
    ON am.name = ual.name

  LEFT JOIN name_features       AS nf ON am.name = nf.pkg_name
  LEFT JOIN text_features       AS tf ON am.name = tf.pkg_name
  LEFT JOIN maintainer_features AS mf ON am.name = mf.pkg_name
  LEFT JOIN dependency_features AS df ON am.name = df.pkg_name
  LEFT JOIN downloads_agg       AS da ON am.name = da.pkg_name
  LEFT JOIN dependents_est      AS de ON am.name = de.pkg_name
  LEFT JOIN distributions_latest AS dl ON am.name = dl.pkg_name
)

SELECT *
FROM final
WHERE pkg_name = @pkg_name;