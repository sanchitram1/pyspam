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
JOIN sample ON sample.package_name = latest.name
WHERE rn = 1
;