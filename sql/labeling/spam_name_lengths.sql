INSERT INTO `pyspam.ground_truth` (package_name, is_spam)
--SELECT name_length, COUNT(package_name) FROM (
SELECT DISTINCT
  name as package_name,
  1 as is_spam ,
  version,
  summary,
  -- Check the length of the name
  LENGTH(name) as name_length,
  author,
  upload_time
FROM `bigquery-public-data.pypi.distribution_metadata`
WHERE
  -- 1. LENGTH FILTER: Legitimate packages rarely exceed 30-40 chars
  LENGTH(name) > 40
  
  -- 2. RECENCY FILTER: Focus on recent stuff (optional, but likely where the spam is)
  AND upload_time > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 365 DAY)
  
  -- 3. EXCLUDE KNOWN GOOD (Optional safety)
  -- If you have a known good list, exclude them here. 
  -- But usually, long names are safe to assume "suspect" for manual review.

QUALIFY ROW_NUMBER() OVER (PARTITION BY name ORDER BY upload_time DESC) = 1
ORDER BY LENGTH(package_name) DESC
-- LIMIT 5000
-- OFFSET 1000

-- )
-- group by name_length 
-- order by 1 desc 