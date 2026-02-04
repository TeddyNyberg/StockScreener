SELECT 1
FROM watchlist
WHERE user_id = %s
  AND ticker = %s
LIMIT 1;