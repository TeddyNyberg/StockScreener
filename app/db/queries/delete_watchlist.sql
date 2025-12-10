DELETE FROM watchlist
WHERE user_id = %s AND ticker = %s;