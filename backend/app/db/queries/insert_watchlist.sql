INSERT INTO watchlist (user_id, ticker)
VALUES (%s, %s) ON CONFLICT (user_id, ticker) DO NOTHING;