INSERT INTO watchlist (ticker)
VALUES (%s) ON CONFLICT (ticker) DO NOTHING;