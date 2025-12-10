INSERT INTO portfolio (user_id, ticker, total_shares, cost_basis)
VALUES (%s, %s, %s, %s)
ON CONFLICT (user_id, ticker) DO UPDATE
SET
  total_shares = portfolio.total_shares + EXCLUDED.total_shares,
  cost_basis = portfolio.cost_basis + EXCLUDED.cost_basis;