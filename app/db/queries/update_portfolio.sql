UPDATE portfolio
SET total_shares = %s, cost_basis = %s
WHERE ticker = %s;