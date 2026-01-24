UPDATE portfolio
SET total_shares = %s, cost_basis = %s
WHERE user_id = %s AND ticker = %s;