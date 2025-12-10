SELECT total_shares, cost_basis
FROM portfolio WHERE user_id = %s AND ticker = %s FOR UPDATE;