SELECT total_shares, cost_basis
FROM portfolio WHERE ticker = %s FOR UPDATE