SELECT
    p.total_shares,
    p.cost_basis,
    u.cash_balance
FROM users u
LEFT JOIN portfolio p
    ON u.id = p.user_id AND p.ticker = %s
WHERE u.id = %s;