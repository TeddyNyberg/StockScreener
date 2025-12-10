CREATE TABLE IF NOT EXISTS portfolio (
    user_id INTEGER REFERENCES users(id),
    ticker VARCHAR(10),
    total_shares INTEGER NOT NULL,
    cost_basis NUMERIC(12, 4) NOT NULL,
    PRIMARY KEY (user_id)
);