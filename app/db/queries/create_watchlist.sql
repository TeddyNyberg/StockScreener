CREATE TABLE IF NOT EXISTS watchlist (
    user_id INTEGER REFERENCES users(id),
    ticker VARCHAR(10),
    PRIMARY KEY (user_id, ticker)
);