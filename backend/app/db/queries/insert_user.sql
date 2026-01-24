INSERT INTO users (username, password_hash)
VALUES (%s, %s)
RETURNING id;