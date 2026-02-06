from backend.app.db.database import DB


def force_delete_user(username):
    with DB() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE username = %s", (username,))
