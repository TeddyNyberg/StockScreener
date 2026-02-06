import psycopg2

from backend.app.settings import DB_NAME, DB_USERNAME, DB_PASSWORD, DB_HOST, DB_PORT


class DB:
    def __init__(self):
        self.conn = None
    def __enter__(self):
        try:
            self.conn = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USERNAME,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT
            )
            return self.conn
        except psycopg2.Error as e:
            print(f"Could not establish a database connection.")
            raise
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
                print("rollback")
            self.conn.close()
        return False
