from settings import *
import psycopg2

def get_db_connection():
    # Establishes and returns a connection to the PostgreSQL database.
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USERNAME,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to the database: {e}")
        return None


def get_watchlist():
    conn = get_db_connection()
    if conn is None:
        print("Could not establish a database connection.")
        return None
    try:
        with conn.cursor() as cur:
            get_table_query = """
                              SELECT * FROM watchlist
                              """
            cur.execute(get_table_query)
            watchlist = cur.fetchall()
    except psycopg2.Error as e:
        print(f"Database error during get_watchlist: {e}")
        watchlist = []
    finally:
        if conn:
            conn.close()
    return watchlist


def add_watchlist(ticker):
    conn = get_db_connection()
    if conn is None:
        print("Could not establish a database connection.")
        return
    try:
        with conn.cursor() as cur:
            create_table_query = """
                                 CREATE TABLE IF NOT EXISTS watchlist ( 
                                     ticker VARCHAR(10) PRIMARY KEY
                                 ); 
                                 """
            cur.execute(create_table_query)

            insert_ticker_query = """
                                  INSERT INTO watchlist (ticker) 
                                  VALUES (%s) ON CONFLICT (ticker) DO NOTHING; 
                                  """
            cur.execute(insert_ticker_query, (ticker,))
            conn.commit()
            print(f"Successfully processed ticker: {ticker}. Table 'watchlist' ensured to exist.")

    except psycopg2.Error as e:
        print(f"Database error during add_watchlist: {e}")
        conn.rollback()
    finally:
        if conn:
            conn.close()