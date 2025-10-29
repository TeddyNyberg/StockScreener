from settings import *
import psycopg2
from decimal import Decimal

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


def rm_watchlist(ticker):

    conn = get_db_connection()  # Assuming this function is defined elsewhere
    if conn is None:
        print("Could not establish a database connection.")
        return

    try:
        with conn.cursor() as cur:
            create_table_query = """
                                 CREATE TABLE IF NOT EXISTS watchlist
                                 (
                                     ticker VARCHAR(10) PRIMARY KEY
                                 );
                                 """
            cur.execute(create_table_query)

            delete_ticker_query = """
                                  DELETE FROM watchlist
                                  WHERE ticker = %s;
                                  """
            cur.execute(delete_ticker_query, (ticker,))

            rows_deleted = cur.rowcount

            conn.commit()

            if rows_deleted > 0:
                print(f"Successfully removed ticker: {ticker} from watchlist.")
            else:
                print(f"Ticker: {ticker} was not found in the watchlist (0 rows affected).")

    except psycopg2.Error as e:
        print(f"Database error during rm_watchlist: {e}")
        conn.rollback()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()

def get_portfolio():
    conn = get_db_connection()
    if conn is None:
        print("Could not establish a database connection.")
        return None
    try:
        with conn.cursor() as cur:
            get_table_query = """
                              SELECT * FROM portfolio
                              """
            cur.execute(get_table_query)
            portfolio = cur.fetchall()
    except psycopg2.Error as e:
        print(f"Database error during get_portfolio: {e}")
        portfolio = []
    finally:
        if conn:
            conn.close()
    return portfolio



def buy_stock(ticker, quantity, price):
    conn = get_db_connection()
    if conn is None:
        print("Could not establish a database connection.")
        return
    try:
        add_to_portfolio(conn, ticker, quantity, price)
        buy_transaction(conn, ticker, quantity, price)
    except psycopg2.Error as e:
        print(f"Database error during buy_stock: {e}")
        conn.rollback()

    finally:
        if conn:
            conn.close()

def add_to_portfolio(conn, ticker, to_buy, price):
    price = Decimal(str(price))
    to_buy = Decimal(to_buy)

    with conn.cursor() as cur:
        create_table_query = """
                             CREATE TABLE IF NOT EXISTS portfolio ( 
                             ticker VARCHAR(10) PRIMARY KEY,
                             total_shares INTEGER NOT NULL,
                             cost_basis NUMERIC(12, 4) NOT NULL
                             ); 
                             """
        cur.execute(create_table_query)

        upsert_ticker_query = """
                              INSERT INTO portfolio (ticker, total_shares, cost_basis)
                              VALUES (%s, %s, %s)
                              ON CONFLICT (ticker) DO UPDATE
                              SET 
                                total_shares = portfolio.total_shares + EXCLUDED.total_shares,
                                cost_basis = portfolio.cost_basis + EXCLUDED.cost_basis;
                              """
        cur.execute(upsert_ticker_query, (ticker, to_buy, to_buy*price))
        conn.commit()
        print(f"Successfully processed ticker: {ticker}. Table 'portfolio' ensured to exist.")

def buy_transaction(conn, ticker, quantity, price):
    with conn.cursor() as cur:
        create_table_query = """
                             CREATE TABLE IF NOT EXISTS transactions ( 
                             id SERIAL PRIMARY KEY,
                             ticker VARCHAR(10) NOT NULL,
                             quantity INTEGER NOT NULL,
                             price NUMERIC(10, 2) NOT NULL,
                             type VARCHAR(4) NOT NULL 
                             ); 
                             """
        cur.execute(create_table_query)
        insert_ticker_query = """
                              INSERT INTO transactions (ticker, quantity, price, type) 
                              VALUES (%s, %s, %s, 'BUY'); 
                              """
        cur.execute(insert_ticker_query, (ticker, quantity, price))
        conn.commit()
        print(f"Successfully logged BUY transaction for {ticker}.")

def sell_stock(ticker, quantity, price):
    conn = get_db_connection()
    if conn is None:
        print("Could not establish a database connection.")
        return
    try:
        rm_from_portfolio(conn, ticker, quantity, price)
        sell_transaction(conn, ticker, quantity, price)
    except psycopg2.Error as e:
        print(f"Database error during buy_stock: {e}")
        conn.rollback()
    finally:
        if conn:
            conn.close()


def rm_from_portfolio(conn, ticker, to_sell, price):
    with conn.cursor() as cur:
        create_table_query = """
                             CREATE TABLE IF NOT EXISTS portfolio ( 
                             ticker VARCHAR(10) PRIMARY KEY,
                             total_shares INTEGER NOT NULL,
                             cost_basis NUMERIC(12, 4) NOT NULL
                             ); 
                             """
        cur.execute(create_table_query)

        cur.execute("SELECT total_shares, cost_basis FROM portfolio WHERE ticker = %s FOR UPDATE", (ticker,))
        result = cur.fetchone()

        if result is None:
            print(f"Error: Cannot sell {to_sell} shares of {ticker}. Ticker not found in portfolio.")
            conn.rollback()
            return False

        current_shares = result[0]
        current_cost_basis = result[1]

        print("current_shares: ", current_shares)
        print("current_cost_basis: ", current_cost_basis)

        if current_shares < to_sell:
            print(f"Error: Insufficient shares to sell {to_sell} of {ticker}. Current shares: {current_shares}.")
            conn.rollback()
            return False

        avg_cost_per_share = current_cost_basis / Decimal(current_shares) if current_shares > 0 else 0
        to_sell = Decimal(to_sell)
        cost_basis_reduction = to_sell * avg_cost_per_share

        new_shares = current_shares - to_sell
        new_cost_basis = current_cost_basis - cost_basis_reduction

        if new_shares <= 0:
            delete_query = "DELETE FROM portfolio WHERE ticker = %s;"
            cur.execute(delete_query, (ticker,))
            print(f"Successfully sold all shares of {ticker}. Removed from portfolio.")
        else:
            update_query = """
                           UPDATE portfolio 
                           SET total_shares = %s, cost_basis = %s
                           WHERE ticker = %s;
                           """
            cur.execute(update_query, (new_shares, new_cost_basis, ticker))
            print(f"Successfully sold {to_sell} shares of {ticker}. Portfolio updated.")

        conn.commit()
        return True


def sell_transaction(conn, ticker, quantity, price):
    with conn.cursor() as cur:
        create_table_query = """
                             CREATE TABLE IF NOT EXISTS transactions ( 
                             id SERIAL PRIMARY KEY,
                             ticker VARCHAR(10) NOT NULL,
                             quantity INTEGER NOT NULL,
                             price NUMERIC(10, 2) NOT NULL,
                             type VARCHAR(4) NOT NULL
                             ); 
                             """
        cur.execute(create_table_query)

        insert_ticker_query = """
                              INSERT INTO transactions (ticker, quantity, price, type) 
                              VALUES (%s, %s, %s, 'SELL'); 
                              """
        cur.execute(insert_ticker_query, (ticker, quantity, price))
        conn.commit()
        print(f"Successfully logged SELL transaction for {ticker}.")
