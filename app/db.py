from settings import *
import psycopg2
from decimal import Decimal

def load_sql(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    query_path = os.path.join(current_dir, "queries", filename)
    try:
        with open(query_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        print(f"File {filename} not found. Looked for: {query_path}")
        return ""

def get_watchlist():
    GET_WATCHLIST_QUERY = load_sql("select_watchlist.sql")
    try:
        with DB() as conn:
            with conn.cursor() as cur:
                cur.execute(GET_WATCHLIST_QUERY)
                return cur.fetchall()
    except Exception as e:
        print(f"Database error during get_watchlist: {e}")
        return []


def add_watchlist(ticker):
    INSERT_TICKER_QUERY = load_sql("insert_watchlist.sql")
    try:
        with DB() as conn:
            with conn.cursor() as cur:
                cur.execute(INSERT_TICKER_QUERY, (ticker,))
                print(f"Successfully processed ticker: {ticker}.")
                return True
    except Exception as e:
        print(f"Database error during add_watchlist (Rollback): {e}")
        return False


def rm_watchlist(ticker):
    DELETE_TICKER_QUERY = load_sql("delete_watchlist.sql")
    try:
        with DB() as conn:
            with conn.cursor() as cur:
                cur.execute(DELETE_TICKER_QUERY, (ticker,))
                rows_deleted = cur.rowcount
                if rows_deleted > 0:
                    print(f"Successfully removed ticker: {ticker} from watchlist.")
                else:
                    print(f"Ticker: {ticker} was not found in the watchlist (0 rows affected).")
            return True
    except Exception as e:
        print(f"Database error during rm_watchlist (Rollback): {e}")
        return False

def get_portfolio():
    GET_PORTFOLIO_QUERY = load_sql("select_portfolio.sql")
    try:
        with DB() as conn:
            with conn.cursor() as cur:
                cur.execute(GET_PORTFOLIO_QUERY)
                return cur.fetchall()
    except Exception as e:
        print(f"Database error during get_portfolio: {e}")
        return []


def buy_stock(ticker, quantity, price):
    try:
        with DB() as conn:
            _add_to_portfolio(conn, ticker, quantity, price)
            _log_transaction(conn, ticker, quantity, price, 'BUY')
    except Exception as e:
        print(f"Error executing buy_stock (transaction failed): {e}")

def _add_to_portfolio(conn, ticker, to_buy, price):
    UPSERT_QUERY = load_sql("upsert_portfolio.sql")
    price = Decimal(str(price))
    to_buy = Decimal(to_buy)
    with conn.cursor() as cur:
        cur.execute(UPSERT_QUERY, (ticker, to_buy, to_buy*price))
        print(f"Successfully updated portfolio for {ticker}.")


def _log_transaction(conn, ticker, quantity, price, type):
    INSERT_TRANSACTION_QUERY = load_sql("insert_transactions.sql")
    with conn.cursor() as cur:
        cur.execute(INSERT_TRANSACTION_QUERY, (ticker, quantity, price, type))
        print(f"Successfully logged {type} transaction for {ticker}.")


def sell_stock(ticker, quantity, price):
    try:
        with DB() as conn:
            _rm_from_portfolio(conn, ticker, quantity, price)
            _log_transaction(conn, ticker, quantity, price, 'SELL')
    except Exception as e:
        print(f"Error executing sell_stock (transaction failed): {e}")


def _rm_from_portfolio(conn, ticker, to_sell, price):
    SELECT_FOR_UPDATE_SQL = load_sql("select_shares_for_update.sql")
    DELETE_PORTFOLIO_SQL = load_sql("delete_portfolio.sql")
    UPDATE_PORTFOLIO_SQL = load_sql("update_portfolio.sql")

    with conn.cursor() as cur:
        cur.execute(SELECT_FOR_UPDATE_SQL, (ticker,))
        result = cur.fetchone()

        if result is None:
            raise ValueError(f"Error: Cannot sell {to_sell} shares of {ticker}. Ticker not found in portfolio.")

        current_shares = result[0]
        current_cost_basis = result[1]
        to_sell = Decimal(to_sell)

        if current_shares < to_sell:
            raise ValueError(
                f"Error: Insufficient shares to sell {to_sell} of {ticker}. Current shares: {current_shares}.")

        avg_cost_per_share = current_cost_basis / Decimal(current_shares) if current_shares > 0 else 0
        cost_basis_reduction = to_sell * avg_cost_per_share
        new_shares = current_shares - to_sell
        new_cost_basis = current_cost_basis - cost_basis_reduction

        if new_shares <= 0:
            cur.execute(DELETE_PORTFOLIO_SQL, (ticker,))
            print(f"Successfully sold all shares of {ticker}. Removed from portfolio.")
        else:
            cur.execute(UPDATE_PORTFOLIO_SQL, (new_shares, new_cost_basis, ticker))
            print(f"Successfully sold {to_sell} shares of {ticker}. Portfolio updated.")

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

def init_db(conn):
    create_watchlist_sql = load_sql("create_watchlist.sql")
    create_portfolio_sql = load_sql("create_portfolio.sql")
    create_transactions_sql = load_sql("create_transactions.sql")

    with conn.cursor() as cur:
        cur.execute(create_watchlist_sql)
        cur.execute(create_portfolio_sql)
        cur.execute(create_transactions_sql)

    print("Database initialized.")
