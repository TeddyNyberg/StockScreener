from settings import *
import psycopg2
from decimal import Decimal
import bcrypt

def load_sql(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    query_path = os.path.join(current_dir, "queries", filename)
    try:
        with open(query_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        print(f"File {filename} not found. Looked for: {query_path}")
        return ""

def get_watchlist(user_id):
    GET_WATCHLIST_QUERY = load_sql("select_watchlist.sql")
    try:
        with DB() as conn:
            with conn.cursor() as cur:
                cur.execute(GET_WATCHLIST_QUERY, (user_id,))
                return cur.fetchall()
    except Exception as e:
        print(f"Database error during get_watchlist: {e}")
        return []


def add_watchlist(ticker, user_id):
    INSERT_TICKER_QUERY = load_sql("insert_watchlist.sql")
    try:
        with DB() as conn:
            with conn.cursor() as cur:
                cur.execute(INSERT_TICKER_QUERY, (user_id, ticker))
                print(f"Successfully processed ticker: {ticker} for user: {user_id}.")
                return True
    except Exception as e:
        print(f"Database error during add_watchlist (Rollback): {e}")
        return False


def rm_watchlist(ticker, user_id):
    DELETE_TICKER_QUERY = load_sql("delete_watchlist.sql")
    try:
        with DB() as conn:
            with conn.cursor() as cur:
                cur.execute(DELETE_TICKER_QUERY, (user_id, ticker))
                rows_deleted = cur.rowcount
                if rows_deleted > 0:
                    print(f"Successfully removed ticker: {ticker} from watchlist.")
                else:
                    print(f"Ticker: {ticker} was not found in the watchlist (0 rows affected).")
            return True
    except Exception as e:
        print(f"Database error during rm_watchlist (Rollback): {e}")
        return False

def get_portfolio(user_id):
    GET_PORTFOLIO_QUERY = load_sql("select_portfolio.sql")
    try:
        with DB() as conn:
            with conn.cursor() as cur:
                cur.execute(GET_PORTFOLIO_QUERY, (user_id,))
                return cur.fetchall()
    except Exception as e:
        print(f"Database error during get_portfolio: {e}")
        return []


def buy_stock(ticker, quantity, price, user_id):
    try:
        with DB() as conn:
            if _fetch_balance(conn, user_id) > quantity*price:
                _add_to_portfolio(conn, ticker, quantity, price, user_id)
                _log_transaction(conn, ticker, quantity, price, 'BUY', user_id)
                _update_user_balance(conn, user_id, -(price * quantity))
            else:
                print("ya broke")
    except Exception as e:
        print(f"Error executing buy_stock (transaction failed): {e}")

def _add_to_portfolio(conn, ticker, to_buy, price, user_id):
    UPSERT_QUERY = load_sql("upsert_portfolio.sql")
    price = Decimal(str(price))
    to_buy = Decimal(to_buy)
    with conn.cursor() as cur:
        cur.execute(UPSERT_QUERY, (user_id, ticker, to_buy, to_buy*price))
        print(f"Successfully updated portfolio for {ticker}.")


def _log_transaction(conn, ticker, quantity, price, transaction_type, user_id):
    INSERT_TRANSACTION_QUERY = load_sql("insert_transactions.sql")
    with conn.cursor() as cur:
        cur.execute(INSERT_TRANSACTION_QUERY, (user_id, ticker, quantity, price, transaction_type))
        print(f"Successfully logged {transaction_type} transaction for {ticker}.")


def sell_stock(ticker, quantity, price, user_id):
    try:
        with DB() as conn:
            _rm_from_portfolio(conn, ticker, quantity, price, user_id)
            _log_transaction(conn, ticker, quantity, price, 'SELL', user_id)
            _update_user_balance(conn, user_id, (price * quantity))
    except Exception as e:
        print(f"Error executing sell_stock (transaction failed): {e}")


def _rm_from_portfolio(conn, ticker, to_sell, price, user_id):
    SELECT_FOR_UPDATE_SQL = load_sql("select_shares_for_update.sql")
    DELETE_PORTFOLIO_SQL = load_sql("delete_portfolio.sql")
    UPDATE_PORTFOLIO_SQL = load_sql("update_portfolio.sql")

    with conn.cursor() as cur:
        cur.execute(SELECT_FOR_UPDATE_SQL, (user_id, ticker))
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
            cur.execute(DELETE_PORTFOLIO_SQL, (user_id, ticker))
            print(f"Successfully sold all shares of {ticker}. Removed from portfolio.")
        else:
            cur.execute(UPDATE_PORTFOLIO_SQL, (new_shares, new_cost_basis, user_id, ticker))
            print(f"Successfully sold {to_sell} shares of {ticker}. Portfolio updated.")

        return True


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


def init_user_table(conn):
    create_users_sql = load_sql("create_users_table.sql")

    with conn.cursor() as cur:
        cur.execute(create_users_sql)
        print("Users table initialized.")


def register_user(username, plain_password):

    INSERT_USER_QUERY = load_sql("insert_user.sql")

    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(plain_password.encode('utf-8'), salt)

    try:
        with DB() as conn:
            with conn.cursor() as cur:
                cur.execute(INSERT_USER_QUERY, (username, hashed_password.decode('utf-8')))
                user_id = cur.fetchone()[0]
                print(f"User '{username}' registered successfully.")
                return user_id
    except psycopg2.IntegrityError:
        print(f"Registration failed: Username '{username}' already exists.")
        return False
    except Exception as e:
        print(f"Database error during registration: {e}")
        return False


def authenticate_user(username, plain_password):

    GET_USER_QUERY = load_sql("select_user_by_username.sql")

    try:
        with DB() as conn:
            with conn.cursor() as cur:
                cur.execute(GET_USER_QUERY, (username,))
                result = cur.fetchone()

                if result is None:
                    print("Auth failed: User not found.")
                    return False

                user_id, db_username, db_password_hash = result

                if bcrypt.checkpw(plain_password.encode('utf-8'), db_password_hash.encode('utf-8')):
                    print(f"User '{username}' logged in successfully.")
                    return user_id
                else:
                    print("Auth failed: Invalid password.")
                    return False
    except Exception as e:
        print(f"Database error during authentication: {e}")
        return False

def _fetch_balance(conn, user_id):
    GET_BALANCE_QUERY = load_sql("get_balance.sql")

    with conn.cursor() as cur:
        cur.execute(GET_BALANCE_QUERY, (user_id,))
        result = cur.fetchone()
        return result[0] if result else -1

def get_balance(user_id):
    try:
        with DB() as conn:
            return _fetch_balance(conn, user_id)
    except Exception:
        return 0


def _update_user_balance(conn, user_id, amount):
    UPDATE_SQL = load_sql("update_balance.sql")
    with conn.cursor() as cur:
        cur.execute(UPDATE_SQL, (amount, user_id))