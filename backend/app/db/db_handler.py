from backend.app.db.database import DB
from backend.app.schemas import TokenData
from backend.app.settings import *
import psycopg2
from decimal import Decimal
import bcrypt
import os
import jwt
from jwt.exceptions import InvalidTokenError
from fastapi import Depends, HTTPException, status
from datetime import datetime, timedelta, timezone
from typing import Annotated
from fastapi.security import OAuth2PasswordBearer

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def _load_sql(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    query_path = os.path.join(current_dir, "queries", filename)
    try:
        with open(query_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        print(f"File {filename} not found. Looked for: {query_path}")
        return ""

def get_watchlist(user_id):
    GET_WATCHLIST_QUERY = _load_sql("select_watchlist.sql")

    try:
        with DB() as conn:
            with conn.cursor() as cur:
                cur.execute(GET_WATCHLIST_QUERY, (user_id,))
                return cur.fetchall()
    except Exception as e:
        print(f"Database error during get_watchlist: {e}")
        return []


def add_watchlist(ticker, user_id):
    INSERT_TICKER_QUERY = _load_sql("insert_watchlist.sql")
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
    DELETE_TICKER_QUERY = _load_sql("delete_watchlist.sql")
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
    GET_PORTFOLIO_QUERY = _load_sql("select_portfolio.sql")
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
    UPSERT_QUERY = _load_sql("upsert_portfolio.sql")
    price = Decimal(str(price))
    to_buy = Decimal(to_buy)
    with conn.cursor() as cur:
        cur.execute(UPSERT_QUERY, (user_id, ticker, to_buy, to_buy*price))
        print(f"Successfully updated portfolio for {ticker}.")


def _log_transaction(conn, ticker, quantity, price, transaction_type, user_id):
    INSERT_TRANSACTION_QUERY = _load_sql("insert_transactions.sql")
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
    SELECT_FOR_UPDATE_SQL = _load_sql("select_shares_for_update.sql")
    DELETE_PORTFOLIO_SQL = _load_sql("delete_portfolio.sql")
    UPDATE_PORTFOLIO_SQL = _load_sql("update_portfolio.sql")

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


def init_db(conn):
    create_watchlist_sql = _load_sql("create_watchlist.sql")
    create_portfolio_sql = _load_sql("create_portfolio.sql")
    create_transactions_sql = _load_sql("create_transactions.sql")

    with conn.cursor() as cur:
        cur.execute(create_watchlist_sql)
        cur.execute(create_portfolio_sql)
        cur.execute(create_transactions_sql)

    print("Database initialized.")


def init_user_table(conn):
    create_users_sql = _load_sql("create_users_table.sql")

    with conn.cursor() as cur:
        cur.execute(create_users_sql)
        print("Users table initialized.")


def register_user(username, plain_password):

    INSERT_USER_QUERY = _load_sql("insert_user.sql")

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


def get_user_id(username):
    GET_USER_QUERY = _load_sql("select_id_by_username.sql")
    try:
        with DB() as conn:
            with conn.cursor() as cur:
                cur.execute(GET_USER_QUERY, (username,))
                result = cur.fetchone()

                if result is None:
                    print("Auth failed: User not found.")
                    return -1

                user_id = result

                return user_id
    except Exception as e:
        print(f"Database error during authentication: {e}")
        return -1


def authenticate_user(username, plain_password):
    GET_USER_QUERY = _load_sql("select_hashed_password_by_username.sql")
    try:
        with DB() as conn:
            with conn.cursor() as cur:
                cur.execute(GET_USER_QUERY, (username,))
                result = cur.fetchone()

                if result is None:
                    print("Auth failed: User not found.")
                    return -1

                user_id, db_password_hash = result

                if bcrypt.checkpw(plain_password.encode('utf-8'), db_password_hash.encode('utf-8')):
                    print(f"User '{username}' logged in successfully.")
                    return user_id
                else:
                    print("Auth failed: Invalid password.")
                    return -1
    except Exception as e:
        print(f"Database error during authentication: {e}")
        return -1


#TODO: encrypt balance?
def _fetch_balance(conn, user_id):
    GET_BALANCE_QUERY = _load_sql("get_balance.sql")

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
    UPDATE_SQL = _load_sql("update_balance.sql")
    with conn.cursor() as cur:
        cur.execute(UPDATE_SQL, (amount, user_id))


# you can create specific tokens, which have unique permissions
# watch out for user foo, car foo, blog foo
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, TOKEN_SCR_KEY, algorithm=ALGORITHM)
    return encoded_jwt



async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, TOKEN_SCR_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except InvalidTokenError:
        raise credentials_exception
    user_id = get_user_id(token_data.username)
    if user_id is None or user_id == -1:
        raise credentials_exception
    return user_id

def check_if_in_watchlist(user_id, ticker):
    CHECK_WL_SQL = _load_sql("check_in_watchlist.sql")
    with DB() as conn:
        with conn.cursor() as cur:
            cur.execute(CHECK_WL_SQL, (user_id, ticker))
            result = cur.fetchone()
            if result is not None:
                return True
    return False

def get_ticker_stats_owned(user_id, ticker):
    TICKER_STATS = _load_sql("select_owned_ticker_stats.sql")
    with DB() as conn:
        with conn.cursor() as cur:
            cur.execute(TICKER_STATS, (user_id, ticker))
            result = cur.fetchone()
            return result
    return []


def get_ticker_stats_owned(user_id, ticker):
    TICKER_STATS = _load_sql("select_owned_ticker_stats.sql")
    with DB() as conn:
        with conn.cursor() as cur:
            cur.execute(TICKER_STATS, (ticker, user_id))
            result = cur.fetchone()
            if result:
                return {
                    "shares_owned": result[0] or 0,
                    "cost_basis": float(result[1] or 0),
                    "cash_balance": float(result[2] or 0)
                }
    return {"shares_owned": 0, "cost_basis": 0.0, "cash_balance": 0.0}


def rm_watchlist(ticker, user_id):
    DELETE_TICKER_QUERY = _load_sql("delete_watchlist.sql")
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

