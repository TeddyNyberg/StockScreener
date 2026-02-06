from fastapi import Query
from contextlib import asynccontextmanager
from backend.app.db.db_handler import *
from backend.app.data.data_cache import get_yfdata_cache
from backend.app.data.yfinance_fetcher import get_info, get_financial_metrics, get_balancesheet
from fastapi.middleware.cors import CORSMiddleware

from backend.app.ml_logic.strategy import calculate_kelly_allocations
from backend.app.services.model_service import ModelService

# to run, uvicorn backend.main:app --reload

model_service = ModelService()

@asynccontextmanager
async def lifespan(_: FastAPI):
    print("Connecting to Database...")
    try:
        with DB() as conn:
            init_user_table(conn)
            init_db(conn)
    except Exception as e:
        print(f"Database Init Failed: {e}")
    try:
        await model_service.initialize()
        print("Model Service Initialized.")
    except Exception as e:
        print(f"Model Service Init Failed: {e}")
    yield
    print("Server Shutting Down...")
    await model_service.shutdown()

app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class WatchlistRequest(BaseModel):
    ticker: str

class StockTransaction(BaseModel):
    ticker: str
    price: float
    quantity: int
    type: str


class LoginCredentials(BaseModel):
    username: str
    password: str


# TODO: maybe a little ocupled? just assumes that data comes back in same order is all
@app.get("/watchlist")
def api_get_watchlist(user_id: int = Depends(get_current_user)):
    watchlist = get_watchlist(user_id)
    ticker_list = [row[0] for row in watchlist]
    data = get_yfdata_cache(ticker_list, "5D", normalize=False)
    response = []

    for df, ticker in zip(data, ticker_list):
        response.append({
            "ticker": ticker,
            "price": df["Close"].iloc[-1],
            "change": df["Close"].iloc[-1] - df["Close"].iloc[-2],
            "change_percent": 100 * ((df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2]),
        })
    return response


@app.post("/watchlist/add")
def api_add_watchlist(request: WatchlistRequest, user_id: int = Depends(get_current_user)):
    success = add_watchlist(request.ticker, user_id)
    if success:
        return {"status": "success", "message": f"Added {request.ticker}"}
    else:
        raise HTTPException(status_code=400, detail="Could not add ticker")

@app.post("/watchlist/remove")
def api_rm_watchlist(request: WatchlistRequest, user_id: int = Depends(get_current_user)):
    success = rm_watchlist(request.ticker, user_id)
    if success:
        return {"status": "success", "message": f"Added {request.ticker}"}
    else:
        raise HTTPException(status_code=400, detail="Could not add ticker")

@app.get("/watchlist/check/{ticker}")
def check_watchlist_status(ticker: str, user_id: int = Depends(get_current_user)):
    exists = check_if_in_watchlist(user_id, ticker)
    print(ticker, exists)
    return {"in_watchlist": exists}


@app.get("/portfolio")
def api_get_portfolio(user_id: int = Depends(get_current_user)):
    portfolio_data = get_portfolio(user_id)
    cash_balance = get_balance(user_id)

    response = []

    response.append({
        "ticker": "Cash",
        "price": 1.0,
        "shares": float(cash_balance),
        "basis": float(cash_balance),
        "avg": 1.0,
        "pl": 0.0,
        "pct": 0.0,
        "market_value": float(cash_balance),
    })


    tickers = [row[0] for row in portfolio_data]
    cur_prices = get_yfdata_cache(tickers, "5D", normalize=False)

    for i, row in enumerate(portfolio_data):
        ticker = row[0]
        shares = int(row[1])
        total_basis = Decimal(str(row[2]))

        cur_price = Decimal(str(cur_prices[i].iloc[-1]["Close"]))

        if shares > 0:
            avg_cost = total_basis / shares
            market_value = cur_price * shares
            profit_loss = market_value - total_basis
            percent_return = (profit_loss / total_basis) * 100 if total_basis != 0 else 0
        else:
            avg_cost = 0
            profit_loss = 0
            percent_return = 0
            market_value = 0

        response.append({
            "ticker": ticker,
            "price": float(cur_price),
            "shares": shares,
            "basis": float(total_basis),
            "avg": float(avg_cost),
            "pl": float(profit_loss),
            "pct": float(percent_return),
            "market_value": float(market_value),
        })
    return response

@app.post("/stock-transaction")
def api_transact_stock(request: StockTransaction, user_id: int = Depends(get_current_user)):
    if request.type == "BUY":
        buy_stock(request.ticker, request.quantity, request.price, user_id)
    if request.type == "SELL":
        sell_stock(request.ticker, request.quantity, request.price, user_id)



@app.post("/register")
def api_register(request: LoginCredentials):
    return register_user(request.username, request.password)


@app.post("/authenticate")
def api_authenticate_user(request: LoginCredentials):
    user_id = authenticate_user(request.username, request.password)
    if user_id == -1:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"message": "Login successful", "username": request.username, "user_id": user_id}

@app.get("/balance")
def api_get_balance(user_id: int):
    return get_balance(user_id)

@app.get("/chart")
def api_get_tickers(tickers: str = Query(..., description="Comma separated tickers"), time: str = "1Y"):
    ticker_list = tickers.split(",")
    data_list = get_yfdata_cache(ticker_list, time)

    response = {}
    for i, df in enumerate(data_list):
        ticker_name = ticker_list[i]
        if df is None or df.empty:
            response[ticker_name] = []
        else:
            # turn data from index of pandas to col of text
            df_reset = df.reset_index()
            if "Date" in df_reset.columns:
                df_reset["Date"] = df_reset["Date"].astype(str)
            response[ticker_name] = df_reset.to_dict(orient="records")

    return response

@app.get("/trade/info")
def api_trade_ticker(ticker: str = Query(...), user_id: int = Depends(get_current_user)):
    stock_info_dict = get_info([ticker])
    ticker_data = stock_info_dict.get(ticker, {})

    user_stats = get_ticker_stats_owned(user_id, ticker)

    print("trying to get trad einfo for ")
    print(ticker)

    response = {
        "bid": ticker_data.get("bid"),
        "ask": ticker_data.get("ask"),
        "last_price": ticker_data.get("regularMarketPrice"),
        **user_stats
    }
    print(response)
    return response


@app.get("/info")
def api_get_info(tickers: str = Query(..., description="Comma separated tickers"), info: str = ""):
    ticker_list = tickers.split(",")
    if info == "":
        return get_info(ticker_list)
    if info == "financials":
        return get_financial_metrics(ticker_list)
    if info == "balance_sheet":
        return get_balancesheet(ticker_list)
    if info == "my_chart":
        response = {}
        stock_info_dict = get_info(ticker_list)
        lookup_stats = ["dividendYield", "beta", "trailingPE", "forwardPE", "volume", "averageVolume", "bid", "ask",
                        "marketCap", "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "priceToSalesTrailing12Months",
                        "twoHundredDayAverage", "profitMargins", "heldPercentInsiders", "priceToBook",
                        "earningsQuarterlyGrowth", "debtToEquity", "returnOnEquity", "earningsGrowth",
                        "revenueGrowth", "grossMargins", "trailingPegRatio"]
        for ticker in ticker_list:
            response[ticker] = {}
            for stat in lookup_stats:
                response[ticker][stat] = stock_info_dict.get(ticker).get(stat)
        return response
    return {}

@app.get("/model/predict/{ticker}")
async def api_model_predict_next_day(ticker: str):
    try:
        prediction, last_close = await model_service.predict_next_day(ticker)
        return {
            "ticker": ticker,
            "prediction": float(prediction),
            "last_close": float(last_close)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model/kelly-picks")
def api_get_kelly_picks(version: str = "A"):
    try:
        final_allocations, _ = calculate_kelly_allocations(version, False)
        if final_allocations is None:
            return []

        response = [
            {"ticker": ticker, "allocation": float(alloc), "projected_return": float(mu)}
            for ticker, alloc, mu in final_allocations
        ]
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/current-picks")
async def api_get_current_picks():
    try:
        result = await model_service.handle_fastest_kelly()
        if result is None:
            return []
        response = [
            {"ticker": ticker, "allocation": float(alloc), "projected_return": float(mu)}
            for ticker, alloc, mu in result
        ]
        return response
    except Exception as e:
        print(f"Error fetching current picks: {e}")
        return []


@app.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:

    user_id = authenticate_user(form_data.username, form_data.password)
    if not user_id or user_id == -1 :
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")

