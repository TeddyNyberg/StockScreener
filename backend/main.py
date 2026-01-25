from fastapi import Query
from contextlib import asynccontextmanager
from backend.app.db.db_handler import *
from backend.app.data.data_cache import get_yfdata_cache
from fastapi.middleware.cors import CORSMiddleware


# to run, uvicorn backend.main:app --reload

@asynccontextmanager
async def lifespan(_: FastAPI):
    print("Connecting to Database...")
    try:
        with DB() as conn:
            init_user_table(conn)
            init_db(conn)
    except Exception as e:
        print(f"Database Init Failed: {e}")
    yield
    print("Server Shutting Down...")

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
    user_id: int


class StockTransaction(BaseModel):
    ticker: str
    user_id: int
    price: float
    quantity: int


class LoginCredentials(BaseModel):
    username: str
    password: str






@app.get("/watchlist/{user_id}")
def api_get_watchlist(user_id: int):
    data = get_watchlist(user_id)
    if not data:
        return []
    return data

@app.post("/watchlist/add")
def api_add_watchlist(request: WatchlistRequest):
    success = add_watchlist(request.ticker, request.user_id)
    if success:
        return {"status": "success", "message": f"Added {request.ticker}"}
    else:
        raise HTTPException(status_code=400, detail="Could not add ticker")

@app.post("/watchlist/remove")
def api_rm_watchlist(request: WatchlistRequest):
    success = rm_watchlist(request.ticker, request.user_id)
    if success:
        return {"status": "success", "message": f"Added {request.ticker}"}
    else:
        raise HTTPException(status_code=400, detail="Could not add ticker")

@app.get("/portfolio")
def api_get_portfolio(user_id: int):
    data = get_portfolio(user_id)
    return data

@app.post("/buy-stock")
def api_buy_stock(request: StockTransaction):
    buy_stock(request.ticker, request.quantity, request.price, request.user_id)

@app.post("/sell-stock")
def api_sell_stock(request: StockTransaction):
    sell_stock(request.ticker, request.quantity, request.price, request.user_id)

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


@app.get("/{tickers}/{time}")
def api_get_tickers(tickers: str, time: str):
    tickers = tickers.split("&")
    data = get_yfdata_cache(tickers, time)
    return data


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
