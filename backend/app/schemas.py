from pydantic import BaseModel


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


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None
