import asyncio
from app.data.preprocessor_utils import prepare_data_for_prediction
from app.data.yfinance_fetcher import get_historical_data, LiveMarketTable
from app.ml_logic.model_loader import load_model_artifacts
from app.ml_logic.prediction_models.only_close_model import setup_pred_model
from app.data.data_cache import get_volatility_cache, get_cached_49days
from app.ml_logic.strategy import fastest_kelly
from app.utils import get_date_range
from config import MODEL_MAP
import numpy as np


class ModelService:
    def __init__(self):
        self.model=None
        self.market_data=None
        self.volatility=None
        self.data=None
        self.tickers=None

    async def initialize(self):
        model_state_dict, config = load_model_artifacts(MODEL_MAP["A"]["model_filepath"])
        self.model = setup_pred_model(model_state_dict, config, False)
        self.market_data = LiveMarketTable()
        self.volatility = get_volatility_cache()
        data = get_cached_49days()
        self.data = data.to_numpy(dtype=np.float32)
        self.tickers = np.array(data.columns)  # need to get tickers from data, so order is maintained

        await self.market_data.start_socket(self.tickers)

    async def predict_next_day(self, ticker):
        start, end = get_date_range("3M")
        data = get_historical_data(ticker, start, end)
        close_data = data["Close"]
        input_tensor, _, mean, std = prepare_data_for_prediction(close_data)
        prediction = self.model.predict_value(input_tensor, mean, std)

        print(f"Predicted value: {prediction}")
        print(f"Lastest data: {close_data.iloc[-1].item()}")
        return prediction, close_data.iloc[-1].item()

    async def handle_fastest_kelly(self):
        final_df = await self.market_data.close_socket()
        new_row = final_df.values.reshape(1, -1).astype(np.float32)
        data = np.vstack((self.data, new_row))
        fastest_kelly(data, self.model, self.volatility, self.tickers)



