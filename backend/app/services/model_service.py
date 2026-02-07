from backend.app.data.preprocessor_utils import prepare_data_for_prediction
from backend.app.data.yfinance_fetcher import get_historical_data, LiveMarketTable
from backend.app.ml_logic.model_loader import load_model_artifacts
from backend.app.ml_logic.prediction_models.only_close_model import setup_pred_model
from backend.app.data.data_cache import get_volatility_cache, get_cached_49days
from backend.app.ml_logic.strategy import fastest_kelly
from backend.app.utils import get_date_range
from backend.app.config import MODEL_MAP
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
        print(self.tickers, " IN MOD SERV INIT  ")

        await self.market_data.initialize(tickers=self.tickers)

    async def predict_next_day(self, ticker):
        start, end = get_date_range("3M")
        data = get_historical_data(ticker, start, end)
        close_data = data["Close"]
        input_tensor, _, mean, std = prepare_data_for_prediction(close_data)
        prediction = self.model.predict_value(input_tensor, mean, std)

        if hasattr(prediction, 'item'):
            prediction = prediction.item()

        print(f"Predicted value: {prediction}")
        print(f"Lastest data: {close_data.iloc[-1].item()}")
        return prediction, close_data.iloc[-1].item()

    async def handle_fastest_kelly(self):
        final_df = self.market_data.get_snapshot()
        new_row = final_df.values.reshape(1, -1).astype(np.float32)
        data = np.vstack((self.data, new_row))
        return fastest_kelly(data, self.model, self.volatility, self.tickers)

    async def shutdown(self):
        if self.market_data is not None:
            await self.market_data.close_socket()



