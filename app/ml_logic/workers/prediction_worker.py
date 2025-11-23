import torch
import numpy as np


_model_cache = None
def quantized_prediction_worker(ticker, input_tensor, latest_close_price, mean, std, model_state_dict, config):
    from app.ml_logic.pred_models.only_close_model import setup_pred_model
    global _model_cache

    if _model_cache is None:
        model, _ = setup_pred_model(model_state_dict, config, True)
        _model_cache = model
    model = _model_cache
    try:
        with torch.no_grad():
            normalized_prediction = model(input_tensor)
        last_prediction = normalized_prediction[-1][0]
        prediction_np = np.array(last_prediction.item()).reshape(1, -1)
        prediction = (prediction_np[0][0] * std) + mean
        delta = ((prediction - latest_close_price) / latest_close_price).item()
        return ticker, delta
    except Exception as e:
        raise Exception(f"Prediction failed for {ticker}: {e}")

def prediction_worker(ticker, input_tensor, latest_close_price, mean, std, model):
    try:
        with torch.no_grad():
            normalized_prediction = model(input_tensor)
        last_prediction = normalized_prediction[-1][0]
        prediction_np = np.array(last_prediction.item()).reshape(1, -1)
        prediction = (prediction_np[0][0] * std) + mean
        delta = ((prediction - latest_close_price) / latest_close_price).item()
        return ticker, delta
    except Exception as e:
        raise Exception(f"Prediction failed for {ticker}: {e}")

