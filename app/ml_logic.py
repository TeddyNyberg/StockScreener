import boto3
import io
import tarfile
import torch
import pandas as pd
from data import fetch_stock_data, normalize_window, get_sp500_tickers
from newattemptmodel import pred_next_day_no_ticker
from search import get_date_range, get_close_on
from settings import *
from datetime import datetime

s3_client = boto3.client(
        's3',
        aws_access_key_id = AWS_ACC_KEY_ID,
        aws_secret_access_key= AWS_SCR_ACC_KEY
    )

_MODEL_STATE_DICT = None
_CONFIG = None

def optimal_picks(today = pd.Timestamp.today().normalize()):

    model_state_dict, config = _load_model_artifacts()

    start, end = get_date_range("3M", today)
    sp_tickers = get_sp500_tickers()
    sp_tickers.append("^SPX")

    all_predictions = []
    for ticker in sp_tickers:
        if "." in ticker:
            ticker = ticker.replace(".", "-")

        try:
            input_tensor, latest_close_price, mean, std = _prepare_data_for_prediction(ticker, start, end)
            prediction = pred_next_day_no_ticker(input_tensor, model_state_dict, config, mean, std)
            delta = ((prediction - latest_close_price) / latest_close_price).item()
            all_predictions.append((ticker, delta))
        except ValueError as e:
            print(f"Warning: Skipping {ticker}. {e}")
            continue
        except Exception as e:
            print(f"Warning: Skipping {ticker} for prediction due to data/network error: {e}")
            continue

    sorted_predictions = sorted(all_predictions, key=lambda x: x[1], reverse=True)

    print("\n--- Sorted Predictions (Highest Predicted Return First) ---")

    for ticker, delta in sorted_predictions[:10]:
        print(f"Stock: {ticker}, Predicted Delta: {delta:+.4f}")

    print("\n--- Lowest Predictions ---")
    for ticker, delta in sorted_predictions[-5:]:
        print(f"Stock: {ticker}, Predicted Delta: {delta:+.4f}")

    if sorted_predictions:
        best_stock, best_value = sorted_predictions[0]
        print(f"\nOverall Best Stock Pick: {best_stock}")
        print(f"Predicted Value: {best_value:+.4f}")
    else:
        print("No predictions were successfully generated.")

    return sorted_predictions, all_predictions[-1]


def predict_single_ticker(ticker):

    model_state_dict, config = _load_model_artifacts()

    start, end = get_date_range("3M")
    input_tensor, _, mean, std = _prepare_data_for_prediction(ticker, start, end)
    prediction = pred_next_day_no_ticker(input_tensor, model_state_dict, config, mean, std)

    print(f"Predicted value: {prediction}")
    return prediction

def get_historical_volatility(ticker, start, end):
    df = fetch_stock_data(ticker, start, end)
    df['Returns'] = df['Close'].pct_change().dropna()
    daily_variance = df['Returns'].var()
    annualized_variance = daily_variance * 252
    return annualized_variance


def _prepare_data_for_prediction(ticker, start, end):
    df = fetch_stock_data(ticker, start, end)
    data = df["Close"]

    seq_size = 50
    if len(data) < seq_size:
        raise ValueError(f"Insufficient data for {ticker}. Need at least {seq_size} points, got {len(data)}.")

    latest_close_price = df["Close"].iloc[-1]
    input_sequence = data[-seq_size:]

    normalized_window, mean, std = normalize_window(input_sequence)

    input_tensor = torch.tensor(normalized_window.to_numpy(), dtype=torch.float32).unsqueeze(0)

    return input_tensor, latest_close_price, mean, std


def _load_model_artifacts():
    global _MODEL_STATE_DICT, _CONFIG

    if _MODEL_STATE_DICT is not None and _CONFIG is not None:
        print("Using cached model artifacts.")
        return _MODEL_STATE_DICT, _CONFIG

    print("Loading model artifacts from S3...")

    model_buffer = io.BytesIO()

    s3_client.download_fileobj("sagemaker-us-east-1-307926602475", MODEL_ARTIFACTS_PREFIX, model_buffer)
    model_buffer.seek(0)
    print("Downloaded entire model archive to memory.")

    with tarfile.open(fileobj=model_buffer, mode='r:gz') as tar:
        with tar.extractfile('model.pth') as f:
            checkpoint = torch.load(io.BytesIO(f.read()))

    _MODEL_STATE_DICT = checkpoint.get("model_state")
    _CONFIG = checkpoint.get("config")

    if _MODEL_STATE_DICT is None:
        raise KeyError("Could not find 'model_state' key in the loaded dictionary.")

    print("Model state dict and config loaded successfully.")
    return _MODEL_STATE_DICT, _CONFIG


def calculate_kelly_allocations(lookback_period="6M", end=None):

    # if end is none, get range until today
    # if end is not none, backtest
    start, end = get_date_range(lookback_period, end)

    sorted_predictions, spy_delta = optimal_picks(end)

    if not sorted_predictions:
        print("No predictions available to calculate Kelly bets.")
        return None

    RISK_FREE_RATE = 0.005  # Assuming 0.5% annualized risk-free rate (r)

    kelly_allocations = []

    for ticker, predicted_delta in sorted_predictions:
        mu = predicted_delta

        try:
            sigma_squared = get_historical_volatility(ticker, start, end)

            # Ensure variance is positive to avoid division by zero or errors
            if sigma_squared <= 0:
                print(f"Skipping {ticker}: Volatility (sigma^2) is zero or negative.")
                continue

            kelly_fraction = (mu - RISK_FREE_RATE) / sigma_squared

            # WE GOING FULL KELLY
            HALF_KELLY_MULTIPLIER = 1
            allocation = kelly_fraction * HALF_KELLY_MULTIPLIER

            if allocation < 0:
                allocation = 0.0
            elif allocation > 1.0:
                allocation = 1.0

            kelly_allocations.append((ticker, allocation, mu, sigma_squared))
        except Exception as e:
            # Handle any failure during volatility calculation and skip the ticker for the day.
            print(f"Warning: Skipping {ticker} from Kelly allocation due to data/calculation error: {e}")
            continue

    total_allocation = sum(allocation for _, allocation, _, _ in kelly_allocations)

    normalization_factor = 1.0 / total_allocation if total_allocation > 1.0 else 1.0

    print("\n--- Continuous Kelly-Based Position Sizing ---")
    print(f"Total Unnormalized Allocation: {total_allocation * 100:.2f}%")
    print(f"Normalization Factor (if > 100%): {normalization_factor:.4f}")

    final_allocations_unsorted = []
    for ticker, allocation, mu, sigma_squared in kelly_allocations:
        normalized_allocation = allocation * normalization_factor
        if normalized_allocation > 0:
            final_allocations_unsorted.append((ticker, normalized_allocation, mu))
            print(
                f"Stock: {ticker}, μ: {mu:+.4f}, σ²: {sigma_squared:.4f}, Allocation: {normalized_allocation * 100:.2f}%")

    final_allocations = sorted(final_allocations_unsorted, key=lambda x: x[1], reverse=True)

    return final_allocations

def handle_backtest(start_date_str: str = "2025-10-03", initial_capital: float = 100000.0, lookback_period: str = "6M"):

    print(f"Starting backtest from {start_date_str} with ${initial_capital:,.2f}...")

    daily_position_reports = {}
    daily_pnl_reports = {}

    try:
        start_date = pd.to_datetime(start_date_str)

        # pd.to_datetime(datetime.now().strftime('%Y-%m-%d')) = date today at 00:00:00
        # TODO: See what yfticker get data until today before open, during hours, after close
        today = pd.to_datetime(datetime.now().strftime('%Y-%m-%d')) - pd.Timedelta(days=1)
        print("today: ", pd.to_datetime(datetime.now().strftime('%Y-%m-%d')))

        if start_date > today:
            print("Start date is in the future or today. Cannot run backtest.")
            return

        date_range = pd.date_range(start_date, today, freq='B')  # Business days

        portfolio_df = pd.DataFrame(index=date_range, columns=['Total_Value_At_Close', 'Cash'], dtype=float)
        portfolio_df.loc[date_range[0], 'Total_Value_At_Close'] = initial_capital
        portfolio_df.loc[date_range[0], "Cash"] = initial_capital

        current_holdings = {}  # ticker: {"shares": float, "entryprice":float}

        for i in range(1, len(date_range)):
            current_day = date_range[i]
            prev_day = date_range[i - 1]

            print(current_day)
            print(prev_day)

            prev_day_cash_raw = portfolio_df.loc[prev_day, 'Cash']
            if pd.isna(prev_day_cash_raw):
                day_start_capital = 0.0
                print(f"Error: Cash for {prev_day.strftime('%Y-%m-%d')} is NaN. Assuming 0 capital.")
            else:
                day_start_capital = float(prev_day_cash_raw)

            # 0.0005 5bp
            # 0 for now
            transaction_cost_pct = 0
            daily_pnl_list = []

            tickers_to_remove = list(current_holdings.keys())
            for ticker in tickers_to_remove: #LIQUIDATE
                holding = current_holdings.get(ticker, {})
                shares = holding.get("shares", 0)
                entry_price = holding.get("entry_price", -1)

                if shares == 0:
                   continue

                closing_price = get_close_on(ticker, prev_day)  # Price from the previous day's close

                if closing_price is not None and entry_price is not -1:
                    sale_value = shares * closing_price
                    transaction_cost = sale_value * transaction_cost_pct

                    purchase_value = shares * entry_price
                    gain_loss = sale_value - purchase_value
                    day_start_capital += sale_value - transaction_cost

                    daily_pnl_list.append({
                        "Ticker": ticker,
                        "Shares_Sold": shares,
                        "Entry_Price": entry_price,
                        "Exit_Price": closing_price,
                        "Sale_Value": sale_value,
                        "Transaction_Cost": transaction_cost,
                        "Gain_Loss": gain_loss
                    })
                else:
                    print(
                        f"Warning: Price unavailable for {ticker} on {prev_day}. Assuming zero contribution to daily capital.")

            if daily_pnl_list:
                daily_pnl_reports[current_day.strftime('%Y-%m-%d')] = pd.DataFrame(daily_pnl_list)

            portfolio_df.loc[current_day, 'Total_Value_At_Close'] = day_start_capital

            data_end_date = prev_day

            kelly_allocations = calculate_kelly_allocations(
                lookback_period,
                data_end_date
            )

            total_capital_available = day_start_capital
            cash_for_trading = day_start_capital  # We start with cash = total capital, then sell/buy

            new_holdings = {}


            daily_allocations_list = []

            total_allocation_value = 0

            for ticker, kelly_fraction, _ in kelly_allocations:
                target_value = total_capital_available * kelly_fraction
                close_price = get_close_on(ticker, prev_day)

                if close_price is not None:
                    buy_value = min(target_value, cash_for_trading)

                    shares_to_buy = buy_value / close_price
                    transaction_cost = buy_value * transaction_cost_pct

                    if cash_for_trading >= (buy_value + transaction_cost):
                        cash_for_trading -= (buy_value + transaction_cost)
                        new_holdings[ticker] = {"shares": shares_to_buy, "entry_price": close_price, "closing_price": close_price}

                        total_allocation_value += buy_value

                        daily_allocations_list.append({
                            "Ticker": ticker,
                            "Allocation_Percent": kelly_fraction,
                            "Entry_Price": close_price,
                            "Target_Value": target_value,
                            "Actual_Shares": shares_to_buy
                        })

            current_holdings = new_holdings
            portfolio_df.loc[prev_day, 'Cash'] = float(cash_for_trading)
            portfolio_df.loc[prev_day, 'Total_Value_At_Close'] = float(cash_for_trading + total_allocation_value)

            if daily_allocations_list:
                allocations_df = pd.DataFrame(daily_allocations_list)
                daily_position_reports[current_day.strftime("%Y-%m-%d")] = allocations_df

            # print(f"Day {current_day.strftime('%Y-%m-%d')} | End Value: ${portfolio_df.loc[current_day, 'Total_Value']:,.2f} | Cash: ${cash_for_trading:,.2f}")


        file_path = 'backtest_results.xlsx'
        writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
        portfolio_df.to_excel(writer, sheet_name="Summary_Performance")

        #for pnl sheet, to see loss and gain on 5/1, look at sheet named 5/2 pnl
        #for allocations, the stocks syou see on 5/1 are the stocks you own on 5/1

        for date_str, df in daily_position_reports.items():
            sheet_name = f"{date_str}_Allocations"
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        for date_str, df in daily_pnl_reports.items():
            sheet_name = f"{date_str}_PnL"
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        writer.close()
        print("\n" + "=" * 50)
        print(f"Backtest complete. Final Value: ${portfolio_df.iloc[-1]['Total_Value']:,.2f}")
        print(f"Results saved to {file_path}")
        print("=" * 50)

    except Exception as e:
        print(f"Backtesting error: {e}")

    print(portfolio_df)

    return portfolio_df