from backend.app.data.data_cache import get_yfdata_cache
import pandas as pd

def print_model_characteristics(nyberg_ticker):

    df_n, df_spy = get_yfdata_cache([nyberg_ticker, "NYBERG-D"],"1Y")
    n_data = df_n["Close"]
    spy = df_spy["Close"]

    combined_data = pd.DataFrame({
        'Nyberg_Close': n_data,
        'SPY_Close': spy
    }).dropna()

    combined_data['Nyberg_Return'] = combined_data['Nyberg_Close'].pct_change().mul(100)
    combined_data['SPY_Return'] = combined_data['SPY_Close'].pct_change().mul(100)

    combined_data.dropna(inplace=True)

    nyberg_better = combined_data['Nyberg_Return'] > combined_data['SPY_Return']

    nyberg_wins = nyberg_better.sum()
    spy_wins = (~nyberg_better).sum()

    spy_on_nyberg_win_returns = combined_data.loc[nyberg_better, 'SPY_Return']

    avg_spy_return_on_nyberg_win = spy_on_nyberg_win_returns.mean()

    spy_on_nyberg_win_and_spy_up = spy_on_nyberg_win_returns[spy_on_nyberg_win_returns > 0].mean()
    spy_on_nyberg_win_and_spy_down = spy_on_nyberg_win_returns[spy_on_nyberg_win_returns <= 0].mean()

    avg_abs_nyberg_move = combined_data['Nyberg_Return'].abs().mean()
    avg_abs_spy_move = combined_data['SPY_Return'].abs().mean()

    print("\n**Volatility Comparison (Average Absolute Daily % Move):**")
    print(f"- **{nyberg_ticker} Average Absolute % Move:** {avg_abs_nyberg_move:.4f}%")
    print(f"- **SPY Average Absolute % Move:** {avg_abs_spy_move:.4f}%")

    print(f"**Performance Metrics: {nyberg_ticker} vs. SPY**")
    print("---")

    print(f"**Total Trading Days Analyzed:** {len(combined_data)}")

    print("\n**Comparison Counts (Daily Returns):**")
    print(f"- **{nyberg_ticker} does better:** {nyberg_wins} days")
    print(f"- **SPY does better (or ties):** {spy_wins} days")

    print("\n**Conditional SPY Return (when Nyberg does better):**")
    print(f"- **Average SPY % Change:** {avg_spy_return_on_nyberg_win:.4f}%")

    print("\n**SPY's Conditional Performance Breakdown (on Nyberg's better days):**")
    print(f"- **SPY's Avg % Change on UP Days:** {spy_on_nyberg_win_and_spy_up:.4f}%")
    print(f"- **SPY's Avg % Change on DOWN Days:** {spy_on_nyberg_win_and_spy_down:.4f}%")

