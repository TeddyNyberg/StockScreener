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







def position_performances():

    from backend.app.config import PL_PATH


    sheets = pd.read_excel(PL_PATH, sheet_name=None)


    results = []

    for sheet_name, df in sheets.items():
        if "PnL" in sheet_name:

            # Ensure required columns exist
            if "Entry_Price" not in df.columns or "Exit_Price" not in df.columns:
                print(f"Skipping {sheet_name}: missing columns")
                continue

            df["return"] = (df["Exit_Price"] - df["Entry_Price"]) / df["Entry_Price"]

            df = df.reset_index(drop=True)
            df["rank"] = df.index

            results.append(df[["rank", "return"]])

    combined = pd.concat(results, ignore_index=True)

    stats_by_rank = (
        combined
        .groupby("rank")["return"]
        .agg(
            avg_return="mean",
            std_return="std",
            p25=lambda x: x.quantile(0.25),
            p75=lambda x: x.quantile(0.75)
        )
        .reset_index()
    )


    counts = combined.groupby("rank").size().reset_index(name="num_days")

    final = stats_by_rank.merge(counts, on="rank")
    final = final.sort_values("rank")


    pd.set_option('display.max_rows', None)
    print(final)

def allocation_rank_movement_performance():
    import pandas as pd
    from backend.app.config import PL_PATH

    sheets = pd.read_excel(PL_PATH, sheet_name=None)

    prev_rank_map = None
    results = []

    for sheet_name, df in sheets.items():
        if "PnL" not in sheet_name:
            continue

        # Validate required columns
        required = {"Ticker", "Entry_Price", "Exit_Price"}
        if not required.issubset(df.columns):
            print(f"Skipping {sheet_name}: missing columns")
            continue

        # Compute return
        df["return"] = (df["Exit_Price"] - df["Entry_Price"]) / df["Entry_Price"]

        # Rank = index (already sorted by allocation)
        df = df.reset_index(drop=True)
        df["rank"] = df.index

        # Build current rank map
        curr_rank_map = dict(zip(df["Ticker"], df["rank"]))

        if prev_rank_map is not None:
            for _, row in df.iterrows():
                ticker = row["Ticker"]
                curr_rank = row["rank"]
                ret = row["return"]

                if ticker not in prev_rank_map:
                    category = "new"
                else:
                    prev_rank = prev_rank_map[ticker]

                    if curr_rank < prev_rank:
                        category = "up"      # higher allocation now
                    elif curr_rank > prev_rank:
                        category = "down"    # lower allocation now
                    else:
                        continue  # unchanged rank → ignore

                results.append({
                    "category": category,
                    "return": ret
                })

        prev_rank_map = curr_rank_map

    combined = pd.DataFrame(results)

    if combined.empty:
        print("No comparable data found.")
        return

    stats = (
        combined
        .groupby("category")["return"]
        .agg(
            avg_return="mean",
            std_return="std",
            p25=lambda x: x.quantile(0.25),
            p75=lambda x: x.quantile(0.75),
            count="size"
        )
        .reset_index()
    )

    print(stats)

if __name__ == "__main__":
    print("--- pos perf ---")

    position_performances()

    print("================")

    allocation_rank_movement_performance()
