from config import *
from datetime import datetime
import os
import traceback
import pandas as pd
import numpy as np
from app.data.yfinance_fetcher import get_close_on
from app.ml_logic.strategy import calculate_kelly_allocations, tune
from app.ml_logic.helpers import is_tuning_day

#use 1/28/2025 for no leak data, model trained on 10/2; all data until 9-7; train from 2022 > 1/27/2025; test 1/28/2025
def handle_backtest(start_date_str = "1/28/2025", initial_capital = initial_capital_fully,
                    model_version="A", tuning_period = None, only_largest=False):



    print(f"Starting backtest from {start_date_str} with ${initial_capital:,.2f}...")
    daily_position_reports = {}
    daily_pnl_reports = {}
    is_quantized = "QUANTIZED" in MODEL_MAP[model_version]["name"]
    filepath = MODEL_MAP[model_version]["model_filepath"]
    try:
        start_date = pd.to_datetime(start_date_str)

        # pd.to_datetime(datetime.now().strftime('%m-%d-%Y')) = date today at 00:00:00
        # TODO: See what yfticker get data until today before open, during hours, after close
        # so today = yesterday bc want to trade until yesterday, dont use todays data
        today = pd.to_datetime(datetime.now().strftime('%m/%d/%Y')) - pd.Timedelta(days=1)
        print("today: ", pd.to_datetime(datetime.now().strftime('%m/%d/%Y')))

        if start_date >= today:
            print("Start date is in the future or today. Cannot run backtest.")
            return None

        date_range = pd.date_range(start_date, today, freq='B')  # Business days
        print(date_range, " DATE RANGE")

        portfolio_df = pd.DataFrame(index=date_range, columns=['Total_Value_At_Close', 'Cash_At_Open'], dtype=float)
        portfolio_df.index.name = "Date"
        portfolio_df.loc[date_range[0], 'Total_Value_At_Close'] = initial_capital
        portfolio_df.loc[date_range[1], "Cash_At_Open"] = initial_capital

        new_holdings = {}
        trading_today = True


        for i in range(1, len(date_range)):

            current_day = date_range[i]
            prev_day = date_range[i - 1]
            print(current_day)

            #TODO: make it trian after i buy on firday to make faster
            if is_tuning_day(current_day, tuning_period):
                ok = tune(filepath, prev_day)
                if not ok:
                    print("tune was not successful")
                    return None

            if trading_today:
                kelly_allocations, all_closes = calculate_kelly_allocations(
                    model_version=model_version,
                    is_quantized=is_quantized,
                    end=current_day,
                    only_largest=only_largest
                )
                if not kelly_allocations:
                    print("no kelly allocations")
                    return None
                total_capital_available = portfolio_df.loc[current_day, 'Cash_At_Open']
                cash_for_trading = total_capital_available

                daily_allocations_list = []

                total_allocation_value = 0

                for ticker, kelly_fraction, _ in kelly_allocations:
                    target_value = total_capital_available * kelly_fraction
                    # buy at prev days close
                    close_price = all_closes[ticker]


                    if close_price is not None:
                        buy_value = min(target_value, cash_for_trading)

                        shares_to_buy = buy_value / close_price
                        transaction_cost = buy_value * transaction_cost_pct

                        if cash_for_trading >= (buy_value + transaction_cost):
                            cash_for_trading -= (buy_value + transaction_cost)
                            new_holdings[ticker] = {"shares": shares_to_buy, "entry_price": close_price}

                            total_allocation_value += buy_value

                            daily_allocations_list.append({
                                "Ticker": ticker,
                                "Allocation_Percent": kelly_fraction,
                                "Entry_Price": close_price,
                                "Target_Value": target_value,
                                "Actual_Shares": shares_to_buy
                            })

            else:
                trading_today = True


            if daily_allocations_list:
                allocations_df = pd.DataFrame(daily_allocations_list)
                daily_position_reports[current_day.strftime("%m-%d-%Y")] = allocations_df

            daily_pnl_list = []

            total_value = cash_for_trading

            tickers_to_remove = list(new_holdings.keys())
            #TODO: again, get all selling data at once, or get it with the other data? nawww
            for ticker in tickers_to_remove: #LIQUIDATE, SELL AT EOD
                holding = new_holdings.get(ticker, {})
                shares = holding.get("shares", 0)
                entry_price = holding.get("entry_price", -1)

                if shares == 0:
                   continue

                closing_price = get_close_on(ticker, current_day)  # Price from the current day's close, bc bought at prev days close

                if closing_price is not None and entry_price != -1:
                    sale_value = shares * closing_price
                    transaction_cost = sale_value * transaction_cost_pct

                    purchase_value = shares * entry_price
                    gain_loss = sale_value - purchase_value
                    total_value += sale_value - transaction_cost

                    daily_pnl_list.append({
                        "Ticker": ticker,
                        "Shares": shares,
                        "Entry_Price": entry_price,
                        "Exit_Price": closing_price,
                        "Sale_Value": sale_value,
                        "Transaction_Cost": transaction_cost,
                        "Gain_Loss": gain_loss
                    })

                else:
                    print(f"Warning: Price unavailable for {ticker} on {current_day}.")
                    trading_today = False
                    break

            if trading_today:
                new_holdings = {}
                daily_pnl_reports[current_day.strftime('%m-%d-%Y')] = pd.DataFrame(daily_pnl_list)
                portfolio_df.loc[current_day, 'Total_Value_At_Close'] = total_value
            else:
                total_value = total_capital_available
                portfolio_df.drop(portfolio_df.index[-1], inplace=True)



            if i < len(date_range)-1:
                if trading_today:
                    portfolio_df.loc[date_range[i+1], 'Cash_At_Open'] = total_value
                else:
                    portfolio_df.loc[date_range[i+1], 'Cash_At_Open'] = cash_for_trading

            print(current_day, " total at close: ", total_value)




        if model_version == "A":
            file_path = 'backtest_portfolio_PL.xlsx'
            writer = pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace')

            for date_str, df in daily_position_reports.items():
                sheet_name = f"{date_str}_Allocations"
                df.to_excel(writer, sheet_name=sheet_name, index=False)

            for date_str, df in daily_pnl_reports.items():
                sheet_name = f"{date_str}_PnL"
                df.to_excel(writer, sheet_name=sheet_name, index=False)


            writer.close()
        print("\n" + "=" * 50)
        print(f"Backtest complete. Final Value: ${portfolio_df.iloc[-1]['Total_Value_At_Close']:,.2f}")


    except Exception as e:
        print(f"Backtesting error: {e}")
        traceback.print_exc()

    print(portfolio_df)

    portfolio_df[['Total_Value_At_Close', 'Cash_At_Open']] = \
        (np.floor(portfolio_df[['Total_Value_At_Close', 'Cash_At_Open']] * 1000) / 1000)
    return portfolio_df

#TODO: combine all th csvs into one and just have dif titles, also tak eout cash at open, uselss

def continue_backtest(version, tuning_period=None, only_largest=False):

    file_path = MODEL_MAP[version]["csv_filepath"]
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    data.index = pd.to_datetime(data.index, format="%m/%d/%Y", errors='raise')


    last_day_tested = data.index.max()
    print("start date: ", last_day_tested, " in ", version)
    last_day_total_value = data.loc[last_day_tested, 'Total_Value_At_Close']


    today = pd.Timestamp(datetime.now().date())
    last_trading_day = today - pd.offsets.BDay(1)

    print(last_trading_day, "ltd")
    if last_day_tested.normalize() >= last_trading_day.normalize():
        print("Backtest is already up-to-date. No new trading days to process.")
        return

    print("-" * 50)
    print(f"Last backtest day: {last_day_tested} with total value: ${last_day_total_value:,.2f}")
    #if error w last day total value, check csv if repeated dates
    print(f"New initial capital: ${last_day_total_value:,.2f}")
    print("-" * 50)

    new_results_df = handle_backtest(
        start_date_str=last_day_tested.strftime('%m/%d/%Y'),
        initial_capital=last_day_total_value,
        model_version=version,
        tuning_period=tuning_period,
        only_largest=only_largest
    )

    if new_results_df is None:
        print("handle_backtest did not return any results")
        return

    new_trading_days_df = new_results_df.iloc[1:]

    if new_trading_days_df.empty:
        print("The new backtest run produced no new trading days to append.")
        return


    mode = "a" if os.path.exists(file_path) else "w"
    header = not os.path.exists(file_path)
    new_trading_days_df.to_csv(file_path, mode=mode, header=header, date_format="%m/%d/%Y")



    print("\n" + "=" * 50)
    print(f"Successfully appended {len(new_trading_days_df)} new days of results.")
    print(f"Backtest now current until: {new_trading_days_df.index[-1].strftime('%m/%d/%Y')}")
    print(f"Results saved to {file_path}")
    print("=" * 50)