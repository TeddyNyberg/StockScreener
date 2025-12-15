Stock Screening app developed by Ted Nyberg.

The goal of this app is to make stock research easier and quicker, increasing investor confidence.

Features
  - Transformer-based Neural Network to predict stock price
      - Finds patterns and connections to previous days data
      - Stores data in AWS S3 bucket
      - Runs on AWS SageMaker env
  - Optimal position sizing guidance
      - Kelly Criterion
      - Assumes 0 cost trades, 0 taxes, instant model_input -> model_output -> buy/sell   
  - Backtesting
      - Default starts 1/28 (model trained on data until 1/27)
      - To view performance:
          - look at nyberg_results_static.csv and backtest_portfolio_PL.xlsx
          - in the app, lookup "nyberg-a" or "nyberg-b"
  - Stock lookup and comparison
      - Comparison charts
  - Persistent per-Account Watchlist and Paper Trading w SQL

Goals
  - Create model for sentiment analysis on earnings calls, classifying them as trustworthy and likely to grow
  - Better transformer model. Hyperparam tuning, feat engineering, better train and test split
  - More models for stock prediction, then combine them for final pick (average, most confident, ...)
  - Cleaner, better looking UI
      - Make stock info cleaner, 1,000M instead of 1000000000
      - Add hoverability to comparison chart for both stocks? see if it looks good
  - Faster predictions, model optimization, logic optimization 
  - Fuller testing
  - Notes and Alerts(custom alerts and earnings, ex-div, splits) for watchlist

Achievements
  - 24.7% gain YTD as of 12/11/2025, outperforming the S&P Total Return by 10.0 points
  - \>95% speed increase (compared to initial model - commit 60f7cb4)
      - Optimized model pipeline by implementing batch prediction/vectorization for 500 tickers (S&P 500), drastically reducing computation time from minutes to seconds
      - More efficient data-loading
  - This speed increase does NOT include the model quantization which will also increase speed but is currently untested
 
















