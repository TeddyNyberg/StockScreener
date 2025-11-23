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
  - Persistent Watchlist and Paper Trading w SQL

Goals
  - Create model for sentiment analysis on earnings calls, classifying them as trustworthy and likely to grow
  - Better transformer model. Hyperparam tuning, feat engineering, better train and test split
  - More models for stock prediciton, then combine them for final pick (average, most confident, ...)
  - Add accounts for paper trading
  - Cleaner, better looking UI
      - Make stock info cleaner, 1,000M instead of 1000000000
      - Add hoverability to comparison chart for both stocks? see if it looks good
  - Faster predictions, model optimization, logic optimization 

Achievements
  - 19.8% gain YTD as of 11/19/2025, outperforming the S&P Total Return by 9.6 points
  - 85% speed increase (compared to intial model - commit 60f7cb4)
      - Data-level parallelization across the CPU's multi-core architecture
      - More efficient data-loading
  - This speed increase does NOT include the model quantization which will also increase speed but is currently untested
  - Enhanced stability and scalability. Resolved critical memory error on Windows spawned processes.









