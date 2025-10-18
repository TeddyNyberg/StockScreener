Stock Screening app developed by Ted Nyberg.

The goal of this app is to make stock research easier and quicker, increasing investor confidence.

Features
  - Transformer-based Neural Network to predict stock price
      - Finds patterns and connections to previous days data
      - stores data in AWS S3 bucket
      - runs on AWS SageMaker env
  - Optimal position sizing guidance
      - Kelly Criterion
      - assumes 0 cost trades, 0 taxes, instant model_input -> model_output -> buy/sell   
  - Backtesting
      - Default starts 10/4 (day after model created) 
  - Stock lookup and comparison
      - Comparison charts
  - Persistent Watchlist w SQL

Goals
  - Create model for sentiment analysis on earnings calls, classifying them as trustworthy and likely to grow
  - Better transformer model. Hyperparam tuning, feat engineering, better train and test split
  - More models for stock prediciton, then combine them for final pick (average, most confident, ...)
  - Paper trading
      - Possibly w multiple accounts? 
  - Cleaner, better looking UI
      - Make stock info cleaner, 1,000M instead of 1000000000
      - Add hoverability to comparison chart for both stocks? see if it looks good
  - Faster predictions, model optimization, logic optimization 

