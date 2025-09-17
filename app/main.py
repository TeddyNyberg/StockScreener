from PySide6.QtWidgets import QApplication
from app.ui import MainWindow


## pyinstaller app/main.py --onedir --name stock_screener

## alt do --onefile for cleaner exp but slowwwwwwwwwwww

def main():
    print("Starting app...")
    #initialize_database() one time use
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()




"""

def initialize_database():

    data_handler = DataHandler()

    tickers = get_sp500_tickers()

    start_date, end_date = get_date_range("3Y")

    all_stock_data = []

    for ticker in tickers:
        try:
            print(f"Fetching data for {ticker}...")
            stock_data = fetch_stock_data(ticker, start_date, end_date)

            if stock_data is not None:
                all_stock_data.append(stock_data)

                s3_path = f"historical_data/{ticker.lower()}.parquet"
                data_handler.save_to_s3(stock_data, S3_BUCKET_NAME, s3_path)
            else:
                print(f"Skipping {ticker} due to no data.")

        except Exception as e:
            print(f"Failed to process {ticker}: {e}")

    print("Finished fetching and saving data. Returning the list of DataFrames.")
    return all_stock_data
"""

if __name__ == "__main__":
    main()


    """
    how all the mdoel works # 1. Load and prepare your data using the functions from data.py
# Assuming you have a list of DataFrames for various stocks
list_of_dfs = [...] # Your function to load data
processed_dfs = data.feat_engr(list_of_dfs)
data_tensors = data.df_to_tensor(processed_dfs)

# 2. Create your Dataset and DataLoader
stock_dataset = data.StockDataset(data_tensors)
dataloader = DataLoader(stock_dataset, batch_size=32, shuffle=True)

# 3. Define your model and other parameters
num_features = 23 # The number of features in your DataFrame columns
embedding_dim = 256
model = model.StockTransformerModel(num_features, embedding_dim)
criterion = nn.MSELoss() # Or another loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. Start the training loop
for epoch in range(10): # Example training loop
    for i, batch in enumerate(dataloader):
        # Batch shape: (batch_size, sequence_length, num_features)
        
        # Get your features (x) and targets (y) from the batch
        # This will depend on how your final tensor is structured
        
        # Forward pass
        # output = model(x)
        # loss = criterion(output, y)
        
        # Backward and optimize
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        
    """
