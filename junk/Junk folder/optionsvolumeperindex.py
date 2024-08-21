import pandas as pd
import yfinance as yf

# Define the index from which you want to fetch the stocks
index = '^RUT'  # Russell 2000 Index (small caps)

# Fetch the constituent stocks of the index
def get_index_constituents(index_symbol):
    index_data = yf.Ticker(index_symbol)
    constituents = index_data.constituents
    return constituents

# Fetch options data for a list of tickers
def fetch_options_data(ticker):
    ticker_data = yf.Ticker(ticker)
    options = ticker_data.options
    options_data = []
    for option_date in options:
        opt = ticker_data.option_chain(option_date)
        calls = opt.calls
        puts = opt.puts
        options_data.append((option_date, calls, puts))
    return options_data

# Main script
if __name__ == "__main__":
    constituents = get_index_constituents(index)
    all_options_data = {}

    for ticker in constituents:
        print(f"Fetching options data for {ticker}")
        options_data = fetch_options_data(ticker)
        all_options_data[ticker] = options_data

    # Save the data to a CSV or analyze further as needed
    for ticker, options_data in all_options_data.items():
        for option_date, calls, puts in options_data:
            calls['Type'] = 'Call'
            puts['Type'] = 'Put'
            df = pd.concat([calls, puts])
            df.to_csv(f'{ticker}_{option_date}_options.csv', index=False)
    
    print("Options data fetching completed.")
