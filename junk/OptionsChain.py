import yfinance as yf
import pandas as pd
import os

def get_ticker_symbol():
    return input("Enter the ticker symbol: ").upper()

def get_expiration_dates(ticker):
    stock = yf.Ticker(ticker)
    return stock.options

def display_expiration_dates(dates):
    for i, date in enumerate(dates):
        print(f"{i + 1}: {date}")

def select_expiration_date(dates):
    index = int(input("Select the expiration date by number: ")) - 1
    return dates[index]

def get_option_chain(ticker, expiration_date):
    stock = yf.Ticker(ticker)
    return stock.option_chain(expiration_date)

def save_option_chain(option_chain, ticker, expiration_date):
    calls = option_chain.calls
    puts = option_chain.puts
    calls['Type'] = 'Call'
    puts['Type'] = 'Put'
    combined = pd.concat([calls, puts])
    combined = combined.reset_index(drop=True)
    
    # Define the path
    path = r"C:\Users\cinco\Desktop\quant practicie\Research\Research Tools\Options\Options Chain"
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Save to CSV
    filename = f"{ticker}_{expiration_date}_options_chain.csv"
    combined.to_csv(os.path.join(path, filename), index=False)
    print(f"Option chain data saved to {os.path.join(path, filename)}")

def main():
    ticker = get_ticker_symbol()
    expiration_dates = get_expiration_dates(ticker)
    display_expiration_dates(expiration_dates)
    selected_date = select_expiration_date(expiration_dates)
    option_chain = get_option_chain(ticker, selected_date)
    save_option_chain(option_chain, ticker, selected_date)

if __name__ == "__main__":
    main()
