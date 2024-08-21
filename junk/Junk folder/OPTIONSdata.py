import yfinance as yf
import pandas as pd

def fetch_all_options_data(ticker):
    stock = yf.Ticker(ticker)
    all_options = []

    # Get all expiration dates
    expiration_dates = stock.options

    # Check if expiration dates are available
    if not expiration_dates:
        print("No expiration dates available for this ticker.")
        return pd.DataFrame()

    # Fetch options data for all expiration dates
    for exp_date in expiration_dates:
        options_chain = stock.option_chain(exp_date)
        calls = options_chain.calls
        puts = options_chain.puts
        
        # Add expiration date to each options DataFrame
        if not calls.empty:
            calls['expirationDate'] = exp_date
        if not puts.empty:
            puts['expirationDate'] = exp_date

        # Append calls and puts data if they are not empty
        if not calls.empty:
            all_options.append(calls)
        if not puts.empty:
            all_options.append(puts)

    # Combine all options data into a single DataFrame
    if all_options:
        options_data = pd.concat(all_options)
    else:
        print("No options data available for this ticker.")
        options_data = pd.DataFrame()

    return options_data

def main():
    ticker = input("Enter the ticker symbol: ").upper()

    options_data = fetch_all_options_data(ticker)
    if not options_data.empty:
        print(options_data)
    else:
        print("No options data to display.")

if __name__ == "__main__":
    main()
