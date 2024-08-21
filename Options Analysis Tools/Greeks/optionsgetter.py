import yfinance as yf
import pandas as pd

def get_all_options_data(stock_symbol):
    # Get the stock object
    stock = yf.Ticker(stock_symbol)
    
    # Get the expiration dates
    expiration_dates = stock.options
    
    # Initialize empty dataframes for calls and puts
    all_calls = pd.DataFrame()
    all_puts = pd.DataFrame()
    
    # Loop over each expiration date and fetch the options data
    for expiration in expiration_dates:
        # Get the options data for the expiration date
        opt_chain = stock.option_chain(expiration)
        
        # Append the calls and puts to the dataframes
        all_calls = all_calls._append(opt_chain.calls, ignore_index=True)
        all_puts = all_puts._append(opt_chain.puts, ignore_index=True)
    
    return all_calls, all_puts

# Example usage
stock_symbol = "SPY"  # Replace with your desired stock symbol
calls, puts = get_all_options_data(stock_symbol)

# Display the data
print("Calls:\n", calls)
print("\nPuts:\n", puts)
