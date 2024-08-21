import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fetch historical data
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Adj Close']

# Calculate daily returns
def calculate_returns(data):
    returns = data.pct_change().dropna()
    return returns

# Calculate Value at Risk (VaR)
def calculate_var(returns, confidence_level=0.95):
    if isinstance(returns, pd.Series):
        sorted_returns = sorted(returns)
        index = int((1-confidence_level) * len(sorted_returns))
        return abs(sorted_returns[index])
    else:
        raise TypeError("Returns must be a pandas Series")

# Calculate Conditional Value at Risk (CVaR)
def calculate_cvar(returns, confidence_level=0.95):
    var = calculate_var(returns, confidence_level)
    cvar = returns[returns <= -var].mean()
    return abs(cvar)

# Perform a basic stress test
def stress_test(returns, stress_factor):
    stressed_returns = returns * stress_factor
    return stressed_returns

# Plotting function
def plot_returns(returns, title):
    plt.figure(figsize=(10, 6))
    returns.plot(title=title)
    plt.show()

# Main function to execute the tasks
def main():
    ticker = input("Enter the stock ticker (e.g., 'AAPL'): ")
    start_date = input("Enter the start date (YYYY-MM-DD): ")
    end_date = input("Enter the end date (YYYY-MM-DD): ")
    
    data = fetch_data(ticker, start_date, end_date)
    returns = calculate_returns(data)
    var = calculate_var(returns)
    cvar = calculate_cvar(returns)
    stress_factor = float(input("Enter a stress factor (e.g., 2 for doubling the impact): "))
    stressed_returns = stress_test(returns, stress_factor)
    
    print(f"Value at Risk (95% confidence): {var:.2%}")
    print(f"Conditional Value at Risk (95% confidence): {cvar:.2%}")
    
    plot_returns(returns, 'Daily Returns')
    plot_returns(stressed_returns, 'Stressed Returns')

if __name__ == "__main__":
    main()
