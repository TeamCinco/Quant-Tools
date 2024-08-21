import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

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

def calculate_std_prices(ticker):
    data = yf.download(ticker, period='6mo')
    data['Daily_Price_Difference'] = data['Close'] - data['Open']
    data['Weekly_Price_Difference'] = data['Close'] - data['Open'].shift(4)
    data['Monthly_Price_Difference'] = data['Close'] - data['Open'].shift(19)

    daily_std = np.std(data['Daily_Price_Difference'])
    weekly_std = np.std(data['Weekly_Price_Difference'].dropna())
    monthly_std = np.std(data['Monthly_Price_Difference'].dropna())

    current_stock_price = data['Close'].iloc[-1]
    
    return data, daily_std, weekly_std, monthly_std, current_stock_price

def plot_frequency_histogram(data, ticker, frequency):
    plt.figure(figsize=(10, 6))
    if frequency == 'daily':
        plt.hist(data['Daily_Price_Difference'], bins=30, color='blue', alpha=0.7)
    elif frequency == 'weekly':
        plt.hist(data['Weekly_Price_Difference'].dropna(), bins=30, color='green', alpha=0.7)
    else:  # monthly
        plt.hist(data['Monthly_Price_Difference'].dropna(), bins=30, color='red', alpha=0.7)
    
    plt.title(f'{ticker} {frequency.capitalize()} Price Difference Histogram')
    plt.xlabel('Price Difference')
    plt.ylabel('Frequency')
    plt.show()

def calculate_option_profit(option_type, strike_price, premium, stock_price):
    if option_type == 'call':
        profit = max(stock_price - strike_price, 0) - premium
    else:  # put
        profit = max(strike_price - stock_price, 0) - premium
    return profit

def main():
    # 1. Ask for ticker
    ticker = get_ticker_symbol()
    
    # 2. User picks expiration from list
    expiration_dates = get_expiration_dates(ticker)
    display_expiration_dates(expiration_dates)
    selected_date = select_expiration_date(expiration_dates)
    
    # 3. User picks whether they want daily, weekly or monthly std, histogram frequency data
    frequency = input("Choose frequency for STD calculation (daily/weekly/monthly): ").lower()
    
    data, daily_std, weekly_std, monthly_std, current_stock_price = calculate_std_prices(ticker)
    
    if frequency == 'daily':
        std = daily_std
    elif frequency == 'weekly':
        std = weekly_std
    else:
        std = monthly_std
    
    print(f"\nCurrent stock price: ${current_stock_price:.2f}")
    print(f"{frequency.capitalize()} Standard Deviation: ${std:.2f}")
    
    # Show the histogram for the selected frequency
    plot_frequency_histogram(data, ticker, frequency)
    
    # 4. User picks either call or puts
    option_type = input("Enter option type (call/put): ").lower()
    
    # 5. The user picks strike price
    strike_price = float(input("Enter the strike price: "))
    
    # 6. User picks premium price
    premium = float(input("Enter the option premium: "))
    
    # 7. User picks what they think the stock price will be
    projected_price = float(input("Enter your projected stock price at expiration: "))
    
    # 8. Option profit calculated
    profit = calculate_option_profit(option_type, strike_price, premium, projected_price)
    
    print(f"\nProjected profit: ${profit:.2f}")

if __name__ == "__main__":
    main()