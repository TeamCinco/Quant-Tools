import json 
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import csv
import os
import sys

# Set default encoding to UTF-8 for console output
sys.stdout.reconfigure(encoding='utf-8')

def load_tickers(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return {item['ticker']: item['cik_str'] for item in data.values()}

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period='6mo')
    return hist

def calculate_aroc(hist):
    daily_returns = (hist['Open'] - hist['Close']) / hist['Open'] * 100
    aroc = daily_returns.mean()
    return aroc

def log_transform_aroc(aroc):
    # Add a small constant to avoid log(0) and handle negative values
    return np.log(np.abs(aroc) + 1) * np.sign(aroc)

def filter_by_market_cap(ticker, size):
    stock = yf.Ticker(ticker)
    info = stock.info
    market_cap = info.get('marketCap', None)

    if market_cap is None:
        print(f"Market cap data not available for {ticker}.")
        return False

    if size == 'small' and market_cap <= 2e9:
        return True
    elif size == 'medium' and 2e9 < market_cap <= 10e9:
        return True
    elif size == 'large' and market_cap > 10e9:
        return True
    else:
        return False

def process_ticker(ticker, cik, size_filter):
    try:
        if not filter_by_market_cap(ticker, size_filter):
            print(f"{ticker} does not match the market cap size '{size_filter}'; skipping.")
            return None

        hist = get_stock_data(ticker)
        if hist is None:
            return None

        aroc = calculate_aroc(hist)
        if not np.isfinite(aroc):
            print(f"Invalid AROC calculated for {ticker}; skipping.")
            return None

        log_aroc = log_transform_aroc(aroc)

        start_price = hist['Open'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        percent_change = ((end_price - start_price) / start_price) * 100

        stock_data = {
            'Ticker': ticker,
            'Date': hist.index[-1].strftime('%Y-%m-%d'),
            'AROC': aroc,
            'Log AROC': log_aroc,
            'Start Price': start_price,
            'End Price': end_price,
            'Percent Change': percent_change
        }

        print(f"Processed {ticker} (CIK: {cik})")
        return stock_data, aroc, log_aroc, hist

    except Exception as e:
        if '404' in str(e):
            print(f"Error processing {ticker}: {e} (Data not found)")
        else:
            print(f"Error processing {ticker}: {e}")
        return None

def plot_trendline_and_std_ranges(hist, mean_aroc, std_deviation, ticker):
    plt.figure(figsize=(10, 6))
    x = np.arange(len(hist))
    y = hist['Close']

    plt.plot(hist.index, y, label='Closing Price', color='blue')

    slope, intercept, _, _, _ = stats.linregress(x, y)
    plt.plot(hist.index, slope * x + intercept, color='green', label='Trendline')

    mean_price = np.mean(y)
    std_prices = np.std(y)

    plt.axhline(y=mean_price, color='orange', label='Mean')
    plt.axhline(y=mean_price + std_prices, color='yellow', linestyle='--', label='+1σ')
    plt.axhline(y=mean_price - std_prices, color='yellow', linestyle='--', label='-1σ')
    plt.axhline(y=mean_price + 2 * std_prices, color='red', linestyle='-.', label='+2σ')
    plt.axhline(y=mean_price - 2 * std_prices, color='red', linestyle='-.', label='-2σ')

    plt.title(f'{ticker} Stock Price with Trendline and Standard Deviation Ranges')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"C:\\Users\\cinco\\Desktop\\DATA FOR SCRIPTS\\Results\\screener 9.19.24\\{ticker}_trendline_std_chart.png")
    print(f"Trendline and standard deviation chart '{ticker}_trendline_std_chart.png' has been created.")
    plt.show()

def plot_aroc_bell_curve(aroc_list, log_aroc_list):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    # Original AROC distribution
    mean_aroc = np.mean(aroc_list)
    std_dev_aroc = np.std(aroc_list)
    
    x1 = np.linspace(mean_aroc - 4*std_dev_aroc, mean_aroc + 4*std_dev_aroc, 100)
    y1 = stats.norm.pdf(x1, mean_aroc, std_dev_aroc)
    
    ax1.plot(x1, y1, 'b-', label='Normal Distribution')
    ax1.hist(aroc_list, bins=50, density=True, alpha=0.7, color='g', label='AROC Distribution')
    
    ax1.set_title('Original AROC Distribution')
    ax1.set_xlabel('AROC (%)')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True)
    
    # Log-transformed AROC distribution
    mean_log_aroc = np.mean(log_aroc_list)
    std_dev_log_aroc = np.std(log_aroc_list)
    
    x2 = np.linspace(mean_log_aroc - 4*std_dev_log_aroc, mean_log_aroc + 4*std_dev_log_aroc, 100)
    y2 = stats.norm.pdf(x2, mean_log_aroc, std_dev_log_aroc)
    
    ax2.plot(x2, y2, 'b-', label='Normal Distribution')
    ax2.hist(log_aroc_list, bins=50, density=True, alpha=0.7, color='r', label='Log AROC Distribution')
    
    ax2.set_title('Log-transformed AROC Distribution')
    ax2.set_xlabel('Log AROC')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\Results\9.21.24\LargeCaparoc_bell_curves.png")
    print("AROC bell curve charts 'aroc_bell_curves.png' have been created.")
    plt.show()

def save_to_csv(data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Ticker', 'Date', 'AROC', 'Log AROC', 'Sigma', 'Log Sigma', 'Sigma Range', 'Percent Change', 'Start Price', 'End Price'])
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def main():
    tickers = load_tickers(r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\FILES FOR SCRIPTS\TICKERs\tickers.json")
    size_filter = input("Enter market cap size to filter by (small, medium, large): ").lower()
    num_stocks = int(input("Enter the number of stocks to analyze: "))

    selected_tickers = random.sample(list(tickers.items()), min(num_stocks, len(tickers)))

    stock_data_list = []
    aroc_list = []
    log_aroc_list = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_ticker, ticker, cik, size_filter): ticker for ticker, cik in selected_tickers}

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                stock_data, aroc, log_aroc, _ = result
                stock_data_list.append(stock_data)
                aroc_list.append(aroc)
                log_aroc_list.append(log_aroc)

    if len(aroc_list) == 0:
        print("No valid AROC data available.")
        return

    mean_aroc = np.mean(aroc_list)
    std_deviation = np.std(aroc_list)
    mean_log_aroc = np.mean(log_aroc_list)
    std_deviation_log = np.std(log_aroc_list)

    # Prepare data for CSV
    csv_data = []
    for stock in stock_data_list:
        sigma = (stock['AROC'] - mean_aroc) / std_deviation
        log_sigma = (stock['Log AROC'] - mean_log_aroc) / std_deviation_log
        sigma_range = 'Within 1σ' if abs(sigma) <= 1 else ('Within 2σ' if abs(sigma) <= 2 else 'Beyond 2σ')
        csv_data.append({
            'Ticker': stock['Ticker'],
            'Date': stock['Date'],
            'AROC': stock['AROC'],
            'Log AROC': stock['Log AROC'],
            'Sigma': sigma,
            'Log Sigma': log_sigma,
            'Sigma Range': sigma_range,
            'Percent Change': stock['Percent Change'],
            'Start Price': stock['Start Price'],
            'End Price': stock['End Price']
        })

    # Save to CSV
    csv_filename = r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\Results\9.21.24\LargeCap.csv"
    save_to_csv(csv_data, csv_filename)
    print(f"Data saved to {csv_filename}")

    # Plot AROC bell curves
    plot_aroc_bell_curve(aroc_list, log_aroc_list)

    while True:
        comparison_ticker = input("Enter a ticker to analyze (or 'quit' to exit): ").upper()
        if comparison_ticker.lower() == 'quit':
            break

        comparison_hist = get_stock_data(comparison_ticker)
        if comparison_hist is None:
            print(f"No data found for {comparison_ticker}")
            continue

        comparison_aroc = calculate_aroc(comparison_hist)
        comparison_log_aroc = log_transform_aroc(comparison_aroc)
        comparison_sigma = (comparison_aroc - mean_aroc) / std_deviation
        comparison_log_sigma = (comparison_log_aroc - mean_log_aroc) / std_deviation_log

        print(f"{comparison_ticker} AROC: {comparison_aroc:.2f}%")
        print(f"{comparison_ticker} Log AROC: {comparison_log_aroc:.2f}")
        print(f"{comparison_ticker} Sigma: {comparison_sigma:.2f}")
        print(f"{comparison_ticker} Log Sigma: {comparison_log_sigma:.2f}")

        plot_trendline_and_std_ranges(comparison_hist, mean_aroc, std_deviation, comparison_ticker)

if __name__ == "__main__":
    main()