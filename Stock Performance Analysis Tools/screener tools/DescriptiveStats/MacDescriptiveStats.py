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

def load_tickers(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return {item['ticker']: item['cik_str'] for item in data.values()}

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period='1mo')
    return hist

def calculate_aroc(hist):
    daily_returns = hist['Close'].pct_change()
    aroc = daily_returns.mean() * 100  # Convert to percentage
    return aroc

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

        start_price = hist['Open'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        percent_change = ((end_price - start_price) / start_price) * 100

        stock_data = {
            'Ticker': ticker,
            'Date': hist.index[-1].strftime('%Y-%m-%d'),
            'AROC': aroc,
            'Start Price': start_price,
            'End Price': end_price,
            'Percent Change': percent_change
        }

        print(f"Processed {ticker} (CIK: {cik})")
        return stock_data, aroc, hist

    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")
        return None

def plot_trendline_and_std_ranges(hist, mean_aroc, std_deviation, mean_price, std_prices, ticker):
    plt.figure(figsize=(10, 6))
    x = np.arange(len(hist))
    y = hist['Close']

    plt.plot(hist.index, y, label='Closing Price', color='blue')

    slope, intercept, _, _, _ = stats.linregress(x, y)
    plt.plot(hist.index, slope * x + intercept, color='green', label='Trendline')

    plt.plot(hist.index, mean_price, color='orange', label='Mean')
    plt.plot(hist.index, mean_price + std_prices, color='yellow', linestyle='--', label='+1σ')
    plt.plot(hist.index, mean_price - std_prices, color='yellow', linestyle='--', label='-1σ')
    plt.plot(hist.index, mean_price + 2 * std_prices, color='red', linestyle='-.', label='+2σ')
    plt.plot(hist.index, mean_price - 2 * std_prices, color='red', linestyle='-.', label='-2σ')

    plt.title(f'{ticker} Stock Price with Trendline and Standard Deviation Ranges')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'/Users/jazzhashzzz/Desktop/data for scripts/results/screener/{ticker}_trendline_std_chart.png')
    print(f"Trendline and standard deviation chart '{ticker}_trendline_std_chart.png' has been created.")
    plt.show()

def plot_aroc_bell_curve(aroc_list):
    plt.figure(figsize=(10, 6))
    
    mean_aroc = np.mean(aroc_list)
    std_dev_aroc = np.std(aroc_list)
    
    x = np.linspace(mean_aroc - 4*std_dev_aroc, mean_aroc + 4*std_dev_aroc, 100)
    y = stats.norm.pdf(x, mean_aroc, std_dev_aroc)
    
    plt.plot(x, y, 'b-', label='Normal Distribution')
    plt.hist(aroc_list, bins=50, density=True, alpha=0.7, color='g', label='AROC Distribution')
    
    plt.title('AROC Distribution with Normal Curve')
    plt.xlabel('AROC (%)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('/Users/jazzhashzzz/Desktop/data for scripts/results/screener/aroc_bell_curve.png')
    print("AROC bell curve chart 'aroc_bell_curve.png' has been created.")
    plt.show()



def plot_trendline_and_std_ranges(hist, mean_aroc, std_deviation, ticker):
    plt.figure(figsize=(10, 6))
    x = np.arange(len(hist))
    y = hist['Close']

    plt.plot(hist.index, y, label='Closing Price', color='blue')

    # Calculate and plot trendline
    slope, intercept, _, _, _ = stats.linregress(x, y)
    plt.plot(hist.index, slope * x + intercept, color='green', label='Trendline')

    # Calculate mean and standard deviation of the closing prices
    mean_price = y.mean()
    std_price = y.std()

    # Plot mean and standard deviation lines
    plt.axhline(y=mean_price, color='orange', linestyle='-', label='Mean Price')
    plt.axhline(y=mean_price + std_price, color='yellow', linestyle='--', label='+1σ Price')
    plt.axhline(y=mean_price - std_price, color='yellow', linestyle='--', label='-1σ Price')
    plt.axhline(y=mean_price + 2*std_price, color='red', linestyle='-.', label='+2σ Price')
    plt.axhline(y=mean_price - 2*std_price, color='red', linestyle='-.', label='-2σ Price')

    plt.title(f'{ticker} Stock Price with Trendline and Standard Deviation Ranges')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'/Users/jazzhashzzz/Desktop/data for scripts/results/screener/{ticker}_trendline_std_chart.png')
    print(f"Trendline and standard deviation chart '{ticker}_trendline_std_chart.png' has been created.")
    plt.show()



def save_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Ticker', 'Date', 'AROC', 'Sigma Range', 'Percent Change'])
        writer.writeheader()
        for row in data:
            writer.writerow(row)

            

def main():
    tickers = load_tickers('/Users/jazzhashzzz/Desktop/data for scripts/tickers and rates/tickers.json')
    size_filter = input("Enter market cap size to filter by (small, medium, large): ").lower()
    num_stocks = int(input("Enter the number of stocks to analyze: "))

    selected_tickers = random.sample(list(tickers.items()), min(num_stocks, len(tickers)))

    stock_data_list = []
    aroc_list = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_ticker, ticker, cik, size_filter): ticker for ticker, cik in selected_tickers}

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                stock_data, aroc, _ = result
                stock_data_list.append(stock_data)
                aroc_list.append(aroc)

    if len(aroc_list) == 0:
        print("No valid AROC data available.")
        return

    mean_aroc = np.mean(aroc_list)
    std_deviation = np.std(aroc_list)

    # Prepare data for CSV
    csv_data = []
    for stock in stock_data_list:
        sigma_range = (stock['AROC'] - mean_aroc) / std_deviation
        csv_data.append({
            'Ticker': stock['Ticker'],
            'Date': stock['Date'],
            'AROC': stock['AROC'],
            'Sigma Range': sigma_range,
            'Percent Change': stock['Percent Change']
        })

    # Save to CSV
    csv_filename = '/Users/jazzhashzzz/Desktop/data for scripts/results/screener/stock_analysis_results.csv'
    save_to_csv(csv_data, csv_filename)
    print(f"Data saved to {csv_filename}")

    # Plot AROC bell curve
    plot_aroc_bell_curve(aroc_list)

    while True:
        comparison_ticker = input("Enter a ticker to analyze (or 'quit' to exit): ").upper()
        if comparison_ticker.lower() == 'quit':
            break

        comparison_hist = get_stock_data(comparison_ticker)
        if comparison_hist is None:
            print(f"No data found for {comparison_ticker}")
            continue

        comparison_aroc = calculate_aroc(comparison_hist)
        comparison_sigma = (comparison_aroc - mean_aroc) / std_deviation

        print(f"{comparison_ticker} AROC: {comparison_aroc:.2f}%")
        print(f"{comparison_ticker} Sigma: {comparison_sigma:.2f}")

        plot_trendline_and_std_ranges(comparison_hist, mean_aroc, std_deviation, comparison_ticker)

if __name__ == "__main__":
    main()