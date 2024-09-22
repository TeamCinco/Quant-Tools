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

# Import Bayesian Optimization library
from bayes_opt import BayesianOptimization

def load_tickers(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return {item['ticker']: item['cik_str'] for item in data.values()}

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period='6mo')
    if hist.empty:
        print(f"No historical data for {ticker}.")
        return None
    return hist

def calculate_aroc(hist):
    daily_returns = hist['Close'].pct_change()
    aroc = daily_returns.mean() * 100  # Convert to percentage
    return aroc

def filter_by_market_cap(ticker, size):
    try:
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
    except Exception as e:
        print(f"Error fetching market cap for {ticker}: {e}")
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
    
    plt.savefig(r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\Results\screener 9.19.24\aroc_bell_curve.png")
    print("AROC bell curve chart 'aroc_bell_curve.png' has been created.")
    plt.show()

def save_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Ticker', 'Date', 'AROC', 'Sigma Range', 'Percent Change'])
        writer.writeheader()
        for row in data:
            writer.writerow(row)
def plot_aroc_with_signals(ticker, hist, window=20):
    # Calculate rolling AROC
    returns = hist['Close'].pct_change()
    rolling_aroc = returns.rolling(window=window).mean() * 100 * 252  # Annualized

    # Calculate mean and standard deviation of AROC
    mean_aroc = rolling_aroc.mean()
    std_aroc = rolling_aroc.std()

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(hist.index, rolling_aroc, label='Rolling AROC')
    plt.axhline(y=mean_aroc, color='r', linestyle='--', label='Mean AROC')
    plt.axhline(y=mean_aroc + std_aroc, color='g', linestyle=':', label='+1 Std Dev')
    plt.axhline(y=mean_aroc - std_aroc, color='g', linestyle=':', label='-1 Std Dev')

    # Add buy/sell signals
    buy_signals = rolling_aroc[rolling_aroc < (mean_aroc - std_aroc)]
    sell_signals = rolling_aroc[rolling_aroc > (mean_aroc + std_aroc)]
    
    plt.scatter(buy_signals.index, buy_signals, color='green', marker='^', s=100, label='Buy Signal')
    plt.scatter(sell_signals.index, sell_signals, color='red', marker='v', s=100, label='Sell Signal')

    plt.title(f'{ticker} Rolling AROC with Trading Signals')
    plt.xlabel('Date')
    plt.ylabel('AROC (%)')
    plt.legend()
    plt.grid(True)
    plt.show()
# Updated trading strategy implementation
def trading_strategy(hist, lower_sigma, upper_sigma):
    prices = hist['Close']
    mean_price = prices.mean()
    std_price = prices.std()

    lower_threshold = mean_price - lower_sigma * std_price
    upper_threshold = mean_price + upper_sigma * std_price

    position = 0  # 0: No position, 1: Long position
    entry_price = 0
    returns = []

    for date, price in prices.items():  # Changed from iteritems() to items()
        if position == 0 and price < lower_threshold:
            # Buy signal
            position = 1
            entry_price = price
        elif position == 1 and price > upper_threshold:
            # Sell signal
            position = 0
            exit_price = price
            return_pct = (exit_price - entry_price) / entry_price
            returns.append(return_pct)

    # Close any open position at the end
    if position == 1:
        exit_price = prices.iloc[-1]
        return_pct = (exit_price - entry_price) / entry_price
        returns.append(return_pct)

    if returns:
        cumulative_return = np.prod([1 + r for r in returns]) - 1
        return cumulative_return
    else:
        return 0

def optimize_strategy(ticker, hist):
    def objective_function(lower_sigma, upper_sigma):
        # Ensure lower_sigma < upper_sigma
        if lower_sigma >= upper_sigma:
            return -1  # Penalize invalid parameter combinations

        cumulative_return = trading_strategy(hist, lower_sigma, upper_sigma)
        return cumulative_return

    # Define the parameter bounds
    pbounds = {
        'lower_sigma': (0.5, 3),
        'upper_sigma': (0.5, 3)
    }

    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        verbose=0,  # Set to 1 to see optimization steps
        random_state=42,
    )

    optimizer.maximize(
        init_points=5,
        n_iter=15,
    )

    best_params = optimizer.max['params']
    best_return = optimizer.max['target']

    print(f"Optimal parameters for {ticker}:")
    print(f"Lower Sigma: {best_params['lower_sigma']:.2f}")
    print(f"Upper Sigma: {best_params['upper_sigma']:.2f}")
    print(f"Expected Cumulative Return: {best_return*100:.2f}%\n")

    return best_params, best_return

def main():
    tickers = load_tickers(r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\FILES FOR SCRIPTS\TICKERs\tickers.json")
    size_filter = input("Enter market cap size to filter by (small, medium, large): ").lower()
    num_stocks = int(input("Enter the number of stocks to analyze: "))

    selected_tickers = random.sample(list(tickers.items()), min(num_stocks, len(tickers)))

    stock_data_list = []
    aroc_list = []

    # Store optimization results
    optimization_results = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_ticker, ticker, cik, size_filter): ticker for ticker, cik in selected_tickers}

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                stock_data, aroc, hist = result
                stock_data_list.append(stock_data)
                aroc_list.append(aroc)

                # Optimize the trading strategy for this ticker
                best_params, best_return = optimize_strategy(stock_data['Ticker'], hist)

                optimization_results.append({
                    'Ticker': stock_data['Ticker'],
                    'Lower Sigma': best_params['lower_sigma'],
                    'Upper Sigma': best_params['upper_sigma'],
                    'Expected Return (%)': best_return * 100
                })

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
    csv_filename = r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\Results\screener 9.19.24\stock_analysis_results_9.19.24.csv"
    save_to_csv(csv_data, csv_filename)
    print(f"Data saved to {csv_filename}")

    # Save optimization results to CSV
    optimization_csv = r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\Results\screener 9.19.24\optimization_results_9.19.24.csv"
    with open(optimization_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Ticker', 'Lower Sigma', 'Upper Sigma', 'Expected Return (%)'])
        writer.writeheader()
        for row in optimization_results:
            writer.writerow(row)
    print(f"Optimization results saved to {optimization_csv}")

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

        # Optimize strategy for the comparison ticker
        best_params, best_return = optimize_strategy(comparison_ticker, comparison_hist)

if __name__ == "__main__":
    main()