import json
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def load_tickers(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return {item['ticker']: item['cik_str'] for item in data.values()}

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period='1mo')  # Get data for the past month
    if hist.empty:
        print(f"No data found for {ticker}; possibly delisted or no recent trading activity.")
        return None
    hist = hist.sort_index()
    return hist

def calculate_daily_percentage_changes(hist):
    open_prices = hist['Open']
    close_prices = hist['Close']
    # Avoid division by zero
    valid_mask = open_prices != 0
    if not valid_mask.any():
        return pd.Series(dtype=float)  # Return an empty series if all opens are zero
    daily_pct_change = ((close_prices[valid_mask] - open_prices[valid_mask]) / open_prices[valid_mask]) * 100
    return daily_pct_change

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

def main():
    tickers = load_tickers('/Users/jazzhashzzz/Desktop/Quant-Tools/Stock Performance Analysis Tools/screener tools/tickers.json')
    size_filter = input("Enter market cap size to filter by (small, medium, large): ").lower()
    
    stock_data_list = []
    aroc_list = []

    for ticker, cik in tickers.items():
        try:
            if not filter_by_market_cap(ticker, size_filter):
                print(f"{ticker} does not match the market cap size '{size_filter}'; skipping.")
                continue

            hist = get_stock_data(ticker)
            if hist is None:
                continue

            daily_pct_change = calculate_daily_percentage_changes(hist)
            if daily_pct_change.empty or not np.isfinite(daily_pct_change).all():
                print(f"Invalid data for {ticker}; skipping.")
                continue

            aroc = daily_pct_change.mean()
            if not np.isfinite(aroc):
                print(f"Invalid AROC calculated for {ticker}; skipping.")
                continue

            stock_data = {
                'Ticker': ticker,
                'Dates': hist.index.strftime('%Y-%m-%d').tolist(),
                'AROC': aroc
            }
            stock_data_list.append(stock_data)
            aroc_list.append(aroc)

            print(f"Processed {ticker} (CIK: {cik})")
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")

    if len(aroc_list) == 0:
        print("No valid AROC data available.")
        return

    aroc_array = np.array(aroc_list)
    if not np.isfinite(aroc_array).all():
        print("Invalid values in AROC array; cannot perform analysis.")
        return

    mean_aroc = np.mean(aroc_array)
    population_variance = np.var(aroc_array)
    std_deviation = np.sqrt(population_variance)

    if std_deviation == 0 or np.isnan(std_deviation):
        print("Standard deviation is zero or NaN; cannot perform analysis.")
        return

    standardized_arocs = (aroc_array - mean_aroc) / std_deviation
    std_ranges = []
    directions = []
    for std_value in standardized_arocs:
        abs_std = abs(std_value)
        if abs_std > 3:
            std_range = 'Beyond 3σ'
        elif abs_std > 2:
            std_range = 'Beyond 2σ'
        elif abs_std > 1:
            std_range = 'Beyond 1σ'
        else:
            std_range = 'Within 1σ'
        std_ranges.append(std_range)
        direction = 'Positive' if std_value > 0 else 'Negative'
        directions.append(direction)

    results_list = []
    for i, stock_data in enumerate(stock_data_list):
        ticker = stock_data['Ticker']
        dates = '; '.join(stock_data['Dates'])
        standardized_aroc = standardized_arocs[i]
        std_range = std_ranges[i]
        direction = directions[i]

        result = {
            'Ticker': ticker,
            'Dates': dates,
            'Standardized_AROC': standardized_aroc,
            'STD_Range': std_range,
            'Direction': direction
        }
        results_list.append(result)

    results_df = pd.DataFrame(results_list)
    results_df.to_csv('stock_std_analysis.csv', index=False)
    print("CSV file 'stock_std_analysis.csv' has been created.")

    plt.figure(figsize=(10, 6))
    count, bins, ignored = plt.hist(aroc_array, bins=30, density=True, alpha=0.6, color='g')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mean_aroc, std_deviation)
    plt.plot(x, p, 'k', linewidth=2)

    plt.axvline(mean_aroc, color='r', linestyle='dashed', linewidth=1.5, label='Mean')
    plt.axvline(mean_aroc + std_deviation, color='b', linestyle='dashed', linewidth=1, label='±1σ')
    plt.axvline(mean_aroc - std_deviation, color='b', linestyle='dashed', linewidth=1)
    plt.axvline(mean_aroc + 2*std_deviation, color='y', linestyle='dashed', linewidth=1, label='±2σ')
    plt.axvline(mean_aroc - 2*std_deviation, color='y', linestyle='dashed', linewidth=1)
    plt.axvline(mean_aroc + 3*std_deviation, color='m', linestyle='dashed', linewidth=1, label='±3σ')
    plt.axvline(mean_aroc - 3*std_deviation, color='m', linestyle='dashed', linewidth=1)

    plt.title('Distribution of AROC Values')
    plt.xlabel('AROC (%)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig('aroc_distribution.png')
    print("Bell curve graph 'aroc_distribution.png' has been created.")

if __name__ == "__main__":
    main()
