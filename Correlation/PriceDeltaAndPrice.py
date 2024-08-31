import yfinance as yf
import pandas_datareader as pdr
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def get_stock_data(ticker, start_date, end_date):
    """Retrieve stock data from yfinance."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date)['Close']
        return data
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return None

def get_fred_data(symbol, start_date, end_date):
    """Retrieve economic data from FRED using pandas_datareader."""
    try:
        data = pdr.get_data_fred(symbol, start=start_date, end=end_date)
        return data
    except Exception as e:
        print(f"Error fetching FRED data for {symbol}: {e}")
        return None

def plot_triangular_heatmap(correlation_matrix):
    """Plot a triangular correlation heatmap."""
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        fmt='.2f',
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .8}
    )
    plt.title('Triangle Correlation Heatmap (Focus on SPY)', fontsize=16)
    plt.tight_layout()
    plt.show()

def calculate_daily_percent_change(data):
    """Calculate daily percent change."""
    return data.pct_change().dropna()

def main():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)  # 5 years of data

    print("Enter stock symbols (SPY will be included automatically):")
    stock_symbols = input("Separate symbols by commas: ").split(',')
    stock_symbols = ['SPY'] + [symbol.strip().upper() for symbol in stock_symbols if symbol.strip()]
    
    print("Enter FRED symbols for economic indicators:")
    fred_symbols = input("Separate symbols by commas: ").split(',')
    fred_symbols = [symbol.strip().upper() for symbol in fred_symbols if symbol.strip()]

    stock_data = {}
    fred_data = {}

    for symbol in stock_symbols:
        data = get_stock_data(symbol, start_date, end_date)
        if data is not None:
            stock_data[symbol] = data

    for symbol in fred_symbols:
        data = get_fred_data(symbol, start_date, end_date)
        if data is not None and data.shape[1] == 1:
            fred_data[symbol] = data.iloc[:, 0]

    if stock_data or fred_data:
        # Combine stock and FRED data
        combined_data = pd.DataFrame({**stock_data, **fred_data})
        combined_data.dropna(inplace=True)

        if not combined_data.empty:
            # Calculate daily percent changes for stocks
            stock_pct_changes = combined_data[stock_symbols].apply(calculate_daily_percent_change)
            
            # Combine percent changes for stocks with raw values for FRED data
            analysis_data = pd.concat([stock_pct_changes, combined_data[fred_symbols]], axis=1)
            
            correlation_matrix = analysis_data.corr(method='pearson')

            # Plot the triangular heatmap
            plot_triangular_heatmap(correlation_matrix)

            # Print correlations with SPY
            spy_correlations = correlation_matrix['SPY'].sort_values(ascending=False)
            print("\nCorrelations with SPY:")
            print(spy_correlations.drop('SPY'))
            
            print("\nBasic Statistics for SPY (based on daily percent changes):")
            spy_stats = stock_pct_changes['SPY'].describe()
            print(spy_stats)

            # Calculate std off of daily percent change from the maximum for SPY
            spy_max = combined_data['SPY'].max()
            spy_daily_pct_change_from_max = (combined_data['SPY'] - spy_max) / spy_max * 100
            std_from_max = spy_daily_pct_change_from_max.std()
            
            print("\nStandard Deviation of Daily Percent Change from Maximum for SPY:")
            print(f"{std_from_max:.6f}%")

            # Print basic statistics for FRED symbols (raw values)
            for symbol in fred_symbols:
                print(f"\nBasic Statistics for {symbol} (raw values):")
                print(combined_data[symbol].describe())

        else:
            print("No valid data available for analysis after alignment.")
    else:
        print("No valid data retrieved.")

if __name__ == "__main__":
    main()