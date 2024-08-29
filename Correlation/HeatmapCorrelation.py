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

def main():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)  # 5 years of data

    print("Enter additional stock symbols (SPY will be included automatically):")
    stock_symbols = input("Separate symbols by commas: ").split(',')
    stock_symbols = ['SPY'] + [symbol.strip().upper() for symbol in stock_symbols if symbol.strip()]
    
    print("Enter FRED symbols for economic indicators:")
    fred_symbols = input("Separate symbols by commas: ").split(',')
    fred_symbols = [symbol.strip().upper() for symbol in fred_symbols if symbol.strip()]

    data_sources = {}

    for symbol in stock_symbols:
        stock_data = get_stock_data(symbol, start_date, end_date)
        if stock_data is not None:
            data_sources[symbol] = stock_data

    for symbol in fred_symbols:
        fred_data = get_fred_data(symbol, start_date, end_date)
        if fred_data is not None and fred_data.shape[1] == 1:
            data_sources[symbol] = fred_data.iloc[:, 0]

    if data_sources:
        combined_data = pd.DataFrame(data_sources)
        combined_data.dropna(inplace=True)

        if not combined_data.empty:
            correlation_matrix = combined_data.corr()

            # Plot the triangular heatmap
            plot_triangular_heatmap(correlation_matrix)

            # Print correlations with SPY
            spy_correlations = correlation_matrix['SPY'].sort_values(ascending=False)
            print("\nCorrelations with SPY:")
            print(spy_correlations.drop('SPY'))
            
            print("\nBasic Statistics for SPY:")
            print(combined_data['SPY'].describe())

        else:
            print("No valid data available for correlation analysis after alignment.")
    else:
        print("No valid data retrieved.")

if __name__ == "__main__":
    main()