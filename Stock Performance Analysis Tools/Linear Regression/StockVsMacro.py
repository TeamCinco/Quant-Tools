import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import holidays

def get_previous_business_day(date):
    while date.weekday() >= 5 or date in holidays.US():
        date -= timedelta(days=1)
    return date

def get_stock_data(tickers, start, end):
    stock_data = {}
    for ticker in tickers:
        stock = yf.download(ticker, start=start, end=end)
        stock_data[ticker] = stock['Close']
    return stock_data

def get_macro_data(indicators, start, end):
    macro_data = {}
    for indicator in indicators:
        macro_data[indicator] = web.DataReader(indicator, 'fred', start, end)
    return macro_data

def plot_individual_data(data, title_prefix):
    for name, series in data.items():
        plt.figure(figsize=(14, 7))
        plt.plot(series.index, series)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title(f'{title_prefix}: {name}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def main():
    stock_tickers = input("Enter stock tickers (comma-separated): ").upper().replace(" ", "").split(',')
    macro_indicators = input("Enter macroeconomic indicator tickers (comma-separated): ").upper().replace(" ", "").split(',')
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    start = get_previous_business_day(start)
    end = get_previous_business_day(end)

    stock_data = get_stock_data(stock_tickers, start, end)
    macro_data = get_macro_data(macro_indicators, start, end)

    plot_individual_data(stock_data, "Stock Price")
    plot_individual_data(macro_data, "Macroeconomic Indicator")

if __name__ == "__main__":
    main()