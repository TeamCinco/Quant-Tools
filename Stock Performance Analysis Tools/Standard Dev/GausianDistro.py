import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_trading_days_for_months(months):
    trading_days_per_year = 252
    trading_days_per_month = trading_days_per_year / 12
    return int(trading_days_per_month * months)

# User inputs
ticker_symbol = input("Please enter the ticker symbol: ")
months = int(input("Please enter the number of months to calculate STD for: "))
trading_days = get_trading_days_for_months(months)

# Downloading data for the last 6 months
data = yf.download(ticker_symbol)

# Calculating daily change percentage
data['Daily Change %'] = ((data['Close'] - data['Open']) / data['Open']) * 100

# Binning data
bins = np.arange(-3, 3.5, 0.5).tolist()
data['Bin'] = pd.cut(data['Daily Change %'], bins=bins, include_lowest=True)
frequency_table = pd.DataFrame({
    'Bins': data['Bin'].value_counts(sort=False).index.categories,
    'Qty': data['Bin'].value_counts(sort=False).values
})

# Plotting the histogram
plt.figure(figsize=(12, 8))
plt.barh(frequency_table['Bins'].astype(str), frequency_table['Qty'], color='blue', edgecolor='black')
plt.xlabel('Frequency')
plt.ylabel('Daily Change in %')
plt.title(f'Daily Change in Percentage from Open to Close, past {months} months - {ticker_symbol}')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
