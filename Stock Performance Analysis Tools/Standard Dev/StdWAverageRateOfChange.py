import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import scipy.stats as stats

def save_plot_to_file(plt):
    temp_plot_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_plot_file.name)
    plt.close()
    return temp_plot_file.name

ticker_symbol = input("Please enter the ticker symbol: ")
data = yf.download(ticker_symbol, period='max')

# Calculate Mid price as the average of High and Low
data['Mid'] = (data['High'] + data['Low']) / 2

# Calculate the rate of change for each segment
data['Rate_Change_Low_Mid'] = ((data['Mid'] - data['Low']) / data['Low']) * 100
data['Rate_Change_Mid_High'] = ((data['High'] - data['Mid']) / data['Mid']) * 100
data['Rate_Change_Open_Close'] = ((data['Close'] - data['Open']) / data['Open']) * 100

# Calculate the average rate of change
data['Daily_Avg_Rate_Change'] = (data['Rate_Change_Low_Mid'] + data['Rate_Change_Mid_High'] + data['Rate_Change_Open_Close']) / 3

# Let's inspect the data to confirm the presence of negative changes
print(data['Daily_Avg_Rate_Change'].describe())  # Print statistical summary
print(data['Daily_Avg_Rate_Change'].head(10))    # Print first 10 values for quick inspection

# Calculate rolling averages for weekly and monthly average rate of change
data['Weekly_Avg_Rate_Change'] = data['Daily_Avg_Rate_Change'].rolling(window=5).mean()
data['Monthly_Avg_Rate_Change'] = data['Daily_Avg_Rate_Change'].rolling(window=21).mean()

# Adjust the binning to ensure negative values are adequately captured
bins = np.arange(-5, 5.5, 0.5).tolist()  # Widen the range to better capture the negative values
data['Bin'] = pd.cut(data['Daily_Avg_Rate_Change'], bins=bins, include_lowest=True)
frequency_table = pd.DataFrame({
    'Bins': data['Bin'].value_counts(sort=False).index.categories,
    'Qty': data['Bin'].value_counts(sort=False).values
})
frequency_table['Qty%'] = (frequency_table['Qty'] / frequency_table['Qty'].sum()) * 100
frequency_table['Cum%'] = frequency_table['Qty%'].cumsum()
frequency_table.sort_values(by='Bins', inplace=True)
frequency_table['Qty%'] = frequency_table['Qty%'].map('{:.2f}%'.format)
frequency_table['Cum%'] = frequency_table['Cum%'].map('{:.2f}%'.format)

print(frequency_table.to_string(index=False))

plt.figure(figsize=(12, 8))
plt.barh(frequency_table['Bins'].astype(str), frequency_table['Qty'], color='blue', edgecolor='black')
plt.xlabel('Frequency')
plt.ylabel('Daily Average Rate of Change in %')
plt.title(f'Daily Average Rate of Change, all available data - {ticker_symbol}')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
