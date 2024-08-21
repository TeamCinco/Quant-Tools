import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tempfile

def save_plot_to_file(plt):
    temp_plot_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_plot_file.name)
    plt.close()
    return temp_plot_file.name



def calculate_and_display_quarterly_prices(latest_close, max_change_pct):
    quarter_increments = np.arange(0.25, max_change_pct + 0.25, 0.25)
    for increment in quarter_increments:
        expected_price_positive, expected_price_negative = calculate_expected_prices(latest_close, increment)
        print(f"At {increment}% change: +{increment}% -> {expected_price_positive:.2f}, -{increment}% -> {expected_price_negative:.2f}")

def calculate_expected_prices(latest_close, expected_change_pct):
    expected_price_positive = latest_close * (1 + expected_change_pct / 100)
    expected_price_negative = latest_close * (1 - expected_change_pct / 100)
    return expected_price_positive, expected_price_negative

# Prompt the user to input the ticker symbol
ticker_symbol = input("Please enter the ticker symbol: ")

# Fetch 6 months of data for the entered ticker
data = yf.download(ticker_symbol, period='6mo')

# Calculate daily percentage change from open to close
data['Daily Change %'] = ((data['Close'] - data['Open']) / data['Open']) * 100

# Define bins for the daily range in percentages, including finer bins
bins = np.concatenate([
    np.arange(-4, 0, 1),
    np.arange(0, 5, 1)
])
bin_labels = [f"{bins[i]:.1f}% to {bins[i+1]:.1f}%" for i in range(len(bins)-1)]

# Categorize daily changes into bins
data['Bin'] = pd.cut(data['Daily Change %'], bins=bins, labels=bin_labels, include_lowest=True)

# Create frequency table
frequency_counts = data['Bin'].value_counts(sort=False)
frequency_table = pd.DataFrame({
    'Bins': frequency_counts.index.categories,
    'Qty': frequency_counts.values
})

# Calculate percentage and cumulative percentage
frequency_table['Qty%'] = (frequency_table['Qty'] / frequency_table['Qty'].sum()) * 100
frequency_table['Cum%'] = frequency_table['Qty%'].cumsum()

# Sort the frequency table by bins
frequency_table.sort_values(by='Bins', inplace=True)

# Format the frequency table
frequency_table['Qty%'] = frequency_table['Qty%'].map('{:.2f}%'.format)
frequency_table['Cum%'] = frequency_table['Cum%'].map('{:.2f}%'.format)

# Print the formatted frequency table
print(frequency_table.to_string(index=False))

# Plotting the bar graph
plt.figure(figsize=(12, 8))  # Adjust the figure size to your preference

# Create horizontal bars
plt.barh(np.arange(len(frequency_table)), frequency_table['Qty'], color='blue', edgecolor='black')

# Add titles and labels
plt.xlabel('Frequency')
plt.ylabel('Daily Change in %')
plt.title(f'Daily Change in Percentage from Open to Close, past 6 months - {ticker_symbol}')

# Add the bin labels as y-ticks
plt.yticks(ticks=np.arange(len(frequency_table)), labels=frequency_table['Bins'])

# Invert the y-axis to have the highest bin at the top
plt.gca().invert_yaxis()

# Add grid lines for better readability
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()

# Calculate statistics
stats_data = {
    'Up Days': ((data['Close'] > data['Open']).sum(), f"{data[data['Close'] > data['Open']]['Daily Change %'].max():.2f}%"),
    'Down Days': ((data['Close'] < data['Open']).sum(), f"{data[data['Close'] < data['Open']]['Daily Change %'].min():.2f}%"),
    'Average': (data['Daily Change %'].mean(),),
    'ST DEV': (data['Daily Change %'].std(),),
    'Variance': (data['Daily Change %'].var(),),
    'Max': (data['Daily Change %'].max(),),
    'Min': (data['Daily Change %'].min(),)
}

# Create the statistics DataFrame
stats_df = pd.DataFrame(stats_data, index=['Value', 'Percent' if 'Percent' in stats_data else '']).T
stats_df['Value'] = stats_df['Value'].astype(float).map('{:.2f}'.format)
stats_df = stats_df.reset_index().rename(columns={'index': 'Statistic'})

# Print the statistics table
print(stats_df.to_string(index=False, header=False))

# Calculate net changes and net percent changes
data['Net Δ'] = data['Close'] - data['Open']
data['Net Δ %'] = ((data['Close'] - data['Open']) / data['Open']) * 100

# Create a DataFrame with the date, open, high, low, close, net change, and net percent change
price_history_df = data[['Open', 'High', 'Low', 'Close', 'Net Δ', 'Net Δ %']].copy()
price_history_df.reset_index(inplace=True)  # Reset index to bring the Date into the columns
price_history_df['Date'] = price_history_df['Date'].dt.strftime('%m/%d/%y')  # Format date

# Format the DataFrame to show two decimal places for prices and changes, and add % symbol to the Net Δ %
price_history_df['Open'] = price_history_df['Open'].map('{:.2f}'.format)
price_history_df['High'] = price_history_df['High'].map('{:.2f}'.format)
price_history_df['Low'] = price_history_df['Low'].map('{:.2f}'.format)
price_history_df['Close'] = price_history_df['Close'].map('{:.2f}'.format)
price_history_df['Net Δ'] = price_history_df['Net Δ'].map('{:.2f}'.format)
price_history_df['Net Δ %'] = price_history_df['Net Δ %'].map('{:.2f}%'.format)

# Print the new DataFrame
print(price_history_df.to_string(index=False))

# Calculate expected stock prices for quarter increments
print("\nCalculating Expected Stock Prices for Quarter Increments\n" + "-"*60)
try:
    expected_change_pct = abs(float(input("Enter expected absolute % change: ")))  # Example: 5 for both +5% and -5%
    latest_close = data['Open'].iloc[-1]
    expected_price_positive, expected_price_negative = calculate_expected_prices(latest_close, expected_change_pct)
    max_change_pct = expected_change_pct

    calculate_and_display_quarterly_prices(latest_close, max_change_pct)
    print(f"Based on the latest close price of {latest_close:.2f} and an expected absolute change of {expected_change_pct}%,")
    print(f"the expected price for a positive change is {expected_price_positive:.2f},")
    print(f"and for a negative change is {expected_price_negative:.2f}.")
except ValueError:
    print("Invalid input. Please enter a valid number for the expected absolute % change.")
