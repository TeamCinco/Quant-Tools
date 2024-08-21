import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.drawing.image import Image
import tempfile

def save_plot_to_file(plt):
    temp_plot_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_plot_file.name)
    plt.close()
    return temp_plot_file.name

def append_df_to_ws(ws, df, include_index=False):
    for r in dataframe_to_rows(df, index=include_index, header=True):
        ws.append(r)

def calculate_and_display_quarterly_prices(latest_close, max_change_pct, std):
    """
    Calculates and displays the expected prices based on the latest close price and quarter increments
    of an expected percentage change up to the maximum change, for both positive and negative directions.
    
    Parameters:
    - latest_close: The most recent close price of the stock.
    - max_change_pct: The maximum expected percentage change in price (as an absolute value).
    - std: The standard deviation of daily percentage changes.
    """
    quarter_increments = np.arange(0.25, max_change_pct + 0.25, 0.25)
    expected_prices = []

    for increment in quarter_increments:
        expected_price_positive, expected_price_negative = calculate_expected_prices(latest_close, increment)
        expected_prices.append((increment, expected_price_positive, expected_price_negative))

    plot_expected_prices(latest_close, expected_prices, std)

def calculate_expected_prices(latest_close, expected_change_pct):
    """
    Calculates the expected prices based on the latest close price and an expected percentage change,
    for both positive and negative directions.
    
    Parameters:
    - latest_close: The most recent close price of the stock.
    - expected_change_pct: The expected percentage change in price (as an absolute value).
    
    Returns:
    - A tuple containing the expected prices for both positive and negative percentage changes.
    """
    expected_price_positive = latest_close * (1 + expected_change_pct / 100)
    expected_price_negative = latest_close * (1 - expected_change_pct / 100)
    return expected_price_positive, expected_price_negative

def plot_expected_prices(latest_close, expected_prices, std):
    """
    Plots the expected prices based on quarter increments of the expected percentage change.
    
    Parameters:
    - latest_close: The most recent close price of the stock.
    - expected_prices: A list of tuples containing the increment, expected positive price, and expected negative price.
    - std: The standard deviation of daily percentage changes.
    """
    increments, pos_prices, neg_prices = zip(*expected_prices)

    plt.figure(figsize=(12, 8))

    # Plot positive and negative expected prices
    plt.plot(increments, pos_prices, marker='o', color='g', label='Positive Expected Prices')
    plt.plot(increments, neg_prices, marker='o', color='r', label='Negative Expected Prices')

    # Plot latest close price
    plt.axhline(y=latest_close, color='b', linestyle='--', label=f'Latest Close Price: {latest_close:.2f}')
    
    # Plot standard deviation lines
    plt.axhline(y=latest_close + std, color='c', linestyle='--', label=f'+1 STD: {latest_close + std:.2f}')
    plt.axhline(y=latest_close - std, color='c', linestyle='--', label=f'-1 STD: {latest_close - std:.2f}')

    plt.xlabel('Expected Change (%)')
    plt.ylabel('Stock Price')
    plt.title('Expected Stock Prices for Quarter Increments of Expected Percentage Change')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Prompt the user to input the ticker symbol
ticker_symbol = input("Please enter the ticker symbol: ")

# Fetch 6 months of data for the entered ticker
data = yf.download(ticker_symbol, period='6mo')

# Calculate daily percentage change from open to close
data['Daily Change %'] = ((data['Close'] - data['Open']) / data['Open']) * 100

# Define bins for the daily range in percentages
bins = np.arange(-3, 3.5, 0.5).tolist()  # Adjust the bin ranges as required
bin_labels = [f"{bins[i]:.1f}% to {bins[i+1]:.1f}%" for i in range(len(bins)-1)]

# Categorize daily changes into bins
data['Bin'] = pd.cut(data['Daily Change %'], bins=bins, labels=bin_labels, include_lowest=True)

# Create frequency table
frequency_counts = data['Bin'].value_counts(sort=False)
frequency_table = pd.DataFrame({
    'Bins': frequency_counts.index.categories,
    'Qty': frequency_counts.values
})

# Calculate percentage, cumulative percentage, and probability
frequency_table['Qty%'] = (frequency_table['Qty'] / frequency_table['Qty'].sum()) * 100
frequency_table['Cum%'] = frequency_table['Qty%'].cumsum()
frequency_table['Probability'] = frequency_table['Qty'] / frequency_table['Qty'].sum()

# Sort the frequency table by bins
frequency_table.sort_values(by='Bins', inplace=True)

# Calculate the midpoint of each bin
def calculate_midpoint(bin_str):
    lower, upper = bin_str.split(" to ")
    lower = float(lower.replace('%', ''))
    upper = float(upper.replace('%', ''))
    return (lower + upper) / 2

frequency_table['Midpoint'] = frequency_table['Bins'].apply(calculate_midpoint)

# Calculate the weighted average of the midpoints
weighted_average = (frequency_table['Midpoint'] * frequency_table['Probability']).sum()

# Format the frequency table
frequency_table['Qty%'] = frequency_table['Qty%'].map('{:.2f}%'.format)
frequency_table['Cum%'] = frequency_table['Cum%'].map('{:.2f}%'.format)
frequency_table['Probability'] = frequency_table['Probability'].map('{:.4f}'.format)

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
    'STD': (data['Daily Change %'].std(),),
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

# Calculate the standard deviation of daily percentage changes
std = data['Daily Change %'].std()

# Use the weighted average as the expected absolute % change
expected_change_pct = abs(weighted_average)
latest_close = data['Open'].iloc[-1]

# Calculate and display the expected prices for quarter increments
calculate_and_display_quarterly_prices(latest_close, expected_change_pct, std)
print(f"\nBased on the latest close price of {latest_close:.2f} and an expected absolute change of {expected_change_pct:.2f}%,")
print(f"the expected price for a positive change is {calculate_expected_prices(latest_close, expected_change_pct)[0]:.2f},")
print(f"and for a negative change is {calculate_expected_prices(latest_close, expected_change_pct)[1]:.2f}.")
