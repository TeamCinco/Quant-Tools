import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import scipy.stats as stats
from datetime import datetime, timedelta

def save_plot_to_file(plt):
    temp_plot_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_plot_file.name)
    plt.close()
    return temp_plot_file.name

ticker_symbol = input("Please enter the ticker symbol: ")
data = yf.download(ticker_symbol, period='max')

# Check if the most recent data point is from today
today = datetime.now().date()
last_data_date = data.index[-1].date()

if last_data_date < today:
    # If the data is not from today, fetch the latest quote
    ticker = yf.Ticker(ticker_symbol)
    latest_data = ticker.history(period="1d")
    if not latest_data.empty:
        data = data.append(latest_data)

data['Daily Change %'] = ((data['Close'] - data['Open']) / data['Open']) * 100
bins = np.arange(-3, 3.5, 0.5).tolist()
data['Bin'] = pd.cut(data['Daily Change %'], bins=bins, include_lowest=True)
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
plt.ylabel('Daily Change in %')
plt.title(f'Daily Change in Percentage from Open to Close, all available data - {ticker_symbol}')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Calculate the estimated population mean and variance
data['Daily_Price_Difference'] = data['Close'] - data['Open']
data['Weekly_Price_Difference'] = data['Close'] - data['Open'].shift(4)
data['Monthly_Price_Difference'] = data['Close'] - data['Open'].shift(19)

# Function to calculate estimated population variance and standard deviation
def estimated_population_variance(series):
    n = len(series.dropna())
    mean = np.mean(series)
    variance = np.sum((series - mean) ** 2) / n
    return variance

def estimated_population_std(series):
    return np.sqrt(estimated_population_variance(series))

daily_std = estimated_population_std(data['Daily_Price_Difference'])
weekly_std = estimated_population_std(data['Weekly_Price_Difference'].dropna())
monthly_std = estimated_population_std(data['Monthly_Price_Difference'].dropna())

current_stock_price = data['Close'].iloc[-1]
prices_data = {
    'Frequency': ['Daily', 'Weekly', 'Monthly'],
    '1st Std Deviation (-)': [current_stock_price - daily_std, current_stock_price - weekly_std, current_stock_price - monthly_std],
    '1st Std Deviation (+)': [current_stock_price + daily_std, current_stock_price + weekly_std, current_stock_price + monthly_std],
    '2nd Std Deviation (-)': [current_stock_price - 2 * daily_std, current_stock_price - 2 * weekly_std, current_stock_price - 2 * monthly_std],
    '2nd Std Deviation (+)': [current_stock_price + 2 * daily_std, current_stock_price + 2 * weekly_std, current_stock_price + 2 * monthly_std],
    '3rd Std Deviation (-)': [current_stock_price - 3 * daily_std, current_stock_price - 3 * weekly_std, current_stock_price - 3 * monthly_std],
    '3rd Std Deviation (+)': [current_stock_price + 3 * daily_std, current_stock_price + 3 * weekly_std, current_stock_price + 3 * monthly_std]
}
prices_table = pd.DataFrame(prices_data)

print("Standard Deviations (Estimated Population):")
print(prices_table.to_string(index=False))

# Generate and plot distribution fits for each frequency with prices
for i, (changes, std, label) in enumerate([
    (data['Daily_Price_Difference'], daily_std, 'Daily'),
    (data['Weekly_Price_Difference'].dropna(), weekly_std, 'Weekly'),
    (data['Monthly_Price_Difference'].dropna(), monthly_std, 'Monthly')
]):
    mean_change = changes.mean()
    plt.figure(figsize=(10, 6))
    hist_data = plt.hist(changes, bins=30, color='blue', alpha=0.5, density=True, label=f'{label} Price Difference')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mean_change, std)
    plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution Fit')
    plt.title(f'Normal Distribution Fit for {label} Price Differences of {ticker_symbol}')
    plt.xlabel(f'{label} Price Difference')
    plt.ylabel('Density')
    
    plt.axvline(mean_change, color='red', linestyle='dashed', linewidth=2, label='Mean')
    plt.text(mean_change, plt.ylim()[1]*0.8, f'{current_stock_price:.2f}', horizontalalignment='right', color='red')
    
    plt.axvline(mean_change + std, color='green', linestyle='dashed', linewidth=2, label='+1 STD')
    plt.text(mean_change + std, plt.ylim()[1]*0.7, f'{current_stock_price + std:.2f}', horizontalalignment='right', color='green')
    
    plt.axvline(mean_change - std, color='green', linestyle='dashed', linewidth=2, label='-1 STD')
    plt.text(mean_change - std, plt.ylim()[1]*0.7, f'{current_stock_price - std:.2f}', horizontalalignment='right', color='green')
    
    plt.axvline(mean_change + 2 * std, color='yellow', linestyle='dashed', linewidth=2, label='+2 STD')
    plt.text(mean_change + 2 * std, plt.ylim()[1]*0.6, f'{current_stock_price + 2 * std:.2f}', horizontalalignment='right', color='yellow')
    
    plt.axvline(mean_change - 2 * std, color='yellow', linestyle='dashed', linewidth=2, label='-2 STD')
    plt.text(mean_change - 2 * std, plt.ylim()[1]*0.6, f'{current_stock_price - 2 * std:.2f}', horizontalalignment='right', color='yellow')
    
    plt.axvline(mean_change + 3 * std, color='orange', linestyle='dashed', linewidth=2, label='+3 STD')
    plt.text(mean_change + 3 * std, plt.ylim()[1]*0.5, f'{current_stock_price + 3 * std:.2f}', horizontalalignment='right', color='orange')
    
    plt.axvline(mean_change - 3 * std, color='orange', linestyle='dashed', linewidth=2, label='-3 STD')
    plt.text(mean_change - 3 * std, plt.ylim()[1]*0.5, f'{current_stock_price - 3 * std:.2f}', horizontalalignment='right', color='orange')
    
    plt.legend()
    plt.show()

# Function to calculate the probability between two z-scores
def calculate_probability(z1, z2):
    return stats.norm.cdf(z2) - stats.norm.cdf(z1)

# User input for prices and calculation of probabilities
while True:
    try:
        lower_price = float(input("Enter the lower price (or press Enter to exit): "))
        upper_price = float(input("Enter the upper price: "))
        
        # Calculate z-scores for daily, weekly, and monthly
        z1_daily = (lower_price - current_stock_price) / daily_std
        z2_daily = (upper_price - current_stock_price) / daily_std
        
        z1_weekly = (lower_price - current_stock_price) / weekly_std
        z2_weekly = (upper_price - current_stock_price) / weekly_std
        
        z1_monthly = (lower_price - current_stock_price) / monthly_std
        z2_monthly = (upper_price - current_stock_price) / monthly_std
        
        # Calculate probabilities
        prob_daily = calculate_probability(z1_daily, z2_daily) * 100
        prob_weekly = calculate_probability(z1_weekly, z2_weekly) * 100
        prob_monthly = calculate_probability(z1_monthly, z2_monthly) * 100
        
        print(f"\nProbability of price being between {lower_price:.2f} and {upper_price:.2f}:")
        print(f"Daily: {prob_daily:.2f}%")
        print(f"Weekly: {prob_weekly:.2f}%")
        print(f"Monthly: {prob_monthly:.2f}%")
        
    except ValueError:
        print("Exiting...")
        break
