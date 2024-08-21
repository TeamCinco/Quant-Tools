import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
from datetime import datetime, timedelta

def get_ticker_symbol():
    return input("Enter the ticker symbol: ").upper()

def get_expiration_dates(ticker):
    stock = yf.Ticker(ticker)
    return stock.options

def display_expiration_dates(dates):
    for i, date in enumerate(dates):
        print(f"{i + 1}: {date}")

def select_expiration_date(dates):
    index = int(input("Select the expiration date by number: ")) - 1
    return dates[index]

def get_option_chain(ticker, expiration_date):
    stock = yf.Ticker(ticker)
    return stock.option_chain(expiration_date)

def get_analysis_period():
    print("Select the analysis period for standard deviation:")
    print("1: Daily")
    print("2: Weekly")
    print("3: Monthly")
    period_choice = int(input("Enter the number corresponding to your choice: "))
    period_dict = {1: 'Daily', 2: 'Weekly', 3: 'Monthly'}
    return period_dict.get(period_choice, 'Daily')

def calculate_stats_and_std(ticker, period):
    data = yf.download(ticker, period='6mo')

    if period == 'Daily':
        data['Change %'] = ((data['Close'] - data['Open']) / data['Open']) * 100
        data['Price_Difference'] = data['Close'] - data['Open']
    elif period == 'Weekly':
        data = data.resample('W').last()
        data['Change %'] = ((data['Close'] - data['Open']) / data['Open']) * 100
        data['Price_Difference'] = data['Close'] - data['Open'].shift(1)
    elif period == 'Monthly':
        data = data.resample('M').last()
        data['Change %'] = ((data['Close'] - data['Open']) / data['Open']) * 100
        data['Price_Difference'] = data['Close'] - data['Open'].shift(1)
    
    bins = np.arange(-3, 3.5, 0.5).tolist()
    bin_labels = [f"{bins[i]:.1f}% to {bins[i+1]:.1f}%" for i in range(len(bins)-1)]
    data['Bin'] = pd.cut(data['Change %'], bins=bins, labels=bin_labels, include_lowest=True)
    frequency_table = pd.DataFrame({
        'Bins': data['Bin'].value_counts(sort=False).index.categories,
        'Qty': data['Bin'].value_counts(sort=False).values
    })
    frequency_table['Qty%'] = (frequency_table['Qty'] / frequency_table['Qty'].sum()) * 100
    frequency_table['Cum%'] = frequency_table['Qty%'].cumsum()
    frequency_table.sort_values(by='Bins', inplace=True)
    frequency_table['Qty%'] = frequency_table['Qty%'].map('{:.2f}%'.format)
    frequency_table['Cum%'] = frequency_table['Cum%'].map('{:.2f}%'.format)

    period_std = np.std(data['Price_Difference'].dropna())
    current_stock_price = data['Close'].iloc[-1]
    prices_data = {
        'Frequency': [period],
        '1st Std Deviation (-)': [current_stock_price - period_std],
        '1st Std Deviation (+)': [current_stock_price + period_std],
        '2nd Std Deviation (-)': [current_stock_price - 2 * period_std],
        '2nd Std Deviation (+)': [current_stock_price + 2 * period_std],
        '3rd Std Deviation (-)': [current_stock_price - 3 * period_std],
        '3rd Std Deviation (+)': [current_stock_price + 3 * period_std]
    }
    prices_table = pd.DataFrame(prices_data)

    return frequency_table, prices_table, data, period_std, current_stock_price

def display_charts(frequency_table, data, period, period_std, current_stock_price, ticker):
    print(frequency_table.to_string(index=False))

    plt.figure(figsize=(12, 8))
    plt.barh(np.arange(len(frequency_table)), frequency_table['Qty'], color='blue', edgecolor='black')
    plt.xlabel('Frequency')
    plt.ylabel('Change in %')
    plt.title(f'Change in Percentage from Open to Close ({period}) - {ticker}')
    plt.yticks(ticks=np.arange(len(frequency_table)), labels=frequency_table['Bins'])
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(data['Price_Difference'].dropna(), bins=30, color='blue', alpha=0.4, label=period)
    plt.title(f'{ticker} Price Difference Histogram ({period})')
    plt.xlabel('Price Difference')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()

    mean_change = data['Price_Difference'].dropna().mean()
    plt.figure(figsize=(10, 6))
    hist_data = plt.hist(data['Price_Difference'].dropna(), bins=30, color='blue', alpha=0.5, density=True, label='Price Difference')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mean_change, period_std)
    plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution Fit')
    plt.title(f'Normal Distribution Fit for Price Differences ({period}) - {ticker}')
    plt.xlabel('Price Difference')
    plt.ylabel('Density')

    plt.axvline(mean_change, color='red', linestyle='dashed', linewidth=2, label='Mean')
    plt.text(mean_change, plt.ylim()[1]*0.8, f'{current_stock_price:.2f}', horizontalalignment='right', color='red')

    plt.axvline(mean_change + period_std, color='green', linestyle='dashed', linewidth=2, label='+1 STD')
    plt.text(mean_change + period_std, plt.ylim()[1]*0.7, f'{current_stock_price + period_std:.2f}', horizontalalignment='right', color='green')

    plt.axvline(mean_change - period_std, color='green', linestyle='dashed', linewidth=2, label='-1 STD')
    plt.text(mean_change - period_std, plt.ylim()[1]*0.7, f'{current_stock_price - period_std:.2f}', horizontalalignment='right', color='green')

    plt.axvline(mean_change + 2 * period_std, color='yellow', linestyle='dashed', linewidth=2, label='+2 STD')
    plt.text(mean_change + 2 * period_std, plt.ylim()[1]*0.6, f'{current_stock_price + 2 * period_std:.2f}', horizontalalignment='right', color='yellow')

    plt.axvline(mean_change - 2 * period_std, color='yellow', linestyle='dashed', linewidth=2, label='-2 STD')
    plt.text(mean_change - 2 * period_std, plt.ylim()[1]*0.6, f'{current_stock_price - 2 * period_std:.2f}', horizontalalignment='right', color='yellow')

    plt.axvline(mean_change + 3 * period_std, color='orange', linestyle='dashed', linewidth=2, label='+3 STD')
    plt.text(mean_change + 3 * period_std, plt.ylim()[1]*0.5, f'{current_stock_price + 3 * period_std:.2f}', horizontalalignment='right', color='orange')

    plt.axvline(mean_change - 3 * period_std, color='orange', linestyle='dashed', linewidth=2, label='-3 STD')
    plt.text(mean_change - 3 * period_std, plt.ylim()[1]*0.5, f'{current_stock_price - 3 * period_std:.2f}', horizontalalignment='right', color='orange')

    plt.legend()
    plt.show()

def convert_bin_to_numeric(bin_str):
    try:
        parts = bin_str.replace('%', '').split('to')
        if len(parts) == 2:
            return float(parts[1].strip())
        elif len(parts) == 1:
            return float(parts[0].strip())
        else:
            raise ValueError(f"Unexpected bin format: {bin_str}")
    except ValueError as e:
        print(f"Error converting bin to numeric: {e}")
        return np.nan

def find_best_iron_condor_strikes(option_chain, std_levels, ticker, expiration_date, frequency_table, current_stock_price):
    puts = option_chain.puts
    calls = option_chain.calls
    
    frequency_table['Bins_numeric'] = frequency_table['Bins'].apply(convert_bin_to_numeric)
    frequency_table = frequency_table.dropna(subset=['Bins_numeric'])
    
    std_labels = ['1st', '2nd', '3rd']
    suffix = ['st', 'nd', 'rd']
    
    iron_condor_options = []

    for i in range(1, 4):
        std_label = std_labels[i - 1]
        level_suffix = suffix[i - 1]
        
        lower_bound = std_levels[f'{std_label} Std Deviation (-)']
        upper_bound = std_levels[f'{std_label} Std Deviation (+)']
        
        # Find suitable put strikes
        sell_put_options = puts[(puts['strike'] <= lower_bound) & (puts['strike'] >= lower_bound - 10)]
        if not sell_put_options.empty:
            sell_put = sell_put_options.iloc[-1]  # Choose the highest strike in range
            buy_put_options = puts[puts['strike'] < sell_put['strike']]
            buy_put = buy_put_options.iloc[-1] if not buy_put_options.empty else None
        else:
            sell_put = buy_put = None

        # Find suitable call strikes
        sell_call_options = calls[(calls['strike'] >= upper_bound) & (calls['strike'] <= upper_bound + 10)]
        if not sell_call_options.empty:
            sell_call = sell_call_options.iloc[0]  # Choose the lowest strike in range
            buy_call_options = calls[calls['strike'] > sell_call['strike']]
            buy_call = buy_call_options.iloc[0] if not buy_call_options.empty else None
        else:
            sell_call = buy_call = None

        if (sell_put is not None and buy_put is not None and 
            sell_call is not None and buy_call is not None):
            iron_condor_options.append({
                'sell_put': sell_put,
                'buy_put': buy_put,
                'sell_call': sell_call,
                'buy_call': buy_call,
                'std': f'{i}{level_suffix} Std Dev'
            })
        else:
            print(f"No suitable options found for {i}{level_suffix} standard deviation level.")
    
    return iron_condor_options

def get_number_of_spreads():
    while True:
        try:
            num_spreads = int(input("Enter the number of spreads you want to sell: "))
            if num_spreads > 0:
                return num_spreads
            else:
                print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input. Please enter a positive integer.")

def main():
    ticker = get_ticker_symbol()
    expiration_dates = get_expiration_dates(ticker)
    display_expiration_dates(expiration_dates)
    selected_date = select_expiration_date(expiration_dates)
    
    period = get_analysis_period()
    
    frequency_table, prices_table, data, period_std, current_stock_price = calculate_stats_and_std(ticker, period)
    
    display_charts(frequency_table, data, period, period_std, current_stock_price, ticker)
    
    option_chain = get_option_chain(ticker, selected_date)
    
    iron_condor_options = find_best_iron_condor_strikes(option_chain, prices_table.iloc[0], ticker, selected_date, frequency_table, current_stock_price)
    
    if not iron_condor_options:
        print("No suitable Iron Condor strategies found for the given parameters.")
        return

    num_spreads = get_number_of_spreads()
    
    print("\nAvailable Iron Condor Strategies:")
    for i, ic in enumerate(iron_condor_options):
        max_profit = (ic['sell_put']['lastPrice'] - ic['buy_put']['lastPrice'] + ic['sell_call']['lastPrice'] - ic['buy_call']['lastPrice']) * 100 * num_spreads
        max_loss = (ic['sell_put']['strike'] - ic['buy_put']['strike'] - (ic['sell_put']['lastPrice'] - ic['buy_put']['lastPrice'] + ic['sell_call']['lastPrice'] - ic['buy_call']['lastPrice'])) * 100 * num_spreads
        
        print(f"{i+1}. {ic['std']} Iron Condor:")
        print(f"   Sell Put: Strike {ic['sell_put']['strike']}, Premium {ic['sell_put']['lastPrice']}")
        print(f"   Buy Put: Strike {ic['buy_put']['strike']}, Premium {ic['buy_put']['lastPrice']}")
        print(f"   Sell Call: Strike {ic['sell_call']['strike']}, Premium {ic['sell_call']['lastPrice']}")
        print(f"   Buy Call: Strike {ic['buy_call']['strike']}, Premium {ic['buy_call']['lastPrice']}")
        print(f"   Number of Spreads: {num_spreads}")
        print(f"   Max Profit: ${max_profit:.2f}")
        print(f"   Max Loss: ${max_loss:.2f}")
        print()

    print("These are the available Iron Condor strategies based on the standard deviations.")
    print("You can use this information to make your trading decision.")
if __name__ == "__main__":
    main()