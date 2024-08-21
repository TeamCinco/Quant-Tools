import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

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
    elif period == 'Weekly':
        data = data.resample('W').last()
        data['Change %'] = ((data['Close'] - data['Open']) / data['Open']) * 100
    elif period == 'Monthly':
        data = data.resample('M').last()
        data['Change %'] = ((data['Close'] - data['Open']) / data['Open']) * 100
    
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

    data['Price_Difference'] = data['Close'] - data['Open']
    period_std = np.std(data['Price_Difference'])
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
    plt.hist(data['Price_Difference'], bins=30, color='blue', alpha=0.4, label=period)
    plt.title(f'{ticker} Price Difference Histogram ({period})')
    plt.xlabel('Price Difference')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()

    mean_change = data['Price_Difference'].mean()
    plt.figure(figsize=(10, 6))
    hist_data = plt.hist(data['Price_Difference'], bins=30, color='blue', alpha=0.5, density=True, label='Price Difference')
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

def find_best_iron_condor_strikes(option_chain, std_levels, ticker, expiration_date, frequency_table):
    puts = option_chain.puts
    calls = option_chain.calls
    
    std_labels = ['1st', '2nd', '3rd']
    suffix = ['st', 'nd', 'rd']
    
    iron_condor_options = []

    for i in range(1, 4):
        std_label = std_labels[i - 1]
        level_suffix = suffix[i - 1]
        
        lower_puts = puts[puts['strike'] <= std_levels[f'{std_label} Std Deviation (-)']]
        upper_calls = calls[calls['strike'] >= std_levels[f'{std_label} Std Deviation (+)']]
        
        if not lower_puts.empty and not upper_calls.empty:
            lower_puts = lower_puts.merge(frequency_table, left_on='strike', right_on='Bins', how='left').sort_values(by='Qty%', ascending=False)
            upper_calls = upper_calls.merge(frequency_table, left_on='strike', right_on='Bins', how='left').sort_values(by='Qty%', ascending=False)
            
            sell_put = lower_puts.iloc[0]
            buy_put = lower_puts.iloc[1] if len(lower_puts) > 1 else None
            sell_call = upper_calls.iloc[0]
            buy_call = upper_calls.iloc[1] if len(upper_calls) > 1 else None

            print(f"Sell {i}{level_suffix} Std Dev Iron Condor: Sell Put at {sell_put['strike']}, Buy Put at {buy_put['strike'] if buy_put is not None else 'N/A'}, Sell Call at {sell_call['strike']}, Buy Call at {buy_call['strike'] if buy_call is not None else 'N/A'}")
            
            iron_condor_options.append({
                'sell_put': sell_put['strike'],
                'buy_put': buy_put['strike'] if buy_put is not None else None,
                'sell_call': sell_call['strike'],
                'buy_call': buy_call['strike'] if buy_call is not None else None,
                'std': f'{i}{level_suffix} Std Dev'
            })
        else:
            print(f"No suitable options found for {i}{level_suffix} standard deviation level.")
    
    # Combine puts and calls into a single DataFrame
    all_options = pd.concat([puts, calls])
    
    # Add a new column for highlighting and standard deviation
    all_options['Highlight'] = ''
    all_options['Std Deviation'] = ''
    
    # Highlight the selected options and indicate the standard deviation
    for ic in iron_condor_options:
        all_options.loc[all_options['strike'] == ic['sell_put'], 'Highlight'] += 'Sell Put, '
        all_options.loc[all_options['strike'] == ic['sell_put'], 'Std Deviation'] = ic['std']
        if ic['buy_put']:
            all_options.loc[all_options['strike'] == ic['buy_put'], 'Highlight'] += 'Buy Put, '
            all_options.loc[all_options['strike'] == ic['buy_put'], 'Std Deviation'] = ic['std']
        all_options.loc[all_options['strike'] == ic['sell_call'], 'Highlight'] += 'Sell Call, '
        all_options.loc[all_options['strike'] == ic['sell_call'], 'Std Deviation'] = ic['std']
        if ic['buy_call']:
            all_options.loc[all_options['strike'] == ic['buy_call'], 'Highlight'] += 'Buy Call, '
            all_options.loc[all_options['strike'] == ic['buy_call'], 'Std Deviation'] = ic['std']
    
    # Remove trailing comma and space
    all_options['Highlight'] = all_options['Highlight'].str.rstrip(', ')
    
    # Sort the DataFrame by strike price
    all_options = all_options.sort_values('strike')
    
    # Save to CSV
    csv_path = f"C:\\Users\\cinco\\Desktop\\quant practicie\\Research\\Research Tools\\Options\\Options Chain\\{ticker}_{expiration_date}_options_chain_highlighted.csv"
    all_options.to_csv(csv_path, index=False)
    print(f"Highlighted options chain saved to: {csv_path}")

def main():
    ticker = get_ticker_symbol()
    expiration_dates = get_expiration_dates(ticker)
    display_expiration_dates(expiration_dates)
    selected_date = select_expiration_date(expiration_dates)
    
    period = get_analysis_period()
    
    # Calculate and display statistics and standard deviations
    frequency_table, prices_table, data, period_std, current_stock_price = calculate_stats_and_std(ticker, period)
    
    # Display charts
    display_charts(frequency_table, data, period, period_std, current_stock_price, ticker)
    
    # Allow user to pick from a list which std they want to base the legs of the iron condor
    print("Select the standard deviation level to base the iron condor legs on:")
    for i, row in prices_table.iterrows():
        print(f"{i + 1}: {row['Frequency']} 1st Std Dev (+/-) at {row['1st Std Deviation (+)']:.2f} / {row['1st Std Deviation (-)']:.2f}")
        print(f"{i + 1}: {row['Frequency']} 2nd Std Dev (+/-) at {row['2nd Std Deviation (+)']:.2f} / {row['2nd Std Deviation (-)']:.2f}")
        print(f"{i + 1}: {row['Frequency']} 3rd Std Dev (+/-) at {row['3rd Std Deviation (+)']:.2f} / {row['3rd Std Deviation (-)']:.2f}")
    
    # Debugging print statement to verify the input prompt is displayed
    print("Debug: Input prompt displayed")
    
    # Capture input and handle empty string case
    user_input = input("Enter the number corresponding to your choice: ")
    if not user_input.strip():
        print("No input provided. Please enter a valid number.")
        return
    
    try:
        selected_std = int(user_input) - 1
    except ValueError:
        print(f"Invalid input '{user_input}'. Please enter a valid number.")
        return
    
    # Fetch option chain data
    option_chain = get_option_chain(ticker, selected_date)
    
    # Find the best iron condor strikes based on selected standard deviation and create highlighted CSV
    find_best_iron_condor_strikes(option_chain, prices_table.iloc[selected_std], ticker, selected_date, frequency_table)

if __name__ == "__main__":
    main()
