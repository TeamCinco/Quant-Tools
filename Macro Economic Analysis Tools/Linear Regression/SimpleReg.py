import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import yfinance as yf
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def fetch_data(fred_symbol, stock_symbol, start_date, end_date):
    # Fetch FRED data
    fred_data = pdr.get_data_fred(fred_symbol, start_date, end_date)
    
    # Fetch stock data using yfinance directly
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)['Adj Close']
    
    # Merge dataframes
    merged_data = pd.merge(fred_data, stock_data, left_index=True, right_index=True, how='inner')
    merged_data.columns = ['FRED', 'Stock']
    
    return merged_data

def main():
    # Get user input
    fred_symbol = input("Enter the FRED symbol: ")
    stock_symbol = input("Enter the stock symbol: ")
    
    # Get user input for the number of years
    while True:
        try:
            years = int(input("Enter the number of years to go back: ")) 
            if years > 0:
                break
            else:
                print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input. Please enter a positive integer.")

    # Get user input for independent variable
    while True:
        independent_var = input("Choose the independent variable (FRED/Stock): ").lower()
        if independent_var in ['fred', 'stock']:
            break
        else:
            print("Invalid input. Please enter either 'FRED' or 'Stock'.")

    # Set date range based on user input
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)

    # Fetch data
    data = fetch_data(fred_symbol, stock_symbol, start_date, end_date)

    # Check if data is empty
    if data.empty:
        print("No data available for the given symbols and date range.")
        return

    # Log transform the data
    data['FRED_log'] = np.log(data['FRED'])
    data['Stock_log'] = np.log(data['Stock'])

    # Set independent and dependent variables based on user choice
    if independent_var == 'fred':
        x = data['FRED_log']
        y = data['Stock_log']
        x_label = f'Log({fred_symbol})'
        y_label = f'Log({stock_symbol})'
    else:
        x = data['Stock_log']
        y = data['FRED_log']
        x_label = f'Log({stock_symbol})'
        y_label = f'Log({fred_symbol})'

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Print results
    print(f"\nLinear Regression Results:")
    print(f"Independent Variable: {independent_var.upper()}")
    print(f"Dependent Variable: {'Stock' if independent_var == 'fred' else 'FRED'}")
    print(f"Slope: {slope:.4f}")
    print(f"Intercept: {intercept:.4f}")
    print(f"R-squared: {r_value**2:.4f}")
    print(f"P-value: {p_value:.4f}")

    # Plot the data and regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.5)
    plt.plot(x, slope * x + intercept, color='red')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'Linear Regression: {fred_symbol} vs {stock_symbol} (Last {years} years)')
    plt.show()

if __name__ == "__main__":
    main()