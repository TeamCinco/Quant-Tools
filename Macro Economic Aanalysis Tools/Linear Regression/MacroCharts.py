import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates

def get_fred_data(symbols):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3650)  # Last 10 years of data
    
    data = {}
    for symbol in symbols:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={symbol}"
        series = pd.read_csv(url, parse_dates=['DATE'], index_col='DATE')
        series = series.loc[start_date:end_date]
        data[symbol] = series[symbol]
    
    return pd.DataFrame(data)

def plot_fred_data(data):
    for column in data.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data[column], marker='o', markersize=2)  # Adding markers for better readability
        
        # Improve x-axis
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        
        # Rotate and align the tick labels so they look better
        fig.autofmt_xdate()
        
        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        # Add labels and title
        plt.title(f"{column}", fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        
        # Add gridlines
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout and display
        plt.tight_layout()
        plt.show()

def main():
    print("Welcome to the FRED Data Visualizer!")
    print("Please enter the FRED symbols you want to visualize, separated by commas.")
    print("Example: GDPC1,UNRATE,CPIAUCSL")
    
    user_input = input("Enter FRED symbols: ")
    symbols = [symbol.strip() for symbol in user_input.split(',')]
    
    try:
        data = get_fred_data(symbols)
        plot_fred_data(data)
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check your symbols and try again.")

if __name__ == "__main__":
    main()
