import yfinance as yf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import griddata
import pandas as pd

def fetch_available_expirations(ticker):
    stock = yf.Ticker(ticker)
    return stock.options

def fetch_options_chain(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    all_options = []
    for date in stock.options:
        expiration_date = datetime.strptime(date, "%Y-%m-%d")
        if start_date <= expiration_date <= end_date:
            opt = stock.option_chain(date)
            all_options.append((expiration_date, opt.calls, opt.puts))
    return all_options

def calculate_moneyness(strike, underlying_price):
    return strike / underlying_price

def calculate_time_to_expire(expiration_date):
    today = datetime.today()
    delta = expiration_date - today
    return delta.days / 365

def plot_implied_volatility_surface(data, title, filename):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Convert data to numpy arrays
    x = np.array(data['moneyness'])
    y = np.array(data['timeToExpire'])
    z = np.array(data['impliedVolatility'])

    # Remove any non-positive IV values
    mask = z > 0
    x, y, z = x[mask], y[mask], z[mask]

    # Create a grid
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    X, Y = np.meshgrid(xi, yi)

    # Interpolate Z values on the grid
    Z = griddata((x, y), z, (X, Y), method='cubic')

    # Set any remaining negative values to NaN to avoid plotting them
    Z[Z < 0] = np.nan

    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

    ax.set_xlabel('Moneyness')
    ax.set_ylabel('Time to Expire (Years)')
    ax.set_zlabel('Implied Volatility')
    ax.set_title(title)

    # Set a minimum z-axis value of 0
    ax.set_zlim(bottom=0)

    # Adjust the view angle
    ax.view_init(elev=20, azim=45)

    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # Format the axis labels
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: '{:.2f}'.format(val)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: '{:.2f}'.format(val)))
    ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: '{:.2f}'.format(val)))

    # Add a legend explaining the graph
    legend_text = (
        "Moneyness: Scale of In-the-Money (< 1), At-the-Money (= 1), Out-of-the-Money (> 1)\n"
        "Time to Expire: Fraction of a Year\n"
        "Implied Volatility: Market's Expectation of Volatility"
    )
    ax.text2D(0.05, 0.95, legend_text, transform=ax.transAxes, fontsize=10,
              verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', alpha=0.3))

    plt.show()
    plt.savefig(filename)

def plot_stock_price(ticker):
    stock = yf.Ticker(ticker)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # 6 months ago
    history = stock.history(start=start_date, end=end_date)
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.index, history['Close'])
    plt.title(f'{ticker} Stock Price (Past 6 Months)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def save_to_excel(data_calls, data_puts, ticker):
    with pd.ExcelWriter(rf"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\Results\9.28.24\{ticker}_options_data.xlsx") as writer:
        df_calls = pd.DataFrame(data_calls)
        df_puts = pd.DataFrame(data_puts)
        df_calls.to_excel(writer, sheet_name="Calls", index=False)
        df_puts.to_excel(writer, sheet_name="Puts", index=False)
    print(f"Data has been saved to {ticker}_options_data.xlsx")

def main():
    ticker = input("Enter a ticker symbol: ").upper()
    
    print(f"Fetching available expiration dates for {ticker}...")
    expiration_dates = fetch_available_expirations(ticker)
    
    print("Available expiration dates:")
    for i, date in enumerate(expiration_dates):
        print(f"{i+1}. {date}")
    
    start_index = int(input("Enter the number for the start date: ")) - 1
    end_index = int(input("Enter the number for the end date: ")) - 1

    start_date = datetime.strptime(expiration_dates[start_index], "%Y-%m-%d")
    end_date = datetime.strptime(expiration_dates[end_index], "%Y-%m-%d")

    print(f"Fetching options chain for {ticker} from {start_date.date()} to {end_date.date()}...")
    all_options = fetch_options_chain(ticker, start_date, end_date)
    
    stock = yf.Ticker(ticker)
    underlying_price = stock.history(period="1d")['Close'].iloc[0]

    data_calls = {
        'strike': [],
        'expiration': [],
        'impliedVolatility': [],
        'moneyness': [],
        'timeToExpire': []
    }

    data_puts = {
        'strike': [],
        'expiration': [],
        'impliedVolatility': [],
        'moneyness': [],
        'timeToExpire': []
    }

    for date, calls, puts in all_options:
        data_calls['strike'].extend(calls['strike'])
        data_calls['expiration'].extend([date] * len(calls))
        data_calls['impliedVolatility'].extend(calls['impliedVolatility'])
        data_calls['moneyness'].extend([calculate_moneyness(strike, underlying_price) for strike in calls['strike']])
        data_calls['timeToExpire'].extend([calculate_time_to_expire(date)] * len(calls))

        data_puts['strike'].extend(puts['strike'])
        data_puts['expiration'].extend([date] * len(puts))
        data_puts['impliedVolatility'].extend(puts['impliedVolatility'])
        data_puts['moneyness'].extend([calculate_moneyness(strike, underlying_price) for strike in puts['strike']])
        data_puts['timeToExpire'].extend([calculate_time_to_expire(date)] * len(puts))

    print("\nSaving options data to Excel...")
    save_to_excel(data_calls, data_puts, ticker)

    print("\nPlotting and saving Calls Implied Volatility Surface...")
    plot_implied_volatility_surface(data_calls, f'{ticker} Calls Implied Volatility Surface', rf"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\Results\9.28.24\{ticker}_calls_iv_surface.png")

    print("\nPlotting and saving Puts Implied Volatility Surface...")
    plot_implied_volatility_surface(data_puts, f'{ticker} Puts Implied Volatility Surface', rf"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\Results\9.28.24\{ticker}_puts_iv_surface.png")

    print("\nPlotting stock price for the past 6 months...")
    plot_stock_price(ticker)

if __name__ == "__main__":
    main()
