import yfinance as yf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import griddata

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

def plot_implied_volatility_surface(data, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Convert data to numpy arrays
    x = np.array(data['moneyness'])
    y = np.array(data['timeToExpire'])
    z = np.array(data['impliedVolatility'])

    # Create a grid
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    X, Y = np.meshgrid(xi, yi)

    # Interpolate Z values on the grid
    Z = griddata((x, y), z, (X, Y), method='cubic')

    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

    ax.set_xlabel('Moneyness')
    ax.set_ylabel('Time to Expire (Years)')
    ax.set_zlabel('Implied Volatility')
    ax.set_title(title)

    # Adjust the view angle to match the image
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

def main():
    ticker = input("Enter a ticker symbol: ").upper()
    
    print(f"Fetching available expiration dates for {ticker}...")
    expiration_dates = fetch_available_expirations(ticker)
    
    print("Available expiration dates:")
    for i, date in enumerate(expiration_dates):
        print(f"{i+1}. {date}")
    
    start_index = int(input("Enter the number of the start date: ")) - 1
    end_index = int(input("Enter the number of the end date: ")) - 1
    
    start_date = datetime.strptime(expiration_dates[start_index], "%Y-%m-%d")
    end_date = datetime.strptime(expiration_dates[end_index], "%Y-%m-%d")

    print(f"Fetching options chain for {ticker} from {start_date.date()} to {end_date.date()}...")
    all_options = fetch_options_chain(ticker, start_date, end_date)
    
    stock = yf.Ticker(ticker)
    underlying_price = stock.history(period="1d")['Close'].iloc[0]

    for option_type in ['Calls', 'Puts']:
        data = {
            'strike': [],
            'expiration': [],
            'impliedVolatility': [],
            'moneyness': [],
            'timeToExpire': []
        }

        for date, calls, puts in all_options:
            options = calls if option_type == 'Calls' else puts
            data['strike'].extend(options['strike'])
            data['expiration'].extend([date] * len(options))
            data['impliedVolatility'].extend(options['impliedVolatility'])
            data['moneyness'].extend([calculate_moneyness(strike, underlying_price) for strike in options['strike']])
            data['timeToExpire'].extend([calculate_time_to_expire(date)] * len(options))

        print(f"\nPlotting {option_type} optons from 2024 Implied Volatility Surface...")
        plot_implied_volatility_surface(data, f'{ticker} {option_type} Implied Volatility Surface')

        # Keep the plot open
        input("Press Enter to close the plot and continue...")

    print("\nPlotting stock price for the past 6 months...")
    plot_stock_price(ticker)

    # Keep the plot open
    input("Press Enter to close the plot and exit...")

if __name__ == "__main__":
    main()
