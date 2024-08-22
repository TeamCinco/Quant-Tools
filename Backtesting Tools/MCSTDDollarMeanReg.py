import yfinance as yf
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from datetime import datetime
import pandas_market_calendars as mcal
import matplotlib.pyplot as plt

def get_stock_data(ticker, start_date, end_date):
    print(f"Attempting to download data for {ticker} from {start_date} to {end_date}")
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    if stock_data.empty:
        print(f"No data retrieved for {ticker} between {start_date} and {end_date}")
    else:
        print(f"Retrieved {len(stock_data)} rows of data for {ticker}")
    return stock_data

def calculate_population_stats(data):
    data['Daily_Return'] = data['Close'].pct_change()
    
    daily_mean = np.mean(data['Daily_Return'])
    daily_var = np.var(data['Daily_Return'], ddof=1)  # Using sample variance
    daily_std = np.sqrt(daily_var)
    
    return daily_mean, daily_std

def monte_carlo_simulation(current_price, daily_mean, daily_std, days, mc_sims=10000):
    simulations = np.zeros((days, mc_sims))
    for i in range(mc_sims):
        prices = [current_price]
        for _ in range(days - 1):
            daily_return = np.random.normal(daily_mean, daily_std)
            prices.append(prices[-1] * (1 + daily_return))
        simulations[:, i] = prices
    return simulations

class DollarRangeStrategy(Strategy):
    def init(self):
        self.daily_open = self.I(lambda: self.data.Open)

    def next(self):
        lower_price = self.daily_open[-1] - 1
        upper_price = self.daily_open[-1] + 1
        
        if self.position.is_long:
            if self.data.Close[-1] >= upper_price:
                self.position.close()
        elif self.position.is_short:
            if self.data.Close[-1] <= lower_price:
                self.position.close()
        else:
            if self.data.Close[-1] <= lower_price:
                self.buy()
            elif self.data.Close[-1] >= upper_price:
                self.sell()

def main():
    # Parameters
    ticker = "SPY"
    start_date = "2019-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    prediction_days = 30

    print(f"Starting main function with parameters:")
    print(f"Ticker: {ticker}")
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Prediction Days: {prediction_days}")

    # Get stock data
    stock_data = get_stock_data(ticker, start_date, end_date)
    
    if stock_data.empty:
        print("Error: No data retrieved for the specified date range.")
        return

    # Print the first few rows of the data to verify its contents
    print("\nFirst few rows of the data:")
    print(stock_data.head())

    # Calculate population statistics
    daily_mean, daily_std = calculate_population_stats(stock_data)
    print(f"\nCalculated statistics:")
    print(f"Daily Mean: {daily_mean}")
    print(f"Daily Standard Deviation: {daily_std}")

    # Get NYSE calendar
    print("\nGetting NYSE calendar...")
    nyse = mcal.get_calendar('NYSE')
    valid_days = nyse.valid_days(start_date=start_date, end_date=end_date)
    print(f"Number of valid trading days: {len(valid_days)}")

    # Print the first few valid trading days
    print("First few valid trading days:")
    print(valid_days[:5])

    # Print the first few dates from stock_data
    print("\nFirst few dates from stock_data:")
    print(stock_data.index[:5])

    # Check if the index of stock_data is tz-aware
    print(f"\nIs stock_data index tz-aware? {stock_data.index.tz is not None}")
    print(f"Is valid_days tz-aware? {valid_days.tz is not None}")

    # Convert valid_days to the same timezone as stock_data if necessary
    if stock_data.index.tz is None and valid_days.tz is not None:
        valid_days = valid_days.tz_localize(None)
    elif stock_data.index.tz is not None and valid_days.tz is None:
        valid_days = valid_days.tz_localize(stock_data.index.tz)

    # Filter the data to include only valid trading days
    stock_data = stock_data[stock_data.index.isin(valid_days)]

    # Print the shape of the filtered data
    print(f"\nShape of filtered data: {stock_data.shape}")

    if stock_data.empty:
        print("Error: Filtered data is empty. Check the date range and stock ticker.")
        return

    # Run backtest
    print("\nRunning backtest...")
    bt = Backtest(stock_data, DollarRangeStrategy, cash=5500, commission=0)
    results = bt.run()

    print("Backtest Results:")
    print(results)

    # Calculate buy-and-hold returns
    buy_and_hold_returns = (stock_data['Close'][-1] - stock_data['Close'][0]) / stock_data['Close'][0] * 100

    print(f"\nBuy-and-Hold Returns: {buy_and_hold_returns:.2f}%")
    print(f"Strategy Returns: {results['Return [%]']:.2f}%")

    if results['Return [%]'] > buy_and_hold_returns:
        print("The Dollar Range Strategy outperformed Buy-and-Hold.")
    else:
        print("Buy-and-Hold outperformed the Dollar Range Strategy.")

    # Monte Carlo Simulation
    print("\nRunning Monte Carlo Simulation...")
    current_price = stock_data['Close'][-1]
    mc_sims = monte_carlo_simulation(current_price, daily_mean, daily_std, prediction_days)

    # Plot Monte Carlo Simulation results
    plt.figure(figsize=(10, 6))
    plt.plot(mc_sims)
    plt.title(f'Monte Carlo Simulation: {ticker} Price Prediction for {prediction_days} days')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.show()

    # Calculate confidence intervals
    confidence_intervals = np.percentile(mc_sims, [5, 50, 95], axis=1)
    
    print(f"\nMonte Carlo Simulation Results for {prediction_days} days:")
    print(f"5% Lower Bound: ${confidence_intervals[0, -1]:.2f}")
    print(f"50% Median: ${confidence_intervals[1, -1]:.2f}")
    print(f"95% Upper Bound: ${confidence_intervals[2, -1]:.2f}")

if __name__ == "__main__":
    main()