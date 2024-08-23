import yfinance as yf
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from datetime import datetime
import matplotlib.pyplot as plt

# Ask user for ticker and start date
user_ticker = input("Enter a ticker symbol: ")
start_date = input("Enter the start date (YYYY-MM-DD): ")

# Ensure start_date is not in the future
start_date = min(datetime.strptime(start_date, "%Y-%m-%d"), datetime.now()).strftime("%Y-%m-%d")

# Download data for user-specified ticker from start_date to present
end_date = datetime.now().strftime("%Y-%m-%d")
ticker_data = yf.download(user_ticker, start=start_date, end=end_date)

# If no data is available, adjust the start date to the earliest available date
if ticker_data.empty:
    print("No data available for the specified date range. Adjusting start date...")
    ticker_data = yf.download(user_ticker, start="1900-01-01", end=end_date)
    if not ticker_data.empty:
        start_date = ticker_data.index[0].strftime("%Y-%m-%d")
        print(f"Adjusted start date: {start_date}")
        ticker_data = yf.download(user_ticker, start=start_date, end=end_date)
    else:
        print(f"No data available for {user_ticker}. Please check the ticker symbol.")
        exit()

def calculate_rsi(close, window=14):
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.convolve(gain, np.ones(window), 'valid') / window
    avg_loss = np.convolve(loss, np.ones(window), 'valid') / window
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return np.concatenate([np.full(window, np.nan), rsi])

class MeanReversionRSIStrategy(Strategy):
    window = 20
    rsi_window = 14
    rsi_overbought = 70
    rsi_oversold = 30
    
    def init(self):
        close = self.data.Close
        self.sma = self.I(lambda x: pd.Series(x).rolling(window=self.window).mean(), close)
        self.rsi = self.I(calculate_rsi, close, self.rsi_window)
        self.entry_price = 0

    def next(self):
        if np.isnan(self.sma[-1]) or np.isnan(self.rsi[-1]):
            return

        current_price = self.data.Close[-1]
        
        if not self.position:
            if current_price < self.sma[-1] and self.rsi[-1] < self.rsi_oversold:
                self.buy()
                self.entry_price = current_price
            elif current_price > self.sma[-1] and self.rsi[-1] > self.rsi_overbought:
                self.sell()
                self.entry_price = current_price
        else:
            if self.position.is_long:
                if current_price > self.sma[-1] or self.rsi[-1] > self.rsi_overbought:
                    self.position.close()
                elif current_price <= self.entry_price * 0.5:  # 50% stop loss
                    self.position.close()
            elif self.position.is_short:
                if current_price < self.sma[-1] or self.rsi[-1] < self.rsi_oversold:
                    self.position.close()
                elif current_price >= self.entry_price * 1.5:  # 50% stop loss for short positions
                    self.position.close()

# Calculate RSI for the ticker data (for plotting purposes)
ticker_data['RSI'] = calculate_rsi(ticker_data['Close'].values)

# Run backtest
bt = Backtest(ticker_data, MeanReversionRSIStrategy, cash=10000, commission=0)
results = bt.run()

# Calculate buy-and-hold returns
buy_and_hold_returns = (ticker_data['Close'].iloc[-1] - ticker_data['Close'].iloc[0]) / ticker_data['Close'].iloc[0] * 100

print("Mean Reversion RSI Strategy Results:")
print(results)
print(f"\nBuy-and-Hold Returns: {buy_and_hold_returns:.2f}%")
print(f"Strategy Returns: {results['Return [%]']:.2f}%")

if results['Return [%]'] > buy_and_hold_returns:
    print("The Mean Reversion RSI Strategy outperformed Buy-and-Hold.")
else:
    print("Buy-and-Hold outperformed the Mean Reversion RSI Strategy.")

# Plot the results
bt.plot()

# Plot RSI
plt.figure(figsize=(12, 6))
plt.plot(ticker_data.index, ticker_data['RSI'])
plt.axhline(y=70, color='r', linestyle='--')
plt.axhline(y=30, color='g', linestyle='--')
plt.title(f'RSI for {user_ticker}')
plt.ylabel('RSI')
plt.show()
