import yfinance as yf
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from datetime import datetime
import pandas_datareader as pdr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Ask user for ticker, variable input, and start date
user_ticker = input("Enter a ticker symbol: ")
user_variable = input("Enter a variable to analyze (e.g., DAAA for Federal Funds Rate): ")
start_date = input("Enter the start date (YYYY-MM-DD): ")


# Download data for user-specified ticker and SPY from 2024 to present
end_date = datetime.now().strftime("%Y-%m-%d")
ticker_data = yf.download(user_ticker, start=start_date, end=end_date)
spy_data = yf.download("SPY", start=start_date, end=end_date)
variable_data = pdr.get_data_fred(user_variable, start=start_date, end=end_date)




def calculate_rsi(close, window=14):    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.convolve(gain, np.ones(window), 'valid') / window
    avg_loss = np.convolve(loss, np.ones(window), 'valid') / window
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # Pad the beginning with NaNs to match the original length
    return np.concatenate([np.full(window, np.nan), rsi])

class MeanReversionRSIStrategy(Strategy):
    window = 20
    rsi_window = 14
    rsi_overbought = 70
    rsi_oversold = 30
    
    def init(self):
        close = self.data.Close
        self.sma = self.I(lambda x: np.convolve(x, np.ones(self.window), 'same') / self.window, close)
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
buy_and_hold_returns = (ticker_data['Close'][-1] - ticker_data['Close'][0]) / ticker_data['Close'][0] * 100

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

# Analyze the relationship between the user-specified ticker and variable
merged_data = pd.merge(ticker_data, variable_data, left_index=True, right_index=True, how='inner')
merged_data = merged_data.dropna()

X = merged_data[user_variable].values.reshape(-1, 1)
y = merged_data['Close'].values

model = LinearRegression()
model.fit(X, y)

print(f"\nLinear Regression Analysis: {user_ticker} vs {user_variable}")
print(f"Coefficient: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"R-squared: {model.score(X, y):.4f}")

# Identify outliers (days with significant price changes)
merged_data['Returns'] = merged_data['Close'].pct_change()
outliers = merged_data[abs(merged_data['Returns']) > 2 * merged_data['Returns'].std()]

print("\nOutliers (days with significant price changes):")
for date, row in outliers.iterrows():
    print(f"Date: {date.date()}, Close: {row['Close']:.2f}, {user_variable}: {row[user_variable]:.2f}")

# Create a scatter plot of the outliers
plt.figure(figsize=(12, 6))
plt.scatter(outliers[user_variable], outliers['Close'], color='red', label='Outliers')
plt.scatter(merged_data[user_variable], merged_data['Close'], color='blue', alpha=0.5, label='All Data')
plt.xlabel(user_variable)
plt.ylabel(f"{user_ticker} Close Price")
plt.title(f"Outliers: {user_ticker} Close Price vs {user_variable}")
plt.legend()
plt.grid(True)
plt.show()

# Plot RSI
plt.figure(figsize=(12, 6))
plt.plot(ticker_data.index, ticker_data['RSI'])
plt.axhline(y=70, color='r', linestyle='--')
plt.axhline(y=30, color='g', linestyle='--')
plt.title(f'RSI for {user_ticker}')
plt.ylabel('RSI')
plt.show()