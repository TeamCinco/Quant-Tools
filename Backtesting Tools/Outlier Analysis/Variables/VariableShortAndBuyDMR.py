import yfinance as yf
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from datetime import datetime
import pandas_datareader as pdr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Ask user for ticker and variable input
user_ticker = input("Enter a ticker symbol: ")
user_variable = input("Enter a variable to analyze (e.g., DAAA for Federal Funds Rate): ")

# Download data for user-specified ticker and SPY from 2024 to present
end_date = datetime.now().strftime("%Y-%m-%d")
ticker_data = yf.download(user_ticker, start="2023-06-09", end=end_date)
spy_data = yf.download("SPY", start="2023-06-09", end=end_date)

# Download user-specified variable data from FRED
variable_data = pdr.get_data_fred(user_variable, start="2023-06-09", end=end_date)

class DollarRangeStrategy(Strategy):
    def init(self):
        self.daily_open = self.I(lambda: self.data.Open)
        self.entry_price = 0

    def next(self):
        lower_price = self.daily_open[-1] - 1
        upper_price = self.daily_open[-1] + 1
        
        if self.position.is_long:
            if self.data.Close[-1] >= upper_price:
                self.position.close()
            elif self.data.Close[-1] <= self.entry_price * 0.5:  # 50% stop loss
                self.position.close()
        elif self.position.is_short:
            if self.data.Close[-1] <= lower_price:
                self.position.close()
            elif self.data.Close[-1] >= self.entry_price * 1.5:  # 50% stop loss for short positions
                self.position.close()
        else:
            if self.data.Close[-1] <= lower_price:
                self.buy()
                self.entry_price = self.data.Close[-1]
            elif self.data.Close[-1] >= upper_price:
                self.sell()
                self.entry_price = self.data.Close[-1]

# Run backtest
bt = Backtest(spy_data, DollarRangeStrategy, cash=5500, commission=0)
results = bt.run()

# Calculate buy-and-hold returns
buy_and_hold_returns = (spy_data['Close'][-1] - spy_data['Close'][0]) / spy_data['Close'][0] * 100

print("Dollar Range Strategy Results:")
print(results)
print(f"\nBuy-and-Hold Returns: {buy_and_hold_returns:.2f}%")
print(f"Strategy Returns: {results['Return [%]']:.2f}%")

if results['Return [%]'] > buy_and_hold_returns:
    print("The Dollar Range Strategy outperformed Buy-and-Hold.")
else:
    print("Buy-and-Hold outperformed the Dollar Range Strategy.")

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

# Create a line chart of the outliers
plt.figure(figsize=(12, 6))
plt.scatter(outliers[user_variable], outliers['Close'], color='red', label='Outliers')
plt.plot(merged_data[user_variable], merged_data['Close'], color='blue', alpha=0.5, label='All Data')
plt.xlabel(user_variable)
plt.ylabel(f"{user_ticker} Close Price")
plt.title(f"Outliers: {user_ticker} Close Price vs {user_variable}")
plt.legend()
plt.grid(True)
plt.show()