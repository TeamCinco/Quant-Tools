import yfinance as yf
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from datetime import datetime

# Download SPY data from 2000 to present
end_date = datetime.now().strftime("%Y-%m-%d")
spy_data = yf.download("SPY", start="2024-01-01", end=end_date)

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