import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, timedelta

def get_risk_free_rate(expiration_date):
    # Read the CSV file
    treasury_data = pd.read_csv(r'C:\Users\cinco\Desktop\quant practicie\Research\Research Tools\Options\Macro\daily-treasury-rates.csv', index_col='Date', parse_dates=True)
    treasury_data = treasury_data.sort_index(ascending=False)  # Sort by date descending
    
    # Get the most recent rate before or on the expiration date
    for date, row in treasury_data.iterrows():
        if date <= expiration_date:
            # Use the 1 Year rate as a proxy for options expiring within a year
            return row['1 Yr'] / 100  # Convert percentage to decimal
    
    # If no suitable date found, use the most recent rate
    return treasury_data.iloc[0]['1 Yr'] / 100

def get_dividend_yield(ticker):
    try:
        return ticker.info['dividendYield']
    except:
        return 0

def trading_days_to_expiration(expiration_date):
    today = datetime.now().date()
    expiration = datetime.strptime(expiration_date, "%Y-%m-%d").date()
    days = np.busday_count(today, expiration)
    return max(days, 1) / 252  # Convert to years, ensure at least 1 day

def black_scholes_option_price(S, K, T, r, q, sigma, option_type):
    if sigma <= 0 or T <= 0:
        return np.nan

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    try:
        if option_type == 'call':
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # 'put'
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        return price
    except:
        return np.nan

def calculate_option_greeks(S, K, T, r, q, sigma, option_type):
    if sigma <= 0 or T <= 0:
        return {'delta': np.nan, 'gamma': np.nan, 'theta': np.nan, 'vega': np.nan, 'rho': np.nan}

    try:
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            delta = np.exp(-q * T) * norm.cdf(d1)
            theta = (-S * sigma * np.exp(-q * T) * norm.pdf(d1) / (2 * np.sqrt(T)) 
                     - r * K * np.exp(-r * T) * norm.cdf(d2) 
                     + q * S * np.exp(-q * T) * norm.cdf(d1)) / 365
        else:  # 'put'
            delta = -np.exp(-q * T) * norm.cdf(-d1)
            theta = (-S * sigma * np.exp(-q * T) * norm.pdf(d1) / (2 * np.sqrt(T)) 
                     + r * K * np.exp(-r * T) * norm.cdf(-d2) 
                     - q * S * np.exp(-q * T) * norm.cdf(-d1)) / 365

        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100  # Divided by 100 for percentage
        rho = K * T * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2) / 100  # Divided by 100 for percentage

        return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}
    except:
        return {'delta': np.nan, 'gamma': np.nan, 'theta': np.nan, 'vega': np.nan, 'rho': np.nan}

def get_option_data(ticker_symbol, expiration_date, option_type):
    ticker = yf.Ticker(ticker_symbol)
    options = ticker.option_chain(expiration_date)
    
    return options.calls if option_type == 'call' else options.puts

def main():
    ticker_symbol = input("Enter the ticker symbol: ").upper()
    ticker = yf.Ticker(ticker_symbol)
    
    print("\nAvailable expiration dates:")
    for i, date in enumerate(ticker.options):
        print(f"{i+1}. {date}")
    
    date_index = int(input("\nSelect expiration date (enter number): ")) - 1
    expiration_date = ticker.options[date_index]
    
    option_type = input("Enter option type (call/put): ").lower()
    
    options_data = get_option_data(ticker_symbol, expiration_date, option_type)
    
    current_price = ticker.history(period="1d")['Close'].iloc[-1]
    risk_free_rate = get_risk_free_rate(datetime.strptime(expiration_date, "%Y-%m-%d"))
    dividend_yield = get_dividend_yield(ticker)
    
    time_to_expiration = trading_days_to_expiration(expiration_date)

    results = []
    for _, option in options_data.iterrows():
        strike_price = option['strike']
        market_price = option['lastPrice']
        implied_vol = option['impliedVolatility']
        
        theoretical_price = black_scholes_option_price(current_price, strike_price, time_to_expiration, risk_free_rate, dividend_yield, implied_vol, option_type)
        greeks = calculate_option_greeks(current_price, strike_price, time_to_expiration, risk_free_rate, dividend_yield, implied_vol, option_type)
        
        intrinsic_value = max(0, current_price - strike_price) if option_type == 'call' else max(0, strike_price - current_price)
        
        results.append({
            'Strike': strike_price,
            'Market Price': market_price,
            'Theoretical Price': theoretical_price,
            'Implied Volatility': implied_vol,
            'Intrinsic Value': intrinsic_value,
            **greeks
        })

    results_df = pd.DataFrame(results)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(results_df)

if __name__ == "__main__":
    main()
