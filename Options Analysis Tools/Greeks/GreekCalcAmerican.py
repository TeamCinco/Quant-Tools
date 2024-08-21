import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, timedelta

def get_risk_free_rate():
    # Return the latest available FEDFUNDS rate from FRED data
    return 0.048  # 5.33% as of July 2024

def get_dividend_yield(ticker):
    try:
        return ticker.info['dividendYield']
    except:
        return 0

def trading_days_to_expiration(expiration_date):
    now = datetime.now()
    expiration = datetime.strptime(expiration_date, "%Y-%m-%d") + timedelta(hours=16)
    
    seconds_to_expiration = (expiration - now).total_seconds()
    years_to_expiration = seconds_to_expiration / (252 * 24 * 60 * 60)
    
    return max(years_to_expiration, 1/252)

def black_scholes_option_price(S, K, T, r, q, sigma, option_type):
    if sigma <= 0 or T <= 0:
        return np.nan

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return price

def calculate_option_greeks(S, K, T, r, q, sigma, option_type):
    if sigma <= 0 or T <= 0:
        return {'delta': np.nan, 'gamma': np.nan, 'theta': np.nan, 'vega': np.nan, 'rho': np.nan}

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    n_d1 = norm.pdf(d1)

    if option_type == 'call':
        delta = np.exp(-q * T) * N_d1
        theta = (-((S * sigma * np.exp(-q * T) * n_d1) / (2 * sqrt_T)) 
                 - r * K * np.exp(-r * T) * N_d2
                 + q * S * np.exp(-q * T) * N_d1)
    else:
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (-((S * sigma * np.exp(-q * T) * n_d1) / (2 * sqrt_T)) 
                 + r * K * np.exp(-r * T) * (1 - N_d2)
                 - q * S * np.exp(-q * T) * (1 - N_d1))

    gamma = (n_d1 * np.exp(-q * T)) / (S * sigma * sqrt_T)
    vega = S * sqrt_T * n_d1 * np.exp(-q * T) / 100
    rho = K * T * np.exp(-r * T) * (N_d2 if option_type == 'call' else (1 - N_d2)) / 100

    # Convert theta to daily value
    theta = theta / 365

    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}

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
    risk_free_rate = get_risk_free_rate()
    dividend_yield = get_dividend_yield(ticker)
    
    time_to_expiration = trading_days_to_expiration(expiration_date)

    results = []
    for _, option in options_data.iterrows():
        strike_price = option['strike']
        bid_price = option['bid']
        ask_price = option['ask']
        mid_price = (bid_price + ask_price) / 2
        implied_vol = option['impliedVolatility']
        
        theoretical_price = black_scholes_option_price(current_price, strike_price, time_to_expiration, risk_free_rate, dividend_yield, implied_vol, option_type)
        greeks = calculate_option_greeks(current_price, strike_price, time_to_expiration, risk_free_rate, dividend_yield, implied_vol, option_type)
        
        intrinsic_value = max(0, current_price - strike_price) if option_type == 'call' else max(0, strike_price - current_price)
        
        results.append({
            'Strike': strike_price,
            'Bid': bid_price,
            'Ask': ask_price,
            'Mid Price': mid_price,
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