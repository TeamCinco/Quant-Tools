import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, timedelta

def get_risk_free_rate():
    while True:
        try:
            r = float(input("Enter the annual risk-free rate (e.g., enter 5 for 5%): "))
            r_decimal = r / 100
            return r_decimal
        except ValueError:
            print("Invalid input. Please enter a numerical value.")

def get_dividend_yield(ticker):
    try:
        return ticker.info['dividendYield'] if ticker.info['dividendYield'] is not None else 0
    except:
        return 0

def trading_days_to_expiration(expiration_date):
    now = datetime.now().date()
    expiration = datetime.strptime(expiration_date, "%Y-%m-%d").date()
    trading_days = pd.bdate_range(now, expiration).size
    years_to_expiration = trading_days / 252
    return max(years_to_expiration, 1 / 252)

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

def calculate_option_greeks(S, K, T, r, q, sigma, option_type, option_style='european'):
    if sigma <= 0 or T <= 0:
        return {'delta': np.nan, 'gamma': np.nan, 'theta': np.nan, 'vega': np.nan, 'rho': np.nan}

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    n_d1 = norm.pdf(d1)

    if option_style == 'european':
        if option_type == 'call':
            delta = np.exp(-q * T) * N_d1
            theta = (-((S * sigma * np.exp(-q * T) * n_d1) / (2 * sqrt_T))
                     - r * K * np.exp(-r * T) * N_d2
                     + q * S * np.exp(-q * T) * N_d1)
        else:
            delta = -np.exp(-q * T) * norm.cdf(-d1)
            theta = (-((S * sigma * np.exp(-q * T) * n_d1) / (2 * sqrt_T))
                     + r * K * np.exp(-r * T) * norm.cdf(-d2)
                     - q * S * np.exp(-q * T) * norm.cdf(-d1))

        gamma = (n_d1 * np.exp(-q * T)) / (S * sigma * sqrt_T)
        vega = (S * sqrt_T * n_d1 * np.exp(-q * T)) / 100
        rho = K * T * np.exp(-r * T) * (N_d2 if option_type == 'call' else -norm.cdf(-d2)) / 100
        # Convert theta to daily value
        theta = theta / 365
    else:
        # For American options, we approximate Greeks using finite differences
        h = 0.01
        price = binomial_american_option_price(S, K, T, r, q, sigma, 100, option_type)
        price_up = binomial_american_option_price(S + h, K, T, r, q, sigma, 100, option_type)
        price_down = binomial_american_option_price(S - h, K, T, r, q, sigma, 100, option_type)
        delta = (price_up - price_down) / (2 * h)
        gamma = (price_up - 2 * price + price_down) / (h ** 2)
        
        T_h = T - 1 / 365  # One day decrement
        if T_h <= 0:
            theta = np.nan
        else:
            price_theta = binomial_american_option_price(S, K, T_h, r, q, sigma, 100, option_type)
            theta = (price_theta - price) / (1 / 365)
        
        sigma_h = sigma + 0.01
        price_vega = binomial_american_option_price(S, K, T, r, q, sigma_h, 100, option_type)
        vega = (price_vega - price) / 0.01 / 100
        
        r_h = r + 0.01
        price_rho = binomial_american_option_price(S, K, T, r_h, q, sigma, 100, option_type)
        rho = (price_rho - price) / 0.01 / 100

    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}

def binomial_american_option_price(S, K, T, r, q, sigma, N, option_type):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    discount = np.exp(-r * dt)
    
    # Initialize asset prices at maturity
    asset_prices = S * d ** np.arange(N, -1, -1) * u ** np.arange(0, N+1, 1)
    
    # Initialize option values at maturity
    if option_type == 'call':
        option_values = np.maximum(0, asset_prices - K)
    else:
        option_values = np.maximum(0, K - asset_prices)
    
    # Backward induction
    for i in range(N-1, -1, -1):
        asset_prices = asset_prices[:-1] * u
        option_values = (p * option_values[:-1] + (1 - p) * option_values[1:]) * discount
        # Check for early exercise
        if option_type == 'call':
            exercise_values = np.maximum(0, asset_prices - K)
        else:
            exercise_values = np.maximum(0, K - asset_prices)
        option_values = np.maximum(option_values, exercise_values)
    
    return option_values[0]

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
    option_style = input("Enter option style (european/american): ").lower()
    
    options_data = get_option_data(ticker_symbol, expiration_date, option_type)
    
    current_price = ticker.history(period="1d")['Close'].iloc[-1]
    risk_free_rate = get_risk_free_rate()
    dividend_yield = get_dividend_yield(ticker)
    
    time_to_expiration = trading_days_to_expiration(expiration_date)

    results = []
    N = 100  # Number of binomial steps for American option pricing
    
    for _, option in options_data.iterrows():
        strike_price = option['strike']
        bid_price = option['bid']
        ask_price = option['ask']
        if bid_price == 0 and ask_price == 0:
            continue
        mid_price = (bid_price + ask_price) / 2 if bid_price != 0 and ask_price != 0 else option['lastPrice']
        implied_vol = option['impliedVolatility']
        
        if implied_vol == 0 or np.isnan(implied_vol):
            continue  # Skip options with invalid implied volatility
        
        if option_style == 'european':
            theoretical_price = black_scholes_option_price(current_price, strike_price, time_to_expiration, risk_free_rate, dividend_yield, implied_vol, option_type)
            greeks = calculate_option_greeks(current_price, strike_price, time_to_expiration, risk_free_rate, dividend_yield, implied_vol, option_type, option_style='european')
        elif option_style == 'american':
            theoretical_price = binomial_american_option_price(current_price, strike_price, time_to_expiration, risk_free_rate, dividend_yield, implied_vol, N, option_type)
            greeks = calculate_option_greeks(current_price, strike_price, time_to_expiration, risk_free_rate, dividend_yield, implied_vol, option_type, option_style='american')
        else:
            print("Invalid option style.")
            return
        
        intrinsic_value = max(0, current_price - strike_price) if option_type == 'call' else max(0, strike_price - current_price)
        
        results.append({
            'Strike': strike_price,
            'Bid': bid_price,
            'Ask': ask_price,
            'Mid Price': mid_price,
            'Theoretical Price': theoretical_price,
            'Implied Volatility': implied_vol,
            'Intrinsic Value': intrinsic_value,
            'Delta': greeks['delta'],
            'Gamma': greeks['gamma'],
            'Theta': greeks['theta'],
            'Vega': greeks['vega'],
            'Rho': greeks['rho']
        })

    if len(results) == 0:
        print("No options data available for the selected parameters.")
        return

    results_df = pd.DataFrame(results)
    pd.set_option('display.float_format', '{:.4f}'.format)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    print(f"\nOption Style: {option_style.capitalize()}")
    print(results_df)

if __name__ == "__main__":
    main()
