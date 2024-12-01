import yfinance as yf
import numpy as np
from datetime import datetime

def binomial_pricing(S, K, T, r, sigma, n=100, option_type='american'):
    if T <= 0:
        call_price = max(0, S - K)
        put_price = max(0, K - S)
        return call_price, put_price

    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    call_prices = np.zeros((n + 1, n + 1))
    put_prices = np.zeros((n + 1, n + 1))

    # Terminal payoffs
    for i in range(n + 1):
        stock_price = S * (u ** (n - i)) * (d ** i)
        call_prices[i, n] = max(0, stock_price - K)
        put_prices[i, n] = max(0, K - stock_price)

    # Backward induction
    for j in range(n - 1, -1, -1):
        for i in range(j + 1):
            stock_price = S * (u ** (j - i)) * (d ** i)
            
            # Expected value (same for both American and European)
            call_hold = np.exp(-r * dt) * (p * call_prices[i, j + 1] + (1 - p) * call_prices[i + 1, j + 1])
            put_hold = np.exp(-r * dt) * (p * put_prices[i, j + 1] + (1 - p) * put_prices[i + 1, j + 1])
            
            if option_type.lower() == 'american':
                # For American options, compare with immediate exercise value
                call_exercise = max(0, stock_price - K)
                put_exercise = max(0, K - stock_price)
                
                call_prices[i, j] = max(call_hold, call_exercise)
                put_prices[i, j] = max(put_hold, put_exercise)
            else:  # European
                call_prices[i, j] = call_hold
                put_prices[i, j] = put_hold

    return call_prices[0, 0], put_prices[0, 0]

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    
    try:
        S = info['currentPrice']
    except KeyError:
        S = stock.history(period="1d")['Close'].iloc[-1]
    
    hist = stock.history(period="1y")
    returns = np.log(hist['Close'] / hist['Close'].shift(1))
    sigma = returns.std() * np.sqrt(252)
    
    try:
        r = yf.Ticker("^TNX").info['regularMarketPrice'] / 100
    except:
        r = 0.0468

    return S, sigma, r

def get_options_data(ticker):
    stock = yf.Ticker(ticker)
    return stock.options, stock.option_chain

def display_options(exp_dates, option_chain, current_price):
    while True:
        print("\nAvailable expiration dates:")
        for i, date in enumerate(exp_dates):
            print(f"{i + 1}. {date}")
        
        try:
            choice = int(input("\nSelect an expiration date (enter the number): ")) - 1
            if choice < 0 or choice >= len(exp_dates):
                print("Invalid selection. Please try again.")
                continue
            selected_date = exp_dates[choice]
            
            options = option_chain(selected_date)
            strikes = sorted(set(options.calls['strike'].tolist() + options.puts['strike'].tolist()))
            
            print("\nAll available strike prices:")
            print(", ".join([f"${strike:.2f}" for strike in strikes]))
            
            closest_strike = min(strikes, key=lambda x: abs(x - current_price))
            closest_index = strikes.index(closest_strike)
            relevant_strikes = strikes[max(0, closest_index - 5):closest_index + 6]
            
            print("\nRelevant strike prices (5 above and below current price):")
            for i, strike in enumerate(relevant_strikes):
                print(f"{i + 1}. {strike}")
            
            choice = int(input("\nSelect a strike price for theoretical pricing (enter the number): ")) - 1
            if choice < 0 or choice >= len(relevant_strikes):
                print("Invalid selection. Please try again.")
                continue
            selected_strike = relevant_strikes[choice]
            
            return selected_date, selected_strike, options, relevant_strikes
        except ValueError:
            print("Invalid input. Please enter a number.")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again.")

def analyze_price_range(S, K, T, r, sigma):
    lower_bound = max(0, S - 5)
    upper_bound = S + 5
    price_range = np.arange(lower_bound, upper_bound + 0.5, 0.5)
    results = []

    for price in price_range:
        # Changed S to price in the binomial_pricing call
        call_price, put_price = binomial_pricing(price, K, T, r, sigma, n=100, option_type='american')
        results.append((price, call_price, put_price))

    return results

def main():
    while True:
        try:
            ticker = input("Enter the stock ticker: ").upper()
            S, sigma, r = get_stock_data(ticker)
            print(f"\nCurrent stock price for {ticker}: ${S:.2f}")  # Add this line
            
            exp_dates, option_chain = get_options_data(ticker)
            
            if not exp_dates:
                print(f"No options data available for {ticker}. Please try another ticker.")
                continue

            selected_date, K, options, relevant_strikes = display_options(exp_dates, option_chain, S)
            
            today = datetime.now().date()
            expiration = datetime.strptime(selected_date, "%Y-%m-%d").date()
            T = max((expiration - today).days / 365, 0)

            call_price, put_price = binomial_pricing(S, K, T, r, sigma, option_type='american')
            results = analyze_price_range(S, K, T, r, sigma)

            print(f"\nAnalysis for {ticker}")
            print(f"Current stock price: ${S:.2f}")
            print(f"Volatility: {sigma:.2%}")
            print(f"Risk-free rate: {r:.2%}")
            print(f"Selected expiration date: {selected_date}")
            print(f"Selected strike price: ${K:.2f}")
            print(f"Time to expiration: {T:.4f} years ({max(0, int(T*365))} days)")
            print(f"Theoretical call price: ${call_price:.2f}")
            print(f"Theoretical put price: ${put_price:.2f}")

            call_market = options.calls[options.calls['strike'] == K]['lastPrice'].values[0]
            put_market = options.puts[options.puts['strike'] == K]['lastPrice'].values[0]
            
            print(f"\nMarket call price: ${call_market:.2f}")
            print(f"Market put price: ${put_market:.2f}")

            print("\nOption Chain Data (5 strikes above and below):")
            print("Strike | Call Bid | Call Ask | Put Bid | Put Ask")
            print("------------------------------------------------")
            for strike in relevant_strikes:
                try:
                    call_data = options.calls[options.calls['strike'] == strike]
                    put_data = options.puts[options.puts['strike'] == strike]
                    
                    if not call_data.empty and not put_data.empty:
                        call_row = call_data.iloc[0]
                        put_row = put_data.iloc[0]
                        print(f"${strike:<6.2f} | ${call_row['bid']:<8.2f} | ${call_row['ask']:<8.2f} | ${put_row['bid']:<7.2f} | ${put_row['ask']:.2f}")
                    else:
                        print(f"${strike:<6.2f} | No data available")
                except Exception as e:
                    print(f"${strike:<6.2f} | No data available")
            print("\nTheoretical Option Prices for Selected Strike:")
            print("Stock Price | Call Price | Put Price")
            print("-------------------------------------")
            for price, call_price, put_price in results:
                print(f"${price:<10.2f} | ${call_price:<10.2f} | ${put_price:.2f}")

            break
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again with a different ticker or option.")

if __name__ == '__main__':
    main()
