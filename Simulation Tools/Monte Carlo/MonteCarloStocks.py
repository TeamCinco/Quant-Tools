import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Fetch stock data
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
    return stock_data

# Fetch FRED macro data
def get_fred_data(series_id, start_date, end_date):
    data = pdr.get_data_fred(series_id, start=start_date, end=end_date)
    return data

# Normalize and Log Transform Data
def normalize_and_log(data):
    data = data.dropna()  # Remove any NaN values
    log_data = np.log(data)   # Log transform the data
    normalized_data = (log_data - log_data.min()) / (log_data.max() - log_data.min())  # Normalize data
    return normalized_data, log_data

# De-normalize and De-log data
def denormalize_and_delog(normalized_data, log_data):
    min_val = log_data.min()
    max_val = log_data.max()
    denormalized_data = normalized_data * (max_val - min_val) + min_val  # De-normalize
    original_data = np.exp(denormalized_data)  # De-log (inverse of log is exp)
    return original_data

# Linear Regression between Stock and Macro Indicator
def perform_regression(stock_data, macro_data):
    model = LinearRegression()
    X = macro_data.values.reshape(-1, 1)  # Reshape data for sklearn
    y = stock_data.values
    model.fit(X, y)
    return model

# Monte Carlo Simulation
def monte_carlo_simulation(current_price, projected_daily_return, stdReturn, T, mc_sims=1000):
    portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)
    for m in range(0, mc_sims):
        Z = np.random.normal(size=T)  # Generate random variables for simulation
        dailyReturns = projected_daily_return + stdReturn * Z  # Simulate daily returns
        portfolio_sims[:, m] = np.cumprod(1 + dailyReturns) * current_price  # Calculate cumulative returns
    
    return portfolio_sims

# Main function to run the analysis
def main():
    # User Inputs
    stock_ticker = input("Enter the stock ticker symbol: ")
    macro_ticker = input("Enter FRED macroeconomic indicator ticker: ")
    start_date = '2003-01-01'
    end_date = '2023-01-01'
    
    # Ask the user for the prediction period
    prediction_period = int(input("Enter the number of days to predict (minimum 5 days): "))
    if prediction_period < 5:
        print("Prediction period is too short. Setting to minimum of 5 days.")
        prediction_period = 5
    
    # Fetch Data
    stock_data = get_stock_data(stock_ticker, start_date, end_date)
    macro_data = get_fred_data(macro_ticker, start_date, end_date)
    
    # Align data for analysis
    combined_data = pd.concat([stock_data, macro_data], axis=1).dropna()
    stock_aligned = combined_data.iloc[:, 0]
    macro_aligned = combined_data.iloc[:, 1]
    
    # Normalize and log transform the data
    stock_normalized, stock_log = normalize_and_log(stock_aligned)
    macro_normalized, macro_log = normalize_and_log(macro_aligned)
    
    # Perform Linear Regression
    regression_model = perform_regression(stock_normalized, macro_normalized)
    print(f"Regression Coefficient: {regression_model.coef_[0]}")
    print(f"Regression Intercept: {regression_model.intercept_}")
    
    # Plot the Linear Regression result
    plt.figure(figsize=(10, 5))
    plt.scatter(macro_normalized, stock_normalized, label="Data Points")
    plt.plot(macro_normalized, regression_model.predict(macro_normalized.values.reshape(-1, 1)), color='red', label="Regression Line")
    plt.xlabel(f'{macro_ticker} Data')
    plt.ylabel(f'{stock_ticker} Price')
    plt.title(f'Linear Regression of {stock_ticker} vs {macro_ticker}')
    plt.legend()
    plt.show()
    
    # De-normalize and de-log the data for Monte Carlo simulation
    stock_original = denormalize_and_delog(stock_normalized, stock_log)
    macro_original = denormalize_and_delog(macro_normalized, macro_log)
    
    # Use original data for Monte Carlo Simulation
    mean_return = regression_model.coef_[0]
    std_return = stock_original.pct_change().std()
    current_price = stock_original.iloc[-1]
    
    # Ensure mean return and standard deviation are within reasonable limits
    mean_return = max(min(mean_return, 0.001), -0.001)
    std_return = max(min(std_return, 0.02), 0.005)
    
    # Run Monte Carlo for the user-specified prediction period
    mc_sims_result = monte_carlo_simulation(current_price, mean_return, std_return, T=prediction_period)
    
    # Visualization of Monte Carlo Simulation
    plt.figure(figsize=(10, 5))
    plt.plot(mc_sims_result)
    plt.ylabel(f'{stock_ticker} Price ($)')
    plt.xlabel('Days')
    plt.title(f'MC simulation of {stock_ticker} over {prediction_period} days')
    plt.show()

if __name__ == "__main__":
    main()
