Certainly! I'll reformat the explanation using Markdown syntax with hashtags for headers/titles. Here's the revised version for the README:

# Quant-Tools

## RSI Crossover Folder

### How to Use:

1. Run the script and enter the required inputs:
   - Ticker symbol of the stock you want to analyze
   - Variable to analyze (e.g., DAAA for Federal Funds Rate)
   - Start date for the analysis (YYYY-MM-DD format)

2. The script will:
   - Download historical data for the specified stock and the S&P 500 (SPY)
   - Implement a Mean Reversion RSI Strategy
   - Backtest the strategy and compare it to a buy-and-hold approach
   - Generate plots of the backtest results and RSI

3. Output includes:
   - Strategy performance metrics
   - Comparison with buy-and-hold returns
   - Linear regression analysis of the stock vs. the specified variable
   - Identification of outliers (days with significant price changes)
   - Plots of the backtest results, RSI, and outliers

This tool is useful for traders looking to implement a mean reversion strategy based on RSI and analyze its performance against market benchmarks.

## Dollar Mean Reversion Backtesting Folder

### How to Use:

1. Run the script (no user input required)

2. The script will:
   - Download historical SPY data from 2000 to the present
   - Implement a Dollar Range Strategy
   - Backtest the strategy and compare it to a buy-and-hold approach

3. Output includes:
   - Strategy performance metrics
   - Comparison with buy-and-hold returns
   - A plot of the backtest results

This tool is useful for traders interested in testing a simple mean reversion strategy based on dollar ranges against the S&P 500.

## Macro Economic Analysis Tools Folder

### How to Use:

1. Run the script and enter the required inputs:
   - FRED symbol for the macroeconomic indicator
   - Stock symbol
   - Number of years to analyze
   - Choose the independent variable (FRED or Stock)

2. The script will:
   - Fetch data for both the stock and the macroeconomic indicator
   - Perform linear regression analysis
   - Generate a scatter plot with the regression line

3. Output includes:
   - Linear regression results (slope, intercept, R-squared, p-value)
   - A scatter plot showing the relationship between the stock and the macroeconomic indicator

This tool is useful for investors looking to understand the relationship between stock prices and macroeconomic factors.

## Options Stats Tools Folder

### How to Use:

1. Run the script and enter a ticker symbol

2. The script will:
   - Fetch available expiration dates for the stock's options
   - Prompt you to select a range of expiration dates
   - Generate implied volatility surfaces for both calls and puts
   - Plot the stock price for the past 6 months

3. Output includes:
   - 3D plots of implied volatility surfaces for calls and puts
   - A plot of the stock's price over the past 6 months

This tool is useful for options traders to visualize implied volatility patterns and analyze options pricing across different strikes and expirations.

## Monte Carlo Simulation Folder

### How to Use:

1. Run the script and enter the required inputs:
   - Stock ticker symbol
   - FRED macroeconomic indicator ticker
   - Number of days to predict (minimum 5)
   - Select which standard deviations to include in the analysis

2. The script will:
   - Fetch historical stock and macroeconomic data
   - Perform regression analysis
   - Run Monte Carlo simulations
   - Generate various statistical analyses and plots

3. Output includes:
   - Regression results
   - Monte Carlo simulation plots
   - Standard deviation charts for daily, weekly, and monthly price changes
   - Excel or CSV file with filtered Monte Carlo paths

This tool is useful for investors looking to model potential future stock price movements and analyze the probability of various outcomes.

## Stock Performance Folder

### How to Use:

1. Run the script and enter the required inputs:
   - Ticker symbol

2. The script will:
   - Download historical stock data
   - Calculate daily price changes and bin them
   - Generate frequency tables and plots
   - Calculate and plot standard deviations for daily, weekly, and monthly price changes

3. Output includes:
   - Frequency table of daily price changes
   - Histogram of daily price changes
   - Normal distribution fits for daily, weekly, and monthly price changes
   - Plots showing price levels at different standard deviations

This tool is useful for analyzing the historical price performance and volatility of a stock.

## Financial Metric Compare and Visuals Folder

### How to Use:

1. Run the script and enter the required inputs:
   - Ticker symbol
   - Statement period (annual or quarterly)

2. The script will:
   - Fetch financial data (income statement, balance sheet, cash flow)
   - Display available financial metrics
   - Prompt you to select metrics for visualization

3. Output includes:
   - Line plots of selected financial metrics over time
   - Dollar values formatted in billions or millions as appropriate

This tool is useful for visualizing and comparing various financial metrics of a company over time, helping investors analyze financial performance trends.