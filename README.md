# Quant-Tools

A comprehensive collection of quantitative analysis and trading tools for financial markets.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Modules](#modules)
  - [RSI Crossover](#rsi-crossover)
  - [Dollar Mean Reversion Backtesting](#dollar-mean-reversion-backtesting)
  - [Macro Economic Analysis Tools](#macro-economic-analysis-tools)
  - [Options Stats Tools](#options-stats-tools)
  - [Monte Carlo Simulation](#monte-carlo-simulation)
  - [Stock Performance Analysis](#stock-performance-analysis)
  - [Financial Metric Compare and Visuals](#financial-metric-compare-and-visuals)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Quant-Tools is a powerful suite of Python-based tools designed for quantitative analysis, backtesting, and visualization of financial market data. This project aims to provide traders, analysts, and researchers with a comprehensive set of tools to analyze stocks, options, and macroeconomic indicators.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/Quant-Tools.git
   ```

2. Navigate to the project directory:
   ```
   cd Quant-Tools
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Project Structure

The project is organized into several main folders, each containing specific tools and functionalities:

- `RSI_Crossover/`
- `Dollar_Mean_Reversion_Backtesting/`
- `Macro_Economic_Analysis_Tools/`
- `Options_Stats_Tools/`
- `Monte_Carlo_Simulation/`
- `Stock_Performance_Analysis/`
- `Financial_Metric_Compare_and_Visuals/`

## Modules

### RSI Crossover

This module implements a Mean Reversion RSI Strategy for stock analysis.

#### Features:
- Downloads historical data for a specified stock and the S&P 500
- Implements and backtests the Mean Reversion RSI Strategy
- Compares strategy performance to a buy-and-hold approach
- Generates plots of backtest results and RSI

#### Usage:
1. Run the script and enter the required inputs:
   - Ticker symbol of the stock to analyze
   - Variable to analyze (e.g., DAAA for Federal Funds Rate)
   - Start date for the analysis (YYYY-MM-DD format)

2. The script will process the data and generate output including:
   - Strategy performance metrics
   - Comparison with buy-and-hold returns
   - Linear regression analysis of the stock vs. the specified variable
   - Plots of backtest results, RSI, and outliers

### Dollar Mean Reversion Backtesting

This module implements and backtests a Dollar Range Strategy on historical SPY data.

#### Features:
- Downloads historical SPY data from 2000 to the present
- Implements a Dollar Range Strategy
- Backtests the strategy and compares it to a buy-and-hold approach

#### Usage:
1. Run the script (no user input required)
2. The script will process the data and output:
   - Strategy performance metrics
   - Comparison with buy-and-hold returns
   - A plot of the backtest results

### Macro Economic Analysis Tools

This module analyzes the relationship between stock prices and macroeconomic indicators.

#### Features:
- Fetches data for both stocks and macroeconomic indicators
- Performs linear regression analysis
- Generates scatter plots with regression lines

#### Usage:
1. Run the script and enter the required inputs:
   - FRED symbol for the macroeconomic indicator
   - Stock symbol
   - Number of years to analyze
   - Choose the independent variable (FRED or Stock)

2. The script will output:
   - Linear regression results (slope, intercept, R-squared, p-value)
   - A scatter plot showing the relationship between the stock and the macroeconomic indicator

### Options Stats Tools

This module analyzes and visualizes options data for a given stock.

#### Features:
- Fetches available expiration dates for a stock's options
- Generates implied volatility surfaces for calls and puts
- Plots the stock price for the past 6 months

#### Usage:
1. Run the script and enter a ticker symbol
2. The script will:
   - Fetch available expiration dates
   - Prompt you to select a range of expiration dates
   - Generate and display visualizations

### Monte Carlo Simulation

This module performs Monte Carlo simulations for stock price prediction.

#### Features:
- Fetches historical stock and macroeconomic data
- Performs regression analysis
- Runs Monte Carlo simulations
- Generates statistical analyses and plots

#### Usage:
1. Run the script and enter the required inputs:
   - Stock ticker symbol
   - FRED macroeconomic indicator ticker
   - Number of days to predict (minimum 5)
   - Select which standard deviations to include in the analysis

2. The script will output:
   - Regression results
   - Monte Carlo simulation plots
   - Standard deviation charts for daily, weekly, and monthly price changes
   - Excel or CSV file with filtered Monte Carlo paths

### Stock Performance Analysis

This module analyzes the historical price performance and volatility of a stock.

#### Features:
- Downloads historical stock data
- Calculates daily price changes and bins them
- Generates frequency tables and plots
- Calculates and plots standard deviations for daily, weekly, and monthly price changes

#### Usage:
1. Run the script and enter the ticker symbol
2. The script will output:
   - Frequency table of daily price changes
   - Histogram of daily price changes
   - Normal distribution fits for daily, weekly, and monthly price changes
   - Plots showing price levels at different standard deviations

### Financial Metric Compare and Visuals

This module visualizes and compares various financial metrics of a company over time.

#### Features:
- Fetches financial data (income statement, balance sheet, cash flow)
- Displays available financial metrics
- Visualizes selected metrics over time

#### Usage:
1. Run the script and enter the required inputs:
   - Ticker symbol
   - Statement period (annual or quarterly)
2. Select metrics for visualization
3. The script will generate line plots of selected financial metrics over time

## Usage

Each module can be run independently. Navigate to the respective folder and run the Python script for the desired analysis.

## Contributing

Contributions to Quant-Tools are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.