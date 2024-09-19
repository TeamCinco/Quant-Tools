# Quant-Tools

## Overview

Quant-Tools is a comprehensive suite of tools designed for quantitative analysis, backtesting strategies, options analysis, macroeconomic analysis, and more. This repository is structured to help traders, investors, and analysts perform advanced financial modeling and testing with Python.

## Table of Contents

1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Features](#features)
    - [RSI Crossover Strategies](#rsi-crossover-strategies)
    - [Dollar Mean Reversion Strategies](#dollar-mean-reversion-strategies)
    - [Options Analysis](#options-analysis)
    - [Monte Carlo Simulation](#monte-carlo-simulation)
    - [Macroeconomic Analysis](#macroeconomic-analysis)
    - [Stock Performance Analysis](#stock-performance-analysis)
4. [Usage](#usage)
5. [Backtesting Tools](#backtesting-tools)
6. [Macro Economic Analysis Tools](#macro-economic-analysis-tools)
7. [Options Stats Tools](#options-stats-tools)
8. [Monte Carlo Simulation Tools](#monte-carlo-simulation-tools)
9. [Stock Performance Tools](#stock-performance-tools)
10. [Financial Metric Compare and Visuals](#financial-metric-compare-and-visuals)
11. [Contributing](#contributing)
12. [License](#license)

## Installation

### Prerequisites

Ensure that you have the following installed on your machine:

- Python 3.8 or higher
- pip (Python package manager)

### Dependencies

This project requires several Python packages. Install them using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Additional Dependencies

Depending on your environment and use case, you may need additional libraries for GPU acceleration (e.g., CuPy). Ensure your environment is correctly configured if using these features.

### CUDA Setup (Optional)

If you plan to use GPU acceleration with CuPy, ensure you have a compatible CUDA version installed.

## Project Structure

The project is organized as follows:

```
Quant-Tools/
├── Backtesting Tools/
│   ├── RSI crossover/
│   ├── Dollar Mean Reversion/
│   ├── Outlier Analysis/
│   ├── Realistic backtesting external variables/
│   └── Return Analysis/
├── Monte Carlo Simulation/
├── Macro Economic Analysis Tools/
├── Options Stats Tools/
├── Stock Performance Tools/
├── Financial Metric Compare and Visuals/
├── README.md
└── requirements.txt
```

## Features

### RSI Crossover Strategies

This folder contains various strategies that utilize the Relative Strength Index (RSI) for mean reversion. It includes:

- Basic RSI crossover strategy.
- Strategies with additional conditions like stop-loss rules.
- Strategies that store trade data for further analysis.

### Dollar Mean Reversion Strategies

These strategies are based on mean reversion principles, focusing on price movements within a specified dollar range. The strategies work well for different market conditions and include features like customizable time frames and stop-loss levels.

### Options Analysis

Tools for analyzing options data, including implied volatility surfaces and plotting tools. Useful for options traders looking to visualize market conditions and make informed decisions.

### Monte Carlo Simulation

A set of tools for running Monte Carlo simulations on stock prices, accounting for various statistical factors. These tools help forecast potential future price movements and assess risk.

### Macroeconomic Analysis

Scripts designed to analyze the relationship between macroeconomic indicators (e.g., interest rates, GDP) and stock prices. This is useful for understanding how economic variables impact market behavior.

### Stock Performance Analysis

Tools for analyzing the historical performance of stocks, including standard deviation calculations, distribution fits, and visualizations. Ideal for understanding past price behavior and predicting future volatility.

## Usage

### Running Scripts



https://www.sec.gov/file/company-tickers

download the json file from above.......



1. **Clone the repository:**

   ```bash
   git clone https://github.com/TeamCinco/Quant-Tools.git
   cd Quant-Tools
   ```

2. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run individual scripts:**

   Navigate to the desired folder and execute the script:

   ```bash
   python <script_name>.py
   ```

### Configuring Parameters

Most scripts require user inputs such as ticker symbols, date ranges, and other parameters. Follow the on-screen prompts to enter the required data.

## Backtesting Tools

### Overview

The Backtesting Tools folder contains several subfolders for different strategies:

- **RSI Crossover:** Implements RSI-based mean reversion strategies.
- **Dollar Mean Reversion:** Tests strategies that rely on price movements within a fixed dollar range.
- **Outlier Analysis:** Identifies and analyzes outliers in price data.
- **Realistic Backtesting with External Variables:** Incorporates external macroeconomic factors into backtesting.
- **Return Analysis:** Extracts and analyzes returns from various backtesting scenarios.

### How to Use

1. Choose the appropriate script based on your analysis needs.
2. Run the script and provide the necessary inputs (tickers, dates, variables).
3. Review the output, which includes performance metrics, plots, and statistical analyses.

## Macro Economic Analysis Tools

### Overview

Tools designed to analyze the relationship between macroeconomic variables and stock prices. Useful for understanding how factors like interest rates affect stock performance.

### How to Use

1. Run the script and enter the required inputs (ticker, macroeconomic variable).
2. The script fetches data, performs linear regression, and generates scatter plots and statistical summaries.

## Options Stats Tools

### Overview

Scripts for options analysis, including implied volatility surface plots. These tools help traders visualize volatility across different strike prices and expiration dates.

### How to Use

1. Run the script and enter the ticker symbol.
2. Select the expiration dates and other preferences.
3. The script generates 3D volatility surfaces and stock price charts.

## Monte Carlo Simulation Tools

### Overview

Tools to perform Monte Carlo simulations on stock prices, using historical data to model future price movements. Includes functionality for regression analysis and probability assessments.

### How to Use

1. Run the script and enter the required inputs (ticker, days to predict, standard deviations).
2. Review the output, which includes simulation plots, confidence intervals, and statistical summaries.

## Stock Performance Tools

### Overview

Tools to analyze historical stock performance, including standard deviation calculations, frequency distributions, and normal distribution fits.

### How to Use

1. Run the script and enter the ticker symbol.
2. The script generates frequency tables, histograms, and distribution fits.
3. Use the output to assess historical volatility and price behavior.

## Financial Metric Compare and Visuals

### Overview

Scripts for comparing and visualizing financial metrics from company reports (income statement, balance sheet, cash flow). These tools are essential for fundamental analysis.

### How to Use

1. Run the script and enter the ticker symbol and statement period (annual or quarterly).
2. Select the financial metrics to visualize.
3. The script generates line plots of the selected metrics over time.

## Contributing

We welcome contributions from the community! If you have suggestions or improvements, feel free to open an issue or submit a pull request.

### Guidelines

- Ensure your code adheres to PEP 8 standards.
- Write clear and concise comments.
- Test your code before submitting.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

