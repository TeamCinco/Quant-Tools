Here's a README for the repo based on the code you've provided:

# Quant-Tools

This repository contains a collection of quantitative analysis tools for financial markets. The tools cover various aspects of market analysis, including backtesting, options analysis, correlation studies, and Monte Carlo simulations.

## Table of Contents

1. [Backtesting Tools](#backtesting-tools)
2. [Correlation Analysis](#correlation-analysis)
3. [Financial Analysis Tools](#financial-analysis-tools)
4. [Macro Economic Analysis Tools](#macro-economic-analysis-tools)
5. [Options Analysis Tools](#options-analysis-tools)
6. [Simulation Tools](#simulation-tools)
7. [Stock Performance Analysis Tools](#stock-performance-analysis-tools)

## Backtesting Tools

This section includes tools for backtesting trading strategies:

- `ActualProbabilitySTD.py`: Calculates actual probabilities based on standard deviations.
- `MCSTDDollarMeanReg.py`: Implements a Monte Carlo simulation for dollar mean reversion strategy.
- Dollar Mean Reversion strategies.
- RSI crossover strategies.

## Correlation Analysis

Tools for analyzing correlations between different financial instruments:

- `HeatmapCorrelation.py`: Generates correlation heatmaps.
- `HeatmapCorrelationPriceChange.py`: Analyzes correlations based on price changes.
- `HeatmapCorrelationPriceData.py`: Studies correlations using price data.
- `PriceDeltaAndPrice.py`: Examines relationships between price deltas and prices.

## Financial Analysis Tools

Visualizations and tools for financial analysis:

- `MetricVisuals.py`: Creates visualizations for various financial metrics.
- `PerformanceCompare.py`: Compares performance across different financial instruments.

## Macro Economic Analysis Tools

Tools for analyzing macroeconomic factors:

- Linear regression tools for macroeconomic analysis.
- Time series analysis tools like `MacroCharts.py`.

## Options Analysis Tools

Tools for options analysis and Greeks calculations:

- `GreekCalcAmerican.py`: Calculates Greeks for American options.
- `GreeksCalculator.py`: A general-purpose Greeks calculator.
- Options statistics tools.

## Simulation Tools

Monte Carlo simulation tools:

- `MonteCarlo25Years.py`: Runs a 25-year Monte Carlo simulation.
- `MonteCarloShortRandom.py`: Implements a short-term random Monte Carlo simulation.
- `MonteCarloStocks.py`: Monte Carlo simulation for stock price movements.
- Various other Monte Carlo tools for different scenarios.

## Stock Performance Analysis Tools

Tools for analyzing stock performance:

- News analysis tools.
- Screener tools for filtering stocks based on various criteria.
- Standard deviation analysis tools.
- Linear regression tools for stock vs macro analysis.

## Usage

Each script can be run independently. Most scripts will prompt for user input such as ticker symbols, date ranges, or specific parameters for analysis.

## Requirements

The project requires Python 3.x and several libraries including:

- yfinance
- pandas
- numpy
- matplotlib
- scipy
- scikit-learn
- pandas_datareader
- backtesting

You can install these dependencies using pip:

```
pip install yfinance pandas numpy matplotlib scipy scikit-learn pandas_datareader backtesting
```

## Contributing

Feel free to fork this repository and submit pull requests with improvements or additional tools.

## License

[Specify your license here]

This README provides an overview of the repository structure and the tools available. You may want to add more detailed instructions for each tool, installation steps, and any specific usage guidelines as needed.
