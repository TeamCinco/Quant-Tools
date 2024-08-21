import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def get_financials(ticker, period):
    stock = yf.Ticker(ticker)
    
    if period == 'annual':
        income_stmt = stock.financials.T
        balance_sheet = stock.balance_sheet.T
        cashflow = stock.cashflow.T
    else:
        income_stmt = stock.quarterly_financials.T
        balance_sheet = stock.quarterly_balance_sheet.T
        cashflow = stock.quarterly_cashflow.T

    return income_stmt, balance_sheet, cashflow

def display_metrics(income_stmt, balance_sheet, cashflow):
    metrics = {}
    index = 1

    for df, name in [(income_stmt, 'income_stmt'), 
                     (balance_sheet, 'balance_sheet'), 
                     (cashflow, 'cashflow')]:
        for column in df.columns:
            metrics[index] = (name, column)
            index += 1

    for i in sorted(metrics.keys()):
        print(f"{i}. {metrics[i][0].replace('_', ' ').title()}: {metrics[i][1]}")

    return metrics

def billions(x, pos):
    'The two args are the value and tick position'
    return '$%1.1fB' % (x * 1e-9)

def millions(x, pos):
    'The two args are the value and tick position'
    return '$%1.1fM' % (x * 1e-6)

def plot_metrics(ticker, metrics, selected_indices):
    stock = yf.Ticker(ticker)

    for idx in selected_indices:
        statement, metric = metrics[idx]
        data = getattr(stock, statement).T[metric].fillna(0)  # Handle NaN values by filling them with zero

        plt.figure(figsize=(14, 7))
        plt.plot(data.index, data.values, label=f"{statement.replace('_', ' ').title()}: {metric}")
        plt.title(f"{ticker} - {statement.replace('_', ' ').title()}: {metric} Over Time")
        plt.xlabel('Date')
        plt.ylabel('Dollar Value ($)')

        if data.values.max() > 1e9:
            formatter = FuncFormatter(billions)
        else:
            formatter = FuncFormatter(millions)

        plt.gca().yaxis.set_major_formatter(formatter)

        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    ticker = input("Enter a ticker symbol: ").upper()
    period = input("Select statement period (annual/quarterly): ").lower()

    if period not in ['annual', 'quarterly']:
        print("Invalid period. Please enter 'annual' or 'quarterly'.")
        return

    income_stmt, balance_sheet, cashflow = get_financials(ticker, period)

    print("\nAvailable Financial Metrics:")
    metrics = display_metrics(income_stmt, balance_sheet, cashflow)

    while True:
        selected_indices_input = input("\nEnter the numbers of the metrics you want to plot, separated by commas: ")
        selected_indices_input = selected_indices_input.replace(" ", "").split(',')

        try:
            selected_indices = [int(i) for i in selected_indices_input]
            if all(idx in metrics for idx in selected_indices):
                break
            else:
                print("Invalid input. Some numbers do not correspond to available metrics.")
        except ValueError:
            print("Invalid input. Please enter numbers only, separated by commas.")

    plot_metrics(ticker, metrics, selected_indices)

if __name__ == "__main__":
    main()
