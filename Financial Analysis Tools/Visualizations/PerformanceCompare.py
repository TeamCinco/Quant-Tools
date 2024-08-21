import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

def get_financials(ticker, period):
    stock = yf.Ticker(ticker)
    
    if period == 'annual':
        income_stmt = stock.financials.T
        balance_sheet = stock.balance_sheet.T
        cashflow = stock.cashflow.T
    else:  # quarterly
        income_stmt = stock.quarterly_financials.T
        balance_sheet = stock.quarterly_balance_sheet.T
        cashflow = stock.quarterly_cashflow.T

    return income_stmt, balance_sheet, cashflow

def display_metrics(income_stmt, balance_sheet, cashflow):
    metrics = {}
    index = 1

    for df, name in [(income_stmt, 'Income Statement'), 
                     (balance_sheet, 'Balance Sheet'), 
                     (cashflow, 'Cash Flow')]:
        for column in df.columns:
            metrics[index] = (name, column)
            index += 1

    for i in sorted(metrics.keys()):
        print(f"{i}. {metrics[i][0]}: {metrics[i][1]}")

    return metrics

def billions(x, pos):
    'The two args are the value and tick position'
    return '$%1.1fB' % (x * 1e-9)

def millions(x, pos):
    'The two args are the value and tick position'
    return '$%1.1fM' % (x * 1e-6)

def plot_metrics(ticker, metrics, selected_indices, period):
    stock = yf.Ticker(ticker)

    for idx in selected_indices:
        statement, metric = metrics[idx]
        if period == 'annual':
            data = getattr(stock, statement.lower().replace(' ', '')).T[metric].fillna(0).sort_index()
        else:
            if statement == 'Income Statement':
                data = stock.quarterly_financials.T[metric].fillna(0).sort_index()
            elif statement == 'Balance Sheet':
                data = stock.quarterly_balance_sheet.T[metric].fillna(0).sort_index()
            elif statement == 'Cash Flow':
                data = stock.quarterly_cashflow.T[metric].fillna(0).sort_index()

        plt.figure(figsize=(14, 7))
        
        bars = plt.bar(range(len(data)), data.values, align='center')
        
        plt.title(f"{ticker} - {period.capitalize()} {statement}: {metric} Over Time")
        plt.xlabel('Date')
        plt.ylabel('Dollar Value ($)')
        
        plt.axhline(0, color='black', linewidth=0.8)

        plt.gca().set_xticks(range(len(data)))
        plt.gca().set_xticklabels(data.index.strftime('%Y-%m-%d'), rotation=45, ha='right')

        abs_max = max(abs(data.max()), abs(data.min()))
        if abs_max > 1e9:
            formatter = FuncFormatter(billions)
        elif abs_max > 1e6:
            formatter = FuncFormatter(millions)
        else:
            formatter = plt.FuncFormatter(lambda x, p: f'${x:,.0f}')

        plt.gca().yaxis.set_major_formatter(formatter)

        for i, bar in enumerate(bars):
            height = bar.get_height()
            if abs_max > 1e9:
                value_text = f'${height/1e9:.2f}B'
            elif abs_max > 1e6:
                value_text = f'${height/1e6:.2f}M'
            else:
                value_text = f'${height:.2f}'
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     value_text,
                     ha='center', va='bottom', rotation=0)

        plt.legend([metric])
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def main():
    ticker = input("Enter a ticker symbol: ").upper()

    print("\nSelect statement period:")
    print("1. Annual")
    print("2. Quarterly")
    
    while True:
        period_choice = input("Enter the number of your choice (1 or 2): ")
        if period_choice in ['1', '2']:
            period = 'annual' if period_choice == '1' else 'quarterly'
            break
        else:
            print("Invalid input. Please enter 1 or 2.")

    income_stmt, balance_sheet, cashflow = get_financials(ticker, period)

    print(f"\nAvailable {period.capitalize()} Financial Metrics:")
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

    plot_metrics(ticker, metrics, selected_indices, period)

if __name__ == "__main__":
    main()