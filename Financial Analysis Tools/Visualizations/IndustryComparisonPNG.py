import json
import yfinance as yf
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.ticker as mticker
def load_tickers(file_path):
    # Filter out entries where the ticker is "N/A"
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Convert tickers to uppercase to ensure consistency
    tickers = {details['Ticker'].upper(): details for details in data.values() if details['Ticker'] != "N/A"}
    
    return tickers

def get_industry_tickers(selected_ticker, tickers_data):
    # Ensure the selected ticker is uppercase
    selected_data = tickers_data.get(selected_ticker.upper())
    
    if not selected_data:
        print(f"Ticker {selected_ticker.upper()} not found.")
        return None, None
    
    selected_industry = selected_data.get('SICDescription', None)
    if not selected_industry or selected_industry == "N/A":
        print(f"No industry information found for ticker {selected_ticker.upper()}.")
        return None, None

    industry_tickers = {ticker: data for ticker, data in tickers_data.items() if data['SICDescription'] == selected_industry}
    
    return industry_tickers, selected_industry

def fetch_financial_data(ticker):
    stock = yf.Ticker(ticker)
    
    try:
        balance_sheet = stock.balance_sheet.T
        cash_flow = stock.cashflow.T
        income_statement = stock.financials.T

        # Combine all financial data into one DataFrame
        combined_data = pd.concat([balance_sheet, cash_flow, income_statement], axis=1)

        return ticker, combined_data
    except Exception as e:
        print(f"Error fetching financial data for {ticker}: {e}")
        return ticker, None

def fetch_time_series_data(tickers):
    time_series_data = {}

    # Use ThreadPoolExecutor for concurrent data fetching
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_financial_data, ticker): ticker for ticker in tickers}
        
        # Add a progress bar using tqdm
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching financial data"):
            ticker, financial_data = future.result()
            if financial_data is not None:
                time_series_data[ticker] = financial_data

    return time_series_data

def remove_duplicates_across_companies(time_series_data):
    all_metrics = set()
    for data in time_series_data.values():
        all_metrics.update(data.columns)

    return list(all_metrics)

def user_select_metrics(metrics):
    print("\nAvailable Metrics:")
    for idx, metric in enumerate(metrics, 1):
        print(f"{idx}. {metric}")

    # Accept comma-separated input with or without spaces
    selected_indices = input("\nSelect the metric numbers separated by commas (e.g., 1, 3, 5): ").replace(" ", "").split(",")
    selected_metrics = [metrics[int(idx) - 1] for idx in selected_indices if idx.isdigit() and 1 <= int(idx) <= len(metrics)]

    return selected_metrics

def plot_selected_metrics(time_series_data, selected_metrics, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Create a separate PNG file for each company and metric
    for ticker, data in time_series_data.items():
        for metric in selected_metrics:
            if metric in data.columns:
                plt.figure(figsize=(10, 6))
                
                plt.plot(data.index, data[metric], label=f"{ticker}: {metric}")
                
                # Formatting the y-axis to show dollar values
                plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: '${:,.0f}'.format(x)))                
                plt.title(f"{ticker} - {metric} Over Time")
                plt.xlabel("Date")
                plt.ylabel("Dollar Value ($)")
                plt.legend(loc="best")
                plt.grid(True)

                # Save the plot as PNG for each company and metric
                file_path = os.path.join(folder_path, f"{ticker}_{metric.replace(' ', '_')}.png")
                plt.savefig(file_path)
                print(f"Saved {metric} chart for {ticker} as {file_path}")
                plt.close()

def main():
    # Load the ticker data
    file_path = r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\FILES FOR SCRIPTS\TICKERs\tickers.json"
    tickers_data = load_tickers(file_path)

    # Get user input for the ticker
    selected_ticker = input("Enter the ticker symbol of the company: ").upper()

    # Find tickers from the same industry
    industry_tickers, industry = get_industry_tickers(selected_ticker, tickers_data)
    if not industry_tickers:
        return

    print(f"\nSelected industry: {industry}")
    print(f"Companies found in the same industry: {len(industry_tickers)}")

    # Fetch financial data for all companies in the industry
    financial_data = fetch_time_series_data(industry_tickers)

    # Remove duplicate metrics
    all_metrics = remove_duplicates_across_companies(financial_data)

    # Let the user select metrics
    selected_metrics = user_select_metrics(all_metrics)

    # Plot and save charts for the selected metrics
    folder_path = r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\Results\IndustryComparison"
    plot_selected_metrics(financial_data, selected_metrics, folder_path)

if __name__ == "__main__":
    main()
