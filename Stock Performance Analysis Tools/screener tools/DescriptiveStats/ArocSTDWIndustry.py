import json
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import os
import sys
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Color
from openpyxl.styles.differential import DifferentialStyle
from openpyxl.formatting.rule import Rule, ColorScale, FormatObject

# Set default encoding to UTF-8 for console output
sys.stdout.reconfigure(encoding='utf-8')

def load_tickers(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def filter_by_market_cap(ticker, size):
    stock = yf.Ticker(ticker)
    info = stock.info
    market_cap = info.get('marketCap', None)

    if market_cap is None:
        print(f"Market cap data not available for {ticker}.")
        return False

    if size == 'small' and market_cap <= 2e9:
        return True
    elif size == 'medium' and 2e9 < market_cap <= 10e9:
        return True
    elif size == 'large' and market_cap > 10e9:
        return True
    else:
        return False

def filter_by_industry(data, industry):
    return {k: v for k, v in data.items() if v.get('SICDescription', '').lower() == industry.lower()}

def get_unique_industries(data):
    industries = {v['SICDescription'] for v in data.values()}
    return sorted(industries)

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period='6mo')
    return hist

def calculate_aroc(hist):
    daily_returns = (hist['Open'] - hist['Close']) / hist['Open'] * 100
    aroc = daily_returns.mean()
    return aroc

def log_transform_aroc(aroc):
    return np.log(np.abs(aroc) + 1) * np.sign(aroc)

def process_ticker(ticker, cik, sic, sic_description, company_name, size_filter):
    try:
        if not filter_by_market_cap(ticker, size_filter):
            print(f"{ticker} does not match the market cap size '{size_filter}'; skipping.")
            return None

        hist = get_stock_data(ticker)
        if hist is None:
            return None

        aroc = calculate_aroc(hist)
        if not np.isfinite(aroc):
            print(f"Invalid AROC calculated for {ticker}; skipping.")
            return None

        log_aroc = log_transform_aroc(aroc)

        start_price = hist['Open'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        percent_change = ((end_price - start_price) / start_price) * 100

        stock_data = {
            'CIK': cik,
            'SIC': sic,
            'ticker': ticker,
            'SICDescription': sic_description,
            'CompanyName': company_name,
            'Date': hist.index[-1].strftime('%Y-%m-%d'),
            'AROC': aroc,
            'Log AROC': log_aroc,
            'Start Price': start_price,
            'End Price': end_price,
            'Percent Change': percent_change
        }

        print(f"Processed {ticker} (CIK: {cik})")
        return stock_data, aroc, log_aroc, hist

    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None

def main():
    # Load the tickers from the provided file
    tickers_data = load_tickers(r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\FILES FOR SCRIPTS\TICKERs\tickers.json")
    
    # Show list of SIC descriptions for user selection
    unique_industries = get_unique_industries(tickers_data)
    print("Available industries (SIC Descriptions):")
    for i, industry in enumerate(unique_industries, 1):
        print(f"{i}. {industry}")

    # User inputs
    size_filter = input("Enter market cap size to filter by (small, medium, large): ").lower()
    industry_choice = int(input(f"Select an industry by number (1-{len(unique_industries)}): "))
    num_companies = int(input("Enter the number of companies to analyze: "))

    # Get selected industry
    selected_industry = unique_industries[industry_choice - 1]
    print(f"Selected Industry: {selected_industry}")

    # Filter the companies by industry
    filtered_tickers = filter_by_industry(tickers_data, selected_industry)

    if len(filtered_tickers) == 0:
        print(f"No companies found in the '{selected_industry}' industry.")
        return

    # Randomly sample the required number of companies
    selected_tickers = random.sample(list(filtered_tickers.items()), min(num_companies, len(filtered_tickers)))

    stock_data_list = []
    aroc_list = []
    log_aroc_list = []

    # Process the selected tickers
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(
                process_ticker, 
                v['ticker'], 
                v['cik_str'], 
                v['SIC'], 
                v['SICDescription'], 
                v['title'], 
                size_filter
            ): k 
            for k, v in selected_tickers
        }

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                stock_data, aroc, log_aroc, _ = result
                stock_data_list.append(stock_data)
                aroc_list.append(aroc)
                log_aroc_list.append(log_aroc)

    if len(aroc_list) == 0:
        print("No valid AROC data available.")
        return

    # Use estimated population mean and variance
    mean_aroc = np.mean(aroc_list)
    population_variance = np.var(aroc_list, ddof=0)  # ddof=0 for population variance
    std_deviation = np.sqrt(population_variance)

    mean_log_aroc = np.mean(log_aroc_list)
    log_population_variance = np.var(log_aroc_list, ddof=0)
    std_deviation_log = np.sqrt(log_population_variance)

    # Output the processed data
    print(stock_data_list)  # Placeholder to show the processed data

    # Proceed with Excel saving and plotting functions (as in the original script)...

if __name__ == "__main__":
    main()
