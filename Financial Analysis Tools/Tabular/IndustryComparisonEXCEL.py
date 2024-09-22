import json
import yfinance as yf
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import re
from openpyxl.utils import get_column_letter

def load_tickers(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    tickers = {details['Ticker'].upper(): details for details in data.values() if details['Ticker'] != "N/A"}
    return tickers

def get_industry_tickers(selected_ticker, tickers_data):
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
        return ticker, {
            'balance_sheet': balance_sheet,
            'income_statement': income_statement,
            'cash_flow': cash_flow
        }
    except Exception as e:
        print(f"Error fetching financial data for {ticker}: {e}")
        return ticker, None

def fetch_time_series_data(tickers):
    time_series_data = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_financial_data, ticker): ticker for ticker in tickers}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching financial data"):
            ticker, financial_data = future.result()
            if financial_data is not None:
                time_series_data[ticker] = financial_data
    return time_series_data

def clean_column_name(column_name):
    # Remove any characters that are not letters, numbers, or underscores
    cleaned_name = re.sub(r'[^\w\s]', '', str(column_name))
    # Replace spaces with underscores
    cleaned_name = cleaned_name.replace(' ', '_')
    # Ensure the column name starts with a letter
    if not cleaned_name[0].isalpha():
        cleaned_name = 'col_' + cleaned_name
    return cleaned_name

def create_excel_file(time_series_data, file_path):
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        for statement in ['balance_sheet', 'income_statement', 'cash_flow']:
            all_data = []
            for ticker, data in time_series_data.items():
                df = data[statement]
                df['Ticker'] = ticker
                # Clean column names
                df.columns = [clean_column_name(col) for col in df.columns]
                all_data.append(df)
            
            combined_df = pd.concat(all_data, axis=0)
            combined_df = combined_df.reset_index()
            combined_df = combined_df.rename(columns={'index': 'Date'})
            
            # Reorder columns to have Ticker and Date first
            columns = ['Ticker', 'Date'] + [col for col in combined_df.columns if col not in ['Ticker', 'Date']]
            combined_df = combined_df[columns]
            
            # Write to Excel
            sheet_name = statement.replace('_', ' ').title()
            combined_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Auto-adjust columns' width
            for col_idx, column in enumerate(combined_df.columns, 1):
                column_length = max(combined_df[column].astype(str).map(len).max(), len(column))
                writer.sheets[sheet_name].column_dimensions[get_column_letter(col_idx)].width = column_length + 2

def main():
    file_path = r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\FILES FOR SCRIPTS\TICKERs\tickers.json"
    tickers_data = load_tickers(file_path)

    selected_ticker = input("Enter the ticker symbol of the company: ").upper()
    industry_tickers, industry = get_industry_tickers(selected_ticker, tickers_data)
    if not industry_tickers:
        return

    print(f"\nSelected industry: {industry}")
    print(f"Companies found in the same industry: {len(industry_tickers)}")

    financial_data = fetch_time_series_data(industry_tickers)

    # Create Excel file
    excel_file_path = r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\Results\IndustryComparison\industry_comparison.xlsx"
    create_excel_file(financial_data, excel_file_path)
    print(f"\nExcel file created: {excel_file_path}")

if __name__ == "__main__":
    main()
