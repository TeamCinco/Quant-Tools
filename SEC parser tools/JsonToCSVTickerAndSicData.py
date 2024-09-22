import json
from tqdm import tqdm
from openpyxl import Workbook
import re
# Paths for input and output files
tickers_file = r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\FILES FOR SCRIPTS\TICKERs\tickers.json"
output_excel = r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\FILES FOR SCRIPTS\TICKERs\ticker_CIK_SIC.xlsx"

# Step 1: Load the ticker data from tickers.json
print("Step 1: Loading ticker data...")
try:
    with tqdm(total=1, desc="Loading ticker data", leave=True, unit="task") as pbar:
        with open(tickers_file, 'r', encoding='utf-8') as f:
            ticker_data = json.load(f)
        pbar.update(1)
except Exception as e:
    print(f"Error loading ticker data: {e}")
    exit()

# Step 2: Organize the data by SIC Description
print("Step 2: Organizing data by SIC Description...")
sic_data = {}
try:
    with tqdm(total=len(ticker_data), desc="Organizing data", leave=True, unit="row") as pbar:
        for key, value in ticker_data.items():
            sic_desc = value.get("SICDescription", "Unknown SIC Description")
            row = {
                "Ticker": value.get("ticker", "N/A"),
                "SIC": value.get("SIC", "N/A"),
                "CompanyName": value.get("CompanyName", "N/A")
            }
            if sic_desc not in sic_data:
                sic_data[sic_desc] = []
            sic_data[sic_desc].append(row)
            pbar.update(1)
except Exception as e:
    print(f"Error organizing data: {e}")
    exit()

# Function to sanitize sheet names by removing invalid characters
def sanitize_sheet_name(name):
    # Replace any invalid character with an underscore
    return re.sub(r'[\/:*?"<>|]', '_', name)

# Step 3: Writing the organized data to an Excel file with multiple sheets
print("Step 3: Writing data to Excel sheets by SIC Description...")
try:
    wb = Workbook()
    for idx, (sic_desc, data_rows) in enumerate(tqdm(sic_data.items(), desc="Writing sheets", leave=True, unit="sheet")):
        sanitized_sic_desc = sanitize_sheet_name(sic_desc[:31])  # Sanitize and truncate to 31 characters
        if idx == 0:
            ws = wb.active
            ws.title = sanitized_sic_desc  # Set title of the first (active) sheet
        else:
            ws = wb.create_sheet(title=sanitized_sic_desc)

        # Write headers
        ws.append(["Ticker", "SIC", "CompanyName"])

        # Write data rows
        for row in data_rows:
            ws.append([row["Ticker"], row["SIC"], row["CompanyName"]])

    # Save the Excel file
    wb.save(output_excel)
    print(f"Data successfully written to {output_excel}")
except Exception as e:
    print(f"Error writing to Excel file: {e}")