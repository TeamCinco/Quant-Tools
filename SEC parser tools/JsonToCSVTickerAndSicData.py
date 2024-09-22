import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# Paths for input JSON file and output Excel file
input_file = r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\FILES FOR SCRIPTS\TICKERs\combined_data.json"
output_excel = r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\FILES FOR SCRIPTS\TICKERs\industry_data.xlsx"

# Function to clean sheet names (remove invalid Excel characters)
def clean_sheet_name(sheet_name):
    invalid_chars = '[]:*?/\\'
    for char in invalid_chars:
        sheet_name = sheet_name.replace(char, "")
    return sheet_name[:31]  # Excel sheet name limit is 31 characters

# Initialize a dictionary to hold the data grouped by SICDescription (industry)
industry_data = defaultdict(list)

# Read the combined_data.json file
print("Loading data from combined_data.json...")
with open(input_file, 'r') as f:
    data = json.load(f)

# Group the data by SICDescription
print("Grouping data by industry (SICDescription)...")
for record in tqdm(data.values(), desc="Processing records", unit="record"):
    sic_description = record.get("SICDescription", "N/A")
    industry_data[sic_description].append({
        "CIK": record.get("CIK", "N/A"),
        "CompanyName": record.get("CompanyName", "N/A"),
        "Ticker": record.get("Ticker", "N/A"),
        "SIC": record.get("SIC", "N/A"),
    })

# Create a Pandas Excel writer object
print("Writing data to Excel...")
with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
    # Loop through each industry and write to a separate sheet
    for industry, records in tqdm(industry_data.items(), desc="Writing to sheets", unit="sheet"):
        if industry == "N/A":
            sheet_name = "Unknown Industry"
        else:
            # Clean the sheet name
            sheet_name = clean_sheet_name(industry)
        
        # Convert the list of records for this industry to a DataFrame
        df = pd.DataFrame(records)
        
        # Write the DataFrame to a new sheet in the Excel file
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Data successfully written to {output_excel}")
