import os
import json
from tqdm import tqdm

# Paths for input folder and output JSON file
input_folder = r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\FILES FOR SCRIPTS\TICKERs\submissions9.21.24"
output_file = r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\FILES FOR SCRIPTS\TICKERs\combined_data.json"

# Initialize a dictionary to hold the parsed data
output_data = {}

# Function to handle missing or null values
def get_value(data, key):
    try:
        return data.get(key, "N/A") if data.get(key) else "N/A"
    except Exception:
        return "N/A"

# Function to process individual files
def process_file(file_path, filename):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)  # Load the JSON data

            # Fetch the required fields: CIK, SIC, SICDescription, Company Name, and Ticker
            cik = get_value(data, "cik")
            sic = get_value(data, "sic")
            sic_description = get_value(data, "sicDescription")
            company_name = get_value(data, "name")
            
            # Ticker may be in an array or missing, handle appropriately
            tickers = data.get("tickers", [])
            ticker = tickers[0] if tickers else "N/A"

            return {
                "CIK": cik,
                "SIC": sic,
                "SICDescription": sic_description,
                "CompanyName": company_name,
                "Ticker": ticker
            }
    except json.JSONDecodeError:
        print(f"Error parsing JSON file: {filename}")
    except Exception as e:
        print(f"An error occurred while processing file {filename}: {e}")
    return None

# Get a list of all JSON files in the folder
json_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]

# Process all files and collect the data
with tqdm(total=len(json_files), desc="Processing files", leave=False, unit="file") as pbar:
    for idx, filename in enumerate(json_files):
        file_path = os.path.join(input_folder, filename)
        result = process_file(file_path, filename)
        if result:
            output_data[str(idx)] = result  # Use index as key for the final JSON
        pbar.update(1)

# Write the final output to a single JSON file
try:
    with open(output_file, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)
    print(f"Data successfully written to {output_file}")
except Exception as e:
    print(f"Error writing to output file: {e}")
