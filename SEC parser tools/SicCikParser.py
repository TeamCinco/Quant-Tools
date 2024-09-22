import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Paths for input folder and output JSON file
input_folder = r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\FILES FOR SCRIPTS\TICKERs\submissions9.21.24"
output_file = r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\FILES FOR SCRIPTS\TICKERs\tickers.json"
tickers_file = r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\FILES FOR SCRIPTS\tickers.json"
# Get the number of CPU cores for optimal multi-threading
max_workers = multiprocessing.cpu_count() * 2  # Using double the number of cores for heavy parallelism


# Load the ticker data from tickers.json
with open(tickers_file, 'r') as f:
    ticker_data = json.load(f)

# Create a dictionary to map CIK to ticker, ensuring CIK is padded to 10 digits
cik_to_ticker = {str(v['cik_str']).zfill(10): v['ticker'] for k, v in ticker_data.items()}

# Initialize a dictionary to hold the parsed data in the specified format
output_data = {}

# Function to handle missing or null values
def get_value(data, key):
    try:
        return data.get(key, "N/A") if data.get(key) else "N/A"
    except Exception:
        return "N/A"

# Function to clean ticker by removing any suffixes like -PN, -PQ, -PM
def clean_ticker(ticker):
    if ticker == "N/A":
        return ticker
    return ticker.split('-')[0]  # Split at '-' and take the base ticker

# Function to process individual files
def process_file(file_path, filename):
    print(f"Processing file: {filename}")
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)  # Load the JSON data
            
            # Fetch the required fields: CIK, SIC, SICDescription, and company name (name)
            cik_str = get_value(data, "cik")
            cik_str_padded = cik_str.zfill(10)  # Pad the CIK with leading zeros to make it 10 digits
            sic = get_value(data, "sic")
            sic_description = get_value(data, "sicDescription")
            company_name = get_value(data, "name")
            ticker = cik_to_ticker.get(cik_str_padded, "N/A")  # Look up the ticker by CIK
            ticker_cleaned = clean_ticker(ticker)  # Clean the ticker to remove unwanted suffixes
            
            return cik_str_padded, {
                "CIK": cik_str_padded,
                "SIC": sic,
                "ticker": ticker_cleaned,
                "SICDescription": sic_description,
                "CompanyName": company_name
            }
    
    except json.JSONDecodeError:
        print(f"Error parsing JSON file: {filename}")
    except Exception as e:
        print(f"An error occurred while processing file {filename}: {e}")
    return None, None

# Get a list of all JSON files
json_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]

# Multi-threaded processing with progress bar
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(process_file, os.path.join(input_folder, filename), filename): filename for filename in json_files}
    with tqdm(total=len(json_files), desc="Processing files", leave=False, unit="file") as pbar:
        idx = 0
        for future in as_completed(futures):
            filename = futures[future]
            pbar.set_description(f"Processing {filename}")
            try:
                cik_str_padded, company_data = future.result()
                if cik_str_padded and company_data:
                    output_data[str(idx)] = company_data  # Index as a string key
                    idx += 1
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
            pbar.update(1)

# Write the final output to a single JSON file
try:
    with open(output_file, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)
    print(f"Data successfully written to {output_file}")
except Exception as e:
    print(f"Error writing to output file: {e}")