import os
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Paths for input folder and output JSON file
input_folder = r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\FILES FOR SCRIPTS\TICKERs\submissions9.21.24"
output_file = r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\FILES FOR SCRIPTS\TICKERs\tickers.json"

# Initialize a dictionary to hold the parsed data
output_data = {}

# Function to handle missing or null values
def get_value(data, key):
    try:
        return data.get(key, "N/A") if data.get(key) else "N/A"
    except Exception:
        return "N/A"

# Function to process individual files
def process_file(file_path):
    filename = os.path.basename(file_path)
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)  # Load the JSON data

            # Fetch the required fields: CIK, SIC, SICDescription, Company Name, Ticker, and ownerOrg
            cik = get_value(data, "cik")
            sic = get_value(data, "sic")
            sic_description = get_value(data, "sicDescription")
            company_name = get_value(data, "name")
            
            # Ticker may be in an array or missing, handle appropriately
            tickers = data.get("tickers", [])
            ticker = tickers[0] if tickers else "N/A"
            
            # Extract ownerOrg field if it exists
            owner_org = get_value(data, "ownerOrg")

            return filename, {
                "CIK": cik,
                "SIC": sic,
                "SICDescription": sic_description,
                "CompanyName": company_name,
                "Ticker": ticker,
                "OwnerOrg": owner_org
            }
    except json.JSONDecodeError:
        print(f"Error parsing JSON file: {filename}")
    except Exception as e:
        print(f"An error occurred while processing file {filename}: {e}")
    return None

# Get a list of all JSON files in the folder
json_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".json")]

# Function to handle the writing of output in a thread-safe way
def save_output(output_data, output_file):
    try:
        with open(output_file, 'w') as outfile:
            json.dump(output_data, outfile, indent=4)
        print(f"Data successfully written to {output_file}")
    except Exception as e:
        print(f"Error writing to output file: {e}")

# Batch processing function to avoid overwhelming system resources
def process_files_in_batches(files, batch_size=1000):
    for i in range(0, len(files), batch_size):
        yield files[i:i + batch_size]

# Main function for parallel processing
def main():
    num_workers = multiprocessing.cpu_count()  # Use all available CPU cores
    batch_size = 1000  # Process files in batches to avoid overloading resources

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for batch in process_files_in_batches(json_files, batch_size):
            # Use tqdm to show progress and map processing to available cores
            with tqdm(total=len(batch), desc="Processing files", leave=False, unit="file") as pbar:
                futures = {executor.submit(process_file, file_path): file_path for file_path in batch}
                
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        filename, data = result
                        output_data[filename] = data
                    pbar.update(1)

    # Write the final output to a single JSON file
    save_output(output_data, output_file)

if __name__ == "__main__":
    main()
