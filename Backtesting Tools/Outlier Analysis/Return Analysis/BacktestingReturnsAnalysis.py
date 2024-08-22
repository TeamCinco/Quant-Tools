import pandas as pd
from bs4 import BeautifulSoup

def extract_returns_from_html(html_path, csv_output_path):
    # Load the HTML file
    with open(html_path, 'r') as file:
        soup = BeautifulSoup(file, 'html.parser')
    
    # Find the table that contains the equity returns
    table = soup.find('table', {'class': 'stats'})  # Adjust class or table identifier based on your HTML
    
    # Extract headers
    headers = [header.text.strip() for header in table.find_all('th')]
    
    # Extract rows
    rows = []
    for row in table.find_all('tr')[1:]:  # Skip the header row
        columns = row.find_all('td')
        rows.append([col.text.strip() for col in columns])
    
    # Create a DataFrame
    df = pd.DataFrame(rows, columns=headers)
    
    # Save to CSV
    df.to_csv(csv_output_path, index=False)

# Example usage
html_path = 'path/to/your/backtesting_results.html'
csv_output_path = 'path/to/save/returns.csv'
extract_returns_from_html(html_path, csv_output_path)
