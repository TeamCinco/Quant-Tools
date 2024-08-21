import pandas as pd
import pandas_datareader as pdr
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from datetime import datetime, timedelta

def get_user_input(prompt):
    return input(prompt).strip()

def get_fred_data(tickers, start_date, end_date):
    data = pd.DataFrame()
    for ticker in tickers:
        try:
            series = pdr.get_data_fred(ticker, start=start_date, end=end_date)
            data[ticker] = series
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    return data

def get_yf_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    if len(tickers) == 1:
        data = data.to_frame()
        data.columns = tickers
    return data

# Set the date range for data fetching
end_date = datetime.now()
start_date = end_date - timedelta(days=365*5)  # 5 years of data

# Get user input for FRED tickers
fred_tickers = []
while True:
    ticker = get_user_input("Enter a FRED ticker (or press Enter to finish): ")
    if ticker == "":
        break
    fred_tickers.append(ticker)

# Get user input for Yahoo Finance tickers
yf_tickers = []
while True:
    ticker = get_user_input("Enter a Yahoo Finance ticker (or press Enter to finish): ")
    if ticker == "":
        break
    yf_tickers.append(ticker)

# Fetch data
fred_data = get_fred_data(fred_tickers, start_date, end_date)
yf_data = get_yf_data(yf_tickers, start_date, end_date)

# Combine the data
df = pd.concat([fred_data, yf_data], axis=1)
df = df.resample('M').last()  # Resample to monthly data
df = df.dropna()  # Remove any rows with missing data

# Display available columns
print("\nAvailable columns:")
for i, col in enumerate(df.columns):
    print(f"{i}: {col}")

# Function to get user input for column selection
def get_column_input(prompt, max_value):
    while True:
        try:
            value = int(input(prompt))
            if 0 <= value < max_value:
                return value
            else:
                print(f"Please enter a number between 0 and {max_value-1}")
        except ValueError:
            print("Please enter a valid number")

# Get user input for simple regression
print("\nSimple Regression:")
x_index = get_column_input("Enter the index of the independent variable: ", len(df.columns))
y_index = get_column_input("Enter the index of the dependent variable: ", len(df.columns))

x_col = df.columns[x_index]
y_col = df.columns[y_index]

# Perform simple regression
X = df[[x_col]]
y = df[y_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Plotting simple regression
plt.figure(figsize=(12, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel(x_col)
plt.ylabel(y_col)
plt.title(f'Simple Regression: {y_col} vs {x_col}')
plt.legend()
plt.show()

# Linear graph for simple regression
plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, model.predict(X), color='red', label='Linear Regression')
plt.xlabel(x_col)
plt.ylabel(y_col)
plt.title(f'Linear Regression: {y_col} vs {x_col}')
plt.legend()
plt.show()

print(f"\nSimple Regression Results:")
print(f"R-squared: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Coefficient: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# Get user input for multivariate regression
print("\nMultivariate Regression:")
num_vars = int(input("Enter the number of independent variables: "))
x_indices = [get_column_input(f"Enter the index of independent variable {i+1}: ", len(df.columns)) for i in range(num_vars)]
y_index = get_column_input("Enter the index of the dependent variable: ", len(df.columns))

x_cols = [df.columns[i] for i in x_indices]
y_col = df.columns[y_index]

# Perform multivariate regression
X = df[x_cols]
y = df[y_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Plotting multivariate regression
fig, axes = plt.subplots(1, len(x_cols), figsize=(6*len(x_cols), 5))
for i, col in enumerate(x_cols):
    axes[i].scatter(X_test[col], y_test, color='blue', label='Actual')
    axes[i].scatter(X_test[col], y_pred, color='red', label='Predicted')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel(y_col)
    axes[i].legend()
plt.tight_layout()
plt.show()

# Linear graphs for multivariate regression
for col in x_cols:
    plt.figure(figsize=(12, 6))
    plt.scatter(X[col], y, color='blue', label='Data')
    plt.plot(X[col], model.predict(X), color='red', label='Linear Regression')
    plt.xlabel(col)
    plt.ylabel(y_col)
    plt.title(f'Multivariate Regression: {y_col} vs {col}')
    plt.legend()
    plt.show()

print(f"\nMultivariate Regression Results:")
print(f"R-squared: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
for i, col in enumerate(x_cols):
    print(f"Coefficient ({col}): {model.coef_[i]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# Display histograms and box plots for each variable
for col in df.columns:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    sns.histplot(df[col], ax=ax1, kde=True)
    ax1.set_title(f'Histogram of {col}')
    
    # Box plot
    sns.boxplot(x=df[col], ax=ax2)
    ax2.set_title(f'Box Plot of {col}')
    
    plt.tight_layout()
    plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()