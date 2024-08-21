import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)['Close']
    return stock_data

def get_fred_data(series_ids, start_date, end_date):
    data = pdr.get_data_fred(series_ids, start=start_date, end=end_date)
    return data

def prepare_data(stock_data, macro_data):
    combined_data = pd.concat([stock_data, macro_data], axis=1).dropna()
    combined_data = combined_data.pct_change().dropna()
    
    # Normalize the data
    scaler = StandardScaler()
    normalized_data = pd.DataFrame(scaler.fit_transform(combined_data), 
                                   columns=combined_data.columns, 
                                   index=combined_data.index)
    
    return normalized_data

def perform_pca(data):
    pca = PCA()
    pca_result = pca.fit_transform(data)
    
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    return pca, pca_result, explained_variance_ratio, cumulative_variance_ratio

def plot_pca_results(explained_variance_ratio, cumulative_variance_ratio):
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'r-')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA: Explained Variance by Components')
    plt.show()

def plot_pca_3d(pca_result, stock_returns):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=stock_returns, cmap='viridis')
    
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.set_zlabel('Third Principal Component')
    ax.set_title('PCA 3D Visualization')
    
    plt.colorbar(scatter, label='Stock Returns')
    plt.show()

def perform_multivariable_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    return model, r2, mse, X_test, y_test, y_pred

def plot_coefficient_importance(model, feature_names):
    coef_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': abs(model.coef_)
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=coef_importance)
    plt.title('Feature Importance in Multivariable Regression')
    plt.show()

def plot_regression_results(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Stock Returns')
    plt.ylabel('Predicted Stock Returns')
    plt.title('Multi-Linear Regression: Actual vs Predicted')
    plt.show()

def main():
    # User inputs
    stock_ticker = input("Enter the stock ticker symbol: ")
    macro_indicators = input("Enter FRED macroeconomic indicator tickers (comma-separated): ").split(',')
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")

    # Fetch data
    stock_data = get_stock_data(stock_ticker, start_date, end_date)
    macro_data = get_fred_data(macro_indicators, start_date, end_date)

    # Prepare and normalize data
    normalized_data = prepare_data(stock_data, macro_data)

    # Perform PCA
    pca, pca_result, explained_variance_ratio, cumulative_variance_ratio = perform_pca(normalized_data)
    plot_pca_results(explained_variance_ratio, cumulative_variance_ratio)
    plot_pca_3d(pca_result, normalized_data.iloc[:, 0])

    # Perform multivariable regression
    X = normalized_data.iloc[:, 1:]  # Macro indicators
    y = normalized_data.iloc[:, 0]   # Stock returns
    model, r2, mse, X_test, y_test, y_pred = perform_multivariable_regression(X, y)

    print(f"R-squared: {r2}")
    print(f"Mean Squared Error: {mse}")

    plot_coefficient_importance(model, X.columns)
    plot_regression_results(y_test, y_pred)

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(normalized_data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap: Stock vs Macro Indicators')
    plt.show()

if __name__ == "__main__":
    main()