import yfinance as yf

def fetch_stock_news(ticker):
    stock = yf.Ticker(ticker)
    news = stock.news

    if news:
        for idx, article in enumerate(news, start=1):
            print(f"{idx}. {article['title']}")
            print(f"   Published on: {article['publisher']} - {article['providerPublishTime']}")
            print(f"   Link: {article['link']}\n")
    else:
        print("No news found for this ticker.")

def main():
    ticker = input("Enter a ticker symbol: ").upper()
    print(f"\nFetching news for {ticker}...\n")
    fetch_stock_news(ticker)

if __name__ == "__main__":
    main()
