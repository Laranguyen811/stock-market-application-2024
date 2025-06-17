#Scraping data from yahoo finance

import yfinance as yf
import agate as ag

'''Get the data of the stock MSFT using API yfinance'''
# List of stock tickers
tickers = [
    'MSFT', 'GOOG', 'AMZN', 'AAPL', 'SAP', 'META', '005930.KS', 'INTC',
    'IBM', 'ORCL', 'BABA', 'TCEHY', 'NVDA', 'TSM', 'NFLX', 'TSLA',
    'CRM', 'ADBE', 'PYPL'

]


# Dictionary to store stock data
stock_data = {f"data_{ticker}": yf.download(ticker, start='2013-12-19', end='2023-12-19') for ticker in tickers}
#the data in all these dataframes are stored as a Python object, indicating mixed data types in columns
