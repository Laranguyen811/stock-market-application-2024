#Scraping data from yahoo finance

import yfinance as yf
import agate as ag

'''Get the data of the stock MSFT using API yfinance'''
data_msft = yf.download('MSFT', start='2013-12-19', end='2023-12-19')
data_goog = yf.download('GOOG',start='2013-12-19', end='2023-12-19')
data_amzn = yf.download('AMZN',start='2013-12-19', end='2023-12-19')
data_aapl = yf.download('AAPL', start='2013-12-20', end='2023-12-19')
data_sap = yf.download('SAP', start='2013-12-19', end='2023-12-19')
data_meta = yf.download('META', start='2013-12-19', end='2023-12-19')
data_005930_ks =yf.download('005930.KS', start='2013-12-19', end='2023-12-19')
data_intc = yf.download('INTC', start='2013-12-19', end='2023-12-19')
data_ibm = yf.download('IBM', start='2013-12-19', end='2023-12-19')
data_orcl = yf.download('ORCL', start='2013-12-19', end='2023-12-19')
data_baba = yf.download('BABA', start='2013-12-19', end='2023-12-19')
data_tcehy = yf.download('TCEHY', start='2013-12-19', end='2023-12-19')
data_nvda = yf.download('NVDA', start='2013-12-19', end='2023-12-19')
data_tsm = yf.download('TSM', start='2013-12-19', end='2023-12-19')
data_nflx = yf.download('NFLX', start='2013-12-19', end='2023-12-19')
data_tsla = yf.download('TSLA', start='2013-12-19', end='2023-12-19')
data_crm = yf.download('CRM', start='2013-12-19', end='2023-12-19')
data_adbe = yf.download('ADBE', start='2013-12-19', end='2023-12-19')
data_pypl = yf.download('PYPL', start='2013-12-19', end='2023-12-19')
#the data in all these dataframes are stored as a Python object, indicating mixed data types in columns
