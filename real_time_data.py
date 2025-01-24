import yfinance as yf
import requests
import os
FMP_API_KEY = os.getenv('FMP_API_KEY')
stock_symbols = ['AAPL', 'TSLA','MSFT','NVDA','UL','GOOGL','PFE','JNJ','PG','DANOY','ADH','ALC','AD8','AHC',
                 'ACL', 'AEF','COH','CSL','RMD','SEK','WEB','XRO']

def fetch_real_time_data(stock_symbols):
    '''
    Fetches real-time stock data for a given stock symbol.
    Inputs:
        stock_symbols(list): A list of stock symbols
    Returns:
        DataFrame: A pandas DataFrame of stock data
    '''
    for stock_symbol in stock_symbols:
        try:
            url = f'https://financialmodelingprep.com/api/v3/quote/{stock_symbol}?apikey={FMP_API_KEY}'
            print(url)
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                # Validate if 'data' is not empty and has the expected key-value structure
                if data and isinstance(data,list) and 'price' in data[0]:
                    current_price=data[0]['price']
                    print(f"Current price of {stock_symbol} is ${current_price}")
                else:
                    print(f"Unexpected data structure or no price available for {stock_symbol}")
            else:
                print(f"Failed to fetch data for {stock_symbol}.HTTP status code: {response.status_code}")
                print(f"Response content: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while fetching data for {stock_symbol}: {e}")
        except (KeyError, IndexError) as e:
            print(f"Error extracting data for stock {stock_symbol}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

fetch_real_time_data(stock_symbols)