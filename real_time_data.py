import yfinance as yf
import requests
import os
import sqlite3
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
    stock_data = []
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
                    stock_data.append({'symbol':stock_symbol,'price':current_price})
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
    return stock_data


def add_column():
    '''
    Add a new column to the database.
    '''
    conn = sqlite3.connect('stock_companies.db')
    cursor = conn.cursor()
    try:
        cursor.execute('''
        ALTER TABLE stock_companies ADD COLUMN symbol TEXT''')  # SQL command to add a new column
        cursor.execute('''
        ALTER TABLE stock_companies ADD COLUMN price REAL''')
        conn.commit()
        print("Columns added successfully")
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:  # A block used to define code that will always be executed, whether there is an error or the try block is executed or not.
        conn.close()

def update_database(symbol, price):
    '''
    Update the database with the latest stock prices.
    '''
    conn = sqlite3.connect('stock_companies.db')
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO stock_companies (symbol, price)
            VALUES (?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
            price = excluded.price,
            timestamp = CURRENT_TIMESTAMP
        ''', (symbol, price))
        conn.commit()
        print(f"Updated database successfully")
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()
def list_tables():
    '''
    Retrieve the names of all tables in the database.
    '''
    conn=sqlite3.connect('stock_companies.db')
    cursor = conn.cursor()
    cursor.execute('''
    SELECT name FROM sqlite_master WHERE type='table'
    ''')
    table = cursor.fetchall()
    cursor.close()

    return[t[0] for t in table]

table_names = list_tables()
print(table_names)

def describe_table(table_name: str) -> list[tuple[str,str]]:
    '''
    Look up the table schema.
    Returns:
        List of columns, where each entry is a tuple of (column, type).
    '''
    conn=sqlite3.connect('stock_companies.db')
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    #Print column details
    for col in columns:
        print(col)
    conn.close()

columns = describe_table('stock_companies')
print(columns)

def __main__():
    add_column()
    stock_data = fetch_real_time_data(stock_symbols)
    if stock_data:
        for data in stock_data:
            symbol = data['symbol']
            current_price = data['price']
            print(f"Added stock data to the database")
            update_database(symbol=symbol,price=current_price)
    else:
        print(f"Failed to fetch data")
    list_tables()
    describe_table('stock_companies')
if __name__ == '__main__':
    __main__()