import logging
import requests
import os
import sqlite3
import time
from contextlib import contextmanager
import sys
from ratelimit import limits, sleep_and_retry
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pymongo import MongoClient

FMP_API_KEY = os.getenv('FMP_API_KEY')
stock_symbols = ['AAPL', 'TSLA','MSFT','NVDA','UL','GOOGL','PFE','JNJ','PG','DANOY','ADH','ALC','AD8','AHC',
                 'ACL', 'AEF','COH','CSL','RMD','SEK','WEB','XRO']
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='real_time_sri_companies.log', filemode='w')

@contextmanager  # Creating context managers, helping to manage resources such as files, database connections, or network connections cleanly and efficiently.
# Defining the runtime context to be established when the block of code is executed
def get_db_connection():
    conn = None
    try:
        conn = sqlite3.connect('real_time_sri_companies.db')
        conn.row_factory = sqlite3.Row  # Returning rows sqlite3.Row objects
        yield conn  # Temporarily return the object to the caller
    except sqlite3.Error as e:
        logging.error(f"Database connection error: {e}")
        raise
    finally:
        if conn:
            conn.close()


def validate_stock_data(data):
    required_fields = {'symbol', 'price', 'name'}
    numeric_fields = {'price', 'marketCap', 'peRatio', 'volume'}

    if not isinstance(data, dict):
        raise ValueError("Stock data must be a dictionary")

    if not all(field in data for field in required_fields):
        raise ValueError(f"Missing required fields: {required_fields - set(data.keys())}")

    for field in numeric_fields & set(data.keys()):
        if not isinstance(data[field], (int, float)) or data[field] < 0:
            raise ValueError(f"Invalid numeric value for {field}: {data[field]}")

    return True


def make_api_request(session, endpoint, symbol, retries=3, backoff_factor=0.5):
    url = f'https://financialmodelingprep.com/api/v3/{endpoint}/{symbol}'
    headers = {'User-Agent': 'YourAppName/1.0'}

    for attempt in range(retries):
        try:
            response = session.get(
                url,
                params={'apikey': FMP_API_KEY},
                headers=headers,
                timeout=(5, 30)  # (connect timeout, read timeout)
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                logging.error(f"Failed after {retries} attempts: {endpoint}/{symbol}")
                raise
            time.sleep(backoff_factor * (2 ** attempt))

CALLS_PER_SECOND = 5

@sleep_and_retry
@limits(calls=CALLS_PER_SECOND, period=1)
def fetch_batch_data(symbols, batch_size=10):
    """Process symbols in batches with rate limiting"""
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        try:
            sql_data = fetch_real_time_sql_data(batch)
            if sql_data:
                update_database(sql_data)
            time.sleep(1)  # Ensure we don't exceed API limits
        except Exception as e:
            logging.error(f"Batch processing failed: {e}")
def create_sql_database(db_name):
    '''
    Create a database to store stock data.
    '''
    logging.debug('Creating database...')
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    try:
        cursor.execute('''
        CREATE TABLE real_time_sri_companies( 
        name TEXT UNIQUE,
        price REAL,
        symbol TEXT PRIMARY KEY,
        industry TEXT,
        real_time_price REAL,
        priceTarget REAL,
        upgrades_downgrades TEXT,
        stock_news_sentiments TEXT,
        mergers_acquisitions TEXT,
        senate_trading TEXT
        )
        ''')
        # Commit the change
        conn.commit()
        logging.info("Database successfully created.")
    except sqlite3.OperationalError as e:
        logging.error(f"Error creating table: {e}")
    finally:
        cursor.close()
        conn.close()

def create_mongo_database(mongo_db_name):
    '''
    Create a MongoDB database to store unstructured stock data.
    '''
    client = MongoClient('localhost', 27017)
    db = client[mongo_db_name]
    collection = db['stock_data']
    return collection


# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
endpoints = ['profile','real-time-price','price-target-rss-feed','upgrades-downgrades-rss-feed','stock-news-sentiments-rss-feed','mergers-acquisitions-rss-feed','senate-trading-rss-feed']


# Define retry strategy
retry_strategy = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504]
)

# Define rate limiter
rate_limiter = limits(calls=5, period=1)
def process_response(data):
    # Process the data as needed
    # For example, extract relevant fields and return a dictionary
    processed_data = {
        'symbol': data.get('symbol'),
        'price': data.get('price'),
        'name': data.get('companyName'),
        'industry': data.get('industry'),
        'real_time_price': {'bidSize':data.get('bidSize'),
                            'askPrice':data.get('askPrice'),
                            'askSize':data.get('askSize'),
                            'bidPrice':data.get('bidPrice'),
                            'lastSalePrice':data.get('lastSalePrice'),
                            'lastSaleSize':data.get('lastSaleSize'),
                            'lastSaleTime':data.get('lastSaleTime'),
                            'fmpLast':data.get('fmpLast'),
                            'lastUpdated':data.get('lastUpdated')
                            },
        'priceTarget': {'publishedDate':data.get('publishedDate'),
                         'newsURL':data.get('newsURL'),
                         'newsTitle':data.get('newsTitle'),
                         'analystName':data.get('analystName'),
                         'priceTarget':data.get('priceTarget'),
                         'adjustedPriceTarget':data.get('adjPriceTarget'),
                         'priceWhenPosted':data.get('priceWhenPosted'),
                         'newsPublisher':data.get('newsPublisher'),
                         'newsBaseURL':data.get('newsBaseURL'),
                         'analystCompany':data.get('analystCompany')
                         },
        'upgrades_downgrades': {'publishedDate': data.get('publishedDate'),
                                'newsURL': data.get('newsURL'),
                                'newsTitle': data.get('newsTitle'),
                                'newsBaseURL': data.get('newsBaseURL'),
                                'newsPublisher': data.get('newsPublisher'),
                                'newGrade': data.get('newGrade'),
                                'previousGrade': data.get('previousGrade'),
                                'gradingCompany': data.get('gradingCompany'),
                                'action': data.get('action'),
                                'priceWhenPosted': data.get('priceWhenPosted')
                                }
        ,
        'stock_news_sentiments': {'publisedDate':data.get('publishedDate'),
                                    'title':data.get('title'),
                                    'site':data.get('site'),
                                    'text':data.get('text'),
                                    'url':data.get('url'),
                                    'sentiment':data.get('sentiment'),
                                    'sentimentScore':data.get('sentimentScore')
                                  },


        'mergers_acquisitions': {'cik': data.get('cik'),
                                 'targetCompany': data.get('targetCompanyName'),
                                    'targetCompanyCIK': data.get('targetCik'),
                                 'targetedSymbol': data.get('targetedSymbol'),
                                 'transactionDate': data.get('transactionDate'),
                                 'acceptanceTime': data.get('acceptanceTime'),
                                 'url': data.get('url'),

        },
        'senate_trading': {'firstName':data.get('firstName'),
                           'lastName':data.get('lastName'),
                          'office':data.get('office'),
                           'link':data.get('link'),
                           'dateReceived':data.get('dateReceived'),
                           'transactionDate':data.get('transactionDate'),
                           'owner':data.get('owner'),
                           'assetDescription':data.get('assetDescription'),
                           'assetType':data.get('assetType'),
                           'type':data.get('type'),
                           'amount':data.get('amount'),
                           'comment':data.get('comment'),

                           }
    }
    return processed_data
@sleep_and_retry
@limits(calls=CALLS_PER_SECOND, period=1)
def fetch_real_time_sql_data(stock_symbols):
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_strategy)  # Configuring HTTP requests with a retry strategy for retrying requests
    session.mount("https://", adapter)  # Attach the adapter to the session

    for symbol in stock_symbols:
        for endpoint in endpoints:
            try:
                #rate_limiter.acquire()  # Implement proper rate limiting
                response = session.get(f"https://financialmodelingprep.com/api/v3/{endpoint}/{symbol}?apikey={FMP_API_KEY}")
                data = response.json()
                logging.debug(f"Fetched data for {symbol} from {endpoint}:{data}")
                if not isinstance(data,dict):
                    raise ValueError("Stock data must be a dictionary")
                validate_stock_data(data)
                yield {
                    'symbol': data.get('symbol'),
                       'price':data.get('price'),
                       'name':data.get('companyName'),
                       'industry':data.get('industry'),
                       'real_time_price':data.get('lastUpdated'),
                       'priceTarget':data.get('priceTarget'),
                       'upgrades_downgrades':data.get(''),
                       'stock_news_sentiments':data.get('sentimentScore'),
                       'mergers_acquisitions':data.get('targetCompany'),
                       'senate_trading':data.get('office')
                }
            except Exception as e:
                logging.error(f"Failed to fetch {endpoint} for {symbol}: {e}")
                continue

def update_database(stock_data):
    if not stock_data:
        logging.warning("No stock data to update")
        return False

    with get_db_connection() as conn:
        try:
            cursor = conn.cursor()

            # Begin transaction
            conn.execute("BEGIN")

            for data in stock_data:
                try:
                    validate_stock_data(data)
                    columns = data.keys()
                    placeholders = ','.join(['?' for _ in columns])
                    column_names = ','.join(columns)

                    query = f'''
                        INSERT OR REPLACE INTO real_time_sri_companies (symbol,industry,name,real_time_price)
                        VALUES ({placeholders})
                    '''

                    cursor.execute(query, list(data.values()))

                except (ValueError, sqlite3.Error) as e:
                    logging.error(f"Error updating {data.get('symbol', 'unknown')}: {e}")
                    continue

            # Commit transaction
            conn.commit()
            return True

        except Exception as e:
            conn.rollback()
            logging.error(f"Transaction failed: {e}")
            return False

def insert_data_into_mongodb(collection,processed_data):
    '''
    Insert data into a MongoDB database.
    '''
    processed_data = process_response(processed_data)
    if processed_data:
        collection.insert_many(processed_data)
        logging.info("Data successfully inserted into MongoDB.")
        logging.info(f"Inserted {len(processed_data)} records into MongoDB.")

def fetch_data_from_mongodb(collection):
    '''
    Retrieve data from a MongoDB database.
    '''
    return list(collection.find())

def list_tables():
    '''
    Retrieve the names of all tables in the database.
    '''
    conn=sqlite3.connect('real_time_sri_companies.db')
    cursor = conn.cursor()
    try:
        cursor.execute('''
        SELECT name FROM sqlite_master WHERE type='table'
        ''')
        table = cursor.fetchall()
    except sqlite3.OperationalError as e:
        print(f"Error fetching table names: {e}")
    finally:
        cursor.close()
        conn.close()
    return[t[0] for t in table]

def describe_table(table_name: str) -> list[tuple[str,str]]:
    '''
    Look up the table schema.
    Returns:
        List of columns, where each entry is a tuple of (column, type).
    '''
    conn=sqlite3.connect('real_time_sri_companies.db')
    cursor = conn.cursor()
    try:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        #Print column details
        for col in columns:
            print(col)
        return [(col[1],col[2]) for col in columns]
    except sqlite3.OperationalError as e:
        print(f"Error fetching table schema: {e}")
    finally:
        conn.close()


def main():
    try:
        # Ensure database exists
        create_sql_database('real_time_sri_companies.db')
        collection = create_mongo_database('real_time_stock_data')


        # Process stocks in batches
        fetch_batch_data(stock_symbols)

        # Verify data
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM real_time_sri_companies")
            count = cursor.fetchone()[0]
            logging.info(f"Updated {count} stock records")


    except Exception as e:
        logging.error(f"Program failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()