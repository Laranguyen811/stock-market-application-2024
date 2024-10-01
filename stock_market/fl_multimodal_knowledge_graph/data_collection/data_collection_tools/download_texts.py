import scrapy
import requests
import yfinance as yf
import alpha_vantage as av
import os
import json
from typing import Dict, Any
from bs4 import BeautifulSoup
import sqlite3

twelve_data_api_url = 'https://api.twelvedata.com'
polygon_io_api_url = 'https://api.polygon.io/v2'

# Setting environment variables
bearer_token = os.environ.get('BEARER_TOKEN')
twitter_search_url = 'https://api.twitter.com/2/tweets/search/all'


# Optional params: start_time,end_time,since_id,until_id,max_results,next_token,expansions,tweet.fields,media.fields,poll.fields,place.fields,user.fields

query_params = {'query': '(from:company - is: retweet) OR #company','tweet.fields' : 'author_id'}

def bearer_oauth(r):
    '''
    Method required by bearer token authentication.
    '''
    r.headers['Authorization'] = f'Bearer {bearer_token}'
    r.headers["User-Agent"] = "v2FullArchiveSearchPython"
    return r

def connect_to_endpoint(url,params):
    '''
    Method to connect to endpoint
    '''
    response = requests.request("GET", url, auth=bearer_oauth, params=params)
    print(f"Request returned status code: {response.status_code}")
    if response.status_code != 200: # If the request is unsuccessful
        raise Exception(f"Request failed: {response.status_code}, {response.text}")
    return response.json()

def main():
    try:
        json_response = connect_to_endpoint(twitter_search_url,query_params)

        print(json.dumps(json_response, indent=4, sort_keys=True))
    except Exception as e:  # Assigning Exception to variable e
        print(f"An error occurred: {e}")

    if __name__ == "__main__":
        main()

stock_exchange_list = ["nyse.com/listings_directory/stock","nasdaq.com/market-activity/stocks/screener","jpx.co.jp/listing/stocks/new/index.html","sse.com.cn/assortment/stock/list/share/","hkex.com.hk/search/titlesearch.xhtml","londonstockexchange.com/live-markets/market-data-dashboard/price-explorer","euronext.com/en/products/equities/list","tsx.com/en/listings/listing-with-us/listed-company-directory","asx.com.au/markets/trade-our-cash-market/directory","bseindia.com/corporates/List_Scrips.html","nseindia.com/invest/company-listing-directory-resources-for-investors","deutsche-boerse-cash-market.com/dbcm-en/instruments-statistics/statistics/listes-companies","six-group.com/en/market-data/shares/companies.html","moex.com/ru/listing/securities-list.aspx","clientportal.jse.co.za/companies-and-financial-instruments"]

for stock_exchange in stock_exchange_list:
    url = "https://www." + stock_exchange
    # Sending a GET request to the website
    response = requests.get(url)
    response.encoding = 'utf-8'  # Unicode Transformation Format - 8-bit, encoding standard use

    # Parsing the HTML content
    soup = BeautifulSoup(response.content,'html.parser')
    companies = []

    # Extracting company names
    for row in soup.find_all('tr'):  # Iterating row in all occurrences of a specified HTML tag (table row, in this case) within a parsed HTML document
        cols = row.find_all('td')  # Assigning columns to all occurrences of a specified HTML tag (table data)
        if cols: # If in cols
            company_name = cols[0].text.strip()  # Stripping the first index and assigning it to a company name
            companies.append(company_name)  # Appending a company name to the companies list

    # Turning the list of companies' names into a fully persistent data structure by transforming it into a database
    # Connecting ro SQLite database
    conn = sqlite3.connect('companies.db')
    cursor = conn.cursor()  # Creating a cursor object to traverse a row and getting a result set one at a time

    # Creating a table to store company names
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS companies (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL
    )
    ''')

    # Inserting company names into the table
    for company in companies:
        cursor.execute('INSERT INTO companies (name) VALUES (?)',(company,))

    # Committing the transaction and closing the connecting
    conn.commit()
    conn.close()

    print("Company names have been stored in the SQLite database.")
    # Finding all company names (adjust

