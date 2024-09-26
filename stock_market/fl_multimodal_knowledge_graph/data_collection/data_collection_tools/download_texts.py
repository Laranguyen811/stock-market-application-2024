import scrapy
import requests
import yfinance as yf
import alpha_vantage as av
import os
import json
from typing import Dict, Any

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

stock_exchange_list = ['nyse.com','nasdaq.com','jpx.co.jp','sse.com.cn','hkex.com.hk','londonstockexchange.com','euronext.com','tsx.com','asx.com.au','bseindia.com','nseindia.com','deutsche-boerse.com','six-group.com','moex.com','jse.co.za']

for stock_exchange in stock_exchange_list:
    url = 'https://www.' + stock_exchange

