import os
import re
import requests
from stock_market.fl_multimodal_knowledge_graph.data_collection.data_collection_tools import proxy_tool
import logging
from typing import List,Tuple

def create_search_uri(text,base_uri,query_param):
    ''' Takes text, base_uri and query_params and returns search_uri.
    Inputs:
        text(string):A string of text to be searched
        base_uri(string): A string of base URI to search for visual files
        query_params(string): A string of query parameters
    Returns:
        dictionary: A dictionary of search urls their respective search engines
    '''
    query = re.sub(r'\s+','+',text)
    return base_uri.format(query_param,query)

def download_visuals(text: str,save_path: str,limits: int = 5) -> List[Tuple[bool,str]]:
    '''Takes text and save path to file to create a record of downloading visual files.
    Inputs:
        text(string): A string of text to search for visuals
        save_path(string): A string of path to save.
        limits(int): The maximum number of visual files to download.Default is 5.
    Returns:
        List[Tuple[bool,str]]: A list of tuples indicating whether the visual file has audio and its format.
    '''
    search_engines = {
        'Google': {
            'video': "https://www.google.com/search?q={query}&source=lnms&tbm=vid",
            'vector': "https://www.google.com/search?q={query}&source=lnms&tbm=isch&tbs=itp:lineart",
            '3d_model': "https://www.google.com/search?q={query}&source=lnms&tbm=isch&tbs=itp:photo",
            'psd': "https://www.google.com/search?q={query}&source=lnms&tbm=isch&tbs=itp:photo",
            'raw': "https://www.google.com/search?q={query}&source=lnms&tbm=isch&tbs=itp:photo"
        },
        'Bing': {
            'video': "https://www.bing.com/videos/search?q={query}",
            'vector': "https://www.bing.com/images/search?q={query}&qft=+filterui:photo-lineart",
            '3d_model': "https://www.bing.com/images/search?q={query}&qft=+filterui:photo-photo",
            'psd': "https://www.bing.com/images/search?q={query}&qft=+filterui:photo-photo",
            'raw': "https://www.bing.com/images/search?q={query}&qft=+filterui:photo-photo"
        },
        'DuckDuckGo': {
            'video': "https://duckduckgo.com/?q={query}&iax=videos&ia=videos",
            'vector': "https://duckduckgo.com/?q={query}&iax=images&ia=images&iaf=type:lineart",
            '3d_model': "https://duckduckgo.com/?q={query}&iax=images&ia=images&iaf=type:photo",
            'psd': "https://duckduckgo.com/?q={query}&iax=images&ia=images&iaf=type:photo",
            'raw': "https://duckduckgo.com/?q={query}&iax=images&ia=images&iaf=type:photo"
        },
        'Shutterstock': {
            'vector': "https://www.shutterstock.com/search/{query}?image_type=vector",
            'video': "https://www.shutterstock.com/video/search/{query}"
        },
        'Pixabay': {
            'vector': "https://pixabay.com/vectors/search/{query}/",
            'video': "https://pixabay.com/videos/search/{query}/"
        },
        'Pexels': {
            'video': "https://www.pexels.com/search/videos/{query}/"
        },
        'Adobe Stock': {
            'vector': "https://stock.adobe.com/search?k={query}&filters[content_type:vector]=1",
            'video': "https://stock.adobe.com/search?k={query}&filters[content_type:video]=1"
        }
    }
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML,like Gecko) Chrome/85.0.4183.121 Safari/537.36',
        'Connection':'close'
        } #Sending HTTP headers. 'User-Agent': string allowing the network protocol peers to detect the application type, operating system, software vendor, or software version of the requesting software user agent.
    #'Connection':'close' => a directive telling the server to close the connection after the completion of the response.

    text = input("Please enter a search term: ")
    search_urls =  {engine:create_search_uri(text,url,param) for engine,(url,param) in search_engines.items()} #Creating search urls from the dictionary of search engines
    proxy_ip = proxy_tool.get_proxy()
    proxies = {
        'http': 'http://{}'.format(proxy_ip),
        'https': 'https://{}'.format(proxy_ip)
    }
    #Create a dictionary of proxies
    search_urls = []
    for i,url in enumerate(search_urls.items()):
        try: #Attempting a block of code before throwing an Exception

