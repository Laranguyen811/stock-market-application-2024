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

def download_visual(text: str,save_path: str,limits: int = 5) -> List[Tuple[bool,str]]:
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
    }  # Create a dictionary of proxies
    visual_urls = []
    file_extensions= {
        '.bmp', '.tiff', '.tif', '.webp', '.svg', '.ico', '.heic',
        '.heif', '.raw', '.cr2', '.nef', '.orf', '.sr2',
        '.eps', '.ai', '.pdf', '.dxf', '.mp4', '.avi', '.mov', '.wmv', '.mkv',
        '.flv', '.webm', '.mpeg', '.mpg', '.3gp', '.avchd', '.vob'
    }  # A set of file extensions
    for engine, urls in search_urls.items():
        for i,url in enumerate(visual_urls):
            try: # Attempting a block of code before throwing an Exception
                response = requests.get(url,headers=headers,proxies=proxies,timeout=10)  # Sending a GET request to the specified URL. The headers parameter is used to pass HTTP headers to the request, and the proxies parameter is used to specify the proxies that the request should go through. The timeout parameter specifies the maximum number of seconds to wait for a response before giving up.
                response.raise_for_status()  # Checking if the request was successful. If not, will raise a HTTPError exception.
                visual_urls += re.findall(r'"(https://[^"]*?\.(?:bmp|tiff|tif|webp|svg|ico|heic|heif|raw|cr2|nef|orf|sr2|eps|ai|pdf|dxf|mp4|avi|mov|wmv|mkv|flv|webm|mpeg|mpg|3gp|avchd|vob))"', response.text)   #Extracting image URLs from the response text
            except requests.exceptions.RequestException as e:  # Throwing a RequestException exception as a variable e
                print(f"Error during request to {url}: {str(e)}")
                continue #Used in a loop, the current iteration ends immediately
    record = []
    count = 0
    for i, url in enumerate(visual_urls):  # Looping over visual_urls
        if count >= limits:
            break  # Ceasing the loop
        for file_extension in file_extensions:
            filename = os.path.join(save_path,str(count) + file_extension)  #Joining save_path, count with a file extension to form a filename
            try:
                r = requests.get(url,headers=headers,proxies=proxies,stream=True,timeout=4) #Sending a GET request to a specified URL. The headers parameter is used to pass HTTP headers to the request, and the proxies parameter is used to specify the proxies that the request should go through. The stream parameter, when set to True, means that the response content should be streamed rather than immediately downloaded. The timeout parameter specifies the maximum number of seconds to wait for a response before giving up.
                r.raise_for_status()  #Checking if the request is successful. If not, will raise a HTTPError exception.
                with open(filename, 'wb') as f:  # Opening a filename in a writing binary mode  and closing when done.
                    f.write(r.content)
                count += 1
                record.append((file_extension[1:]))
            except requests.exceptions.RequestException as e:  # Raising a RequestException exception.
                print(f"Error during request to {url} : {str(e)}")
                continue  #Used in a loop, the current iteration ends immediately.

        return record
# Input prompt
text = input("Please enter a search term: ")




