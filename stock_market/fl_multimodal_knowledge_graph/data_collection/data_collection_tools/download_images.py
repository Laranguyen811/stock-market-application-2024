import os
import re
import requests
from stock_market.fl_multimodal_knowledge_graph.data_collection.data_collection_tools import proxy_tool
import time
import logging

def create_search_uri(text,base_url,query_param):
    '''Takes text, base_url and query_param to return search_url.
    Inputs:
        text(string): A string of text to be searched
        base_url(string): A string of base URL to search for images
        query_param(string): A string of query parameters
    Returns:
        dictionary: A dictionary of search urls their respective search engines
    '''
    query = re.sub(r'\s+','+',text)
    return base_url.format(query_param,query)

def download_images(text,save_path,limits=5):
    '''
    Takes text and returns a record of downloaded images.
    Inputs:
        text(string): A string of text to be used in an image search
        save_path(string): A string of path to save the downloaded images
    Returns:
        list: a list of records of downloaded images
    '''
    search_engines = {'google':("https://www.google.com/search?q={}&source=lnms&tbm=isch","q"),
                      'bing':("https://www.bing.com/images/search?{}={}", "q"),
                      'duckgogo':("https://duckduckgo.com/?{}={}&iax=images&ia=images", "q"),
                      'yahoo':("https://images.search.yahoo.com/search/images;?{}={}", "p"),
                      'pixabay':("https://pixabay.com/images/search/{}", ""),
                      'pexels':("https://pixabay.com/images/search/{}", ""),
                      'shuttlestock':("https://www.pexels.com/search/{}", ""),
                      'creativecommons':("https://search.creativecommons.org/search?q={}", "q"),
                      } #A dictionary of search engines and their corresponding URLs
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML,like Gecko) Chrome/85.0.4183.121 Safari/537.36',
        'Connection':'close'
    } #Sending HTTP headers. 'User-Agent': string allowing the network protocol peers to detect the application type, operating system, software vendor, or software version of the requesting software user agent.
    #'Connection':'close' => a directive telling the server to close the connection after the completion of the response.
    text = input("Please enter a search term: ")
    search_urls = {engine:create_search_uri(text,url,param) for engine,(url,param) in search_engines.items()} #Creating search urls from the dictionary of search engines
    proxy_ip = proxy_tool.get_proxy() #Obtaining proxy IP
    proxies = {
        'http': 'http://{}'.format(proxy_ip),
        'https': 'https://{}'.format(proxy_ip)
    } #Creating a dictionary of proxies
    image_urls = []
    for i,url in enumerate(search_urls.items()):
        try: #Attempting a block of code before throwing an Exception
            response = requests.get(url,headers=headers,proxies=proxies,timeout=10)#Sending a GET request to the specified URL. The headers parameter is used to pass HTTP headers to the request, and the proxies parameter is used to specify the proxies that the request should go through. The timeout parameter specifies the maximum number of seconds to wait for a response before giving up.
            response.raise_for_status()#Checking if the request was successful. If not, will raise a HTTPError exception.
            image_urls += re.findall(r'\["https://[^,]*?\.jpg",',response.text) #Extracting image URLs from the response text
        except requests.exceptions.RequestException as e: #Throwing a RequestException exception as a variable e
            print(f"Error during requests to {url} :{str(e)}")
            continue #Used in a loop, the current iteration ends immediately
    record = []
    count = 0
    for i,url in enumerate(image_urls):#Looping over image_urls
        if cnt >= limits:
            break #Ceasing the loop
        filename = os.path.join(save_path,str(cnt) + ".jpg") #Joining save_path, count with ".jpg" to form a filename
        try:
            r=requests.get(url,headers=headers,proxies=proxies,stream=True,timeout=45) #Sending a GET request to a specified URL. The headers parameter is used to pass HTTP headers to the request, and the proxies parameter is used to specify the proxies that the request should go through. The stream parameter, when set to True, means that the response content should be streamed rather than immediately downloaded. The timeout parameter specifies the maximum number of seconds to wait for a response before giving up.
            r.raise_for_status() #Checking if the request is successful. If not, will raise a HTTPError exception.
            with open(filename,'wb') as f: #Opening a filename in a writing binary mode  and closing when done
                f.write(r.content)
            count += 1
        except requests.exceptions.RequestException as e: #Raising a RequestException exception
            print(f"Error during requests to {url} :{str(e)}")
            continue #Used in a loop, the current iteration ends immediately
        record.append('jpg')
    return record






