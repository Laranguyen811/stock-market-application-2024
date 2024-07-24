import os
import re #A built-in module to work with regular expressions (sequences of characters forming a search pattern
import json
import requests
import subprocess
from time import sleep
from stock_market.fl_multimodal_knowledge_graph.data_collection.data_collection_tools import proxy_tool
import logging
from numpy import random
def download(uri,max_tries=10,timeout=5):
    '''Takes an uri and returns an exception if a request is unsuccessful and headers and proxies in the form of JSON format if a request is unsuccessful. After 10 attempts the request is not
    successful, it raises a RequestException.
    Inputs:
    uri(string): A string of uri
    Returns:
    string: A string of headers and proxies if the request is successful.
    exception(exception): An exception thrown when the request is unsuccessful.
    RequestException(exception): An exception thrown after 10 times the request is unsuccessful.
    '''
    #Download the page specified by the given uri and return in dictionary format
    base_delay = 1 #Setting the base delay for the next request to initially 1 second
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64;)AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36 ',
               'Connection':'close',
               }
    proxy_ip = proxy_tool.get_proxy()
    proxies = {'http': f'http//{proxy_ip}', 'https': f'https://{proxy_ip}'}
    for i in range(max_tries):
        try:
            response = requests.get(f'http://api.conceptnet.io{uri}',headers=headers,proxies=proxies,timeout=timeout) #A block of code we should try if there is no error
            if response.status_code == 200: # A status code of the request being processed successfully
                return response.json() #Returning the response in the JSON format
            else:
                raise requests.exceptions.RequestException
        except requests.exceptions.RequestException as e: #RequestException is an inherent of the built-in IOError exception
            delay =base_delay * 2 ** i #exponential backoff, an error-handling strategy to increase delayed time exponentially to prevent a group of clients from retrying
            jitter = random.uniform(0,delay) #Adding a jitter to randomise exponential backoff to prevent a group of clients starting at the same time from retrying repeatedly, startting from 0
            logging.error(f'Request failed with error {e}.Retrying...') #Creating a log of errors with feedback back to a sender
            sleep(delay+jitter) #Delaying the next request for the amount of time of delay and jitter
    raise Exception(f'Failed to download {uri} after {max_tries} attempts.')

def transform_filename(uri):
    ''' Takes an uri and transforms it into a complete file path.
    Inputs:
    uri(string): a string of an uri.
    Returns:
    string: a string of a corresponding complete file path
    '''
    element_type = uri[1] #Determining the type of the element based on the second character of the URI
    uri = uri[1:] #Removing the leading slash from the URI

    if element_type == 'a': #If an element is an edge
        uri = re.findall('^(.*)\[.*?\]',uri)[0] + re.findall('^.*(\[.*?\])',uri)[0].replace('/','@')#Returning all non-overlapping matches of the regular expression as a list of strings of given characters in the URI, then replacing those strings with other characters.
        uri += '.json' #Adding '.json' to the end of the URI
    elif element_type == 'c': #If the element type is a node
        uri += '.json' #Adding '.json' to the end of the URI

        #Check if the URI contains square brackets
        if re.match('^.*\[.*?\].*',uri):
            blocks = re.findall('^(.*)(\[.*?\])(.*)',uri)[0] #Splitting the URI into 3 blocks: before the brackets, within the brackets, and after the brackets
            uri = blocks[0] + blocks[1].replace('/','@') + blocks[2] #Replacing slashes within the brackets and rearranging the URI
    return uri

def transform_path(uri):
    '''Takes a URI and transforms it into a directory structure.
    Inputs:
    uri(string): A string of URI.
    Returns:
    string: a directory structure.
    '''
    element_type = uri[1]
    if element_type == 'c': #If the element is a node
        path = re.findall('^(.*)\[.*?',uri)[0] #Using regular expression to find and only retain the first part of the file from the URI
    else: #If the element is an edge
        path = re.findall('^/(.*)/\[.*?\]',uri)[0] #Using regular expression to find and remove the last level of the filename contained within the bracket

    return path

def get_page(uri):
    '''Takes a URI and returns a page. If the page is already saved locally, it reads the page from the local file.
    Otherwise,it downloads the page and indicates that the page needs to be saved.
    Inputs:
    uri(string): A string of a URI.
    Returns:
    tuple: A tuple containing the page data and a boolean indicating whether the page needs to be saved.
    '''
    need_to_save = False
    filename = transform_filename(uri)
    try: #Attempting a block of code before throwing an exception
        with open(filename, 'r') as f: #Opening a file in a read mode, automatically closing the file when done reading (with)
            page = json.loads(f) #parsing a JSON formatted string into a Python object
    except FileNotFoundError: #Throwing an exception error if FileNotFoundError
        page = download(uri) #Downloading the page data
        need_to_save = True
    except json.JSONDecodeError: #Throwing an exception by json module when there is an error with JSON decoding a file.
        print(f"Error: {filename} does not contain valid JSON.")
        page = None
    return page, need_to_save

def get_node(uri):
    '''Takes a URI and returns a node. If the node is paginated, the function fetches all the pages and combines them into a single node.
    Inputs:
    uri(string): A string of a URI.
    Returns:
    tuple: A tuple containing the combined node data and a boolean indicating whether the node was fetched from a web
    '''
    node,is_from_web = get_page(uri) #Returning node and whether it is from the web.
    if is_from_web and 'view' in node: #If the node was fetched from the web and is paginated, combine all the pages
        node = combine_pages(node)
        return node, is_from_web
def combine_pages(node):
    ''' Takes all the pages of a paginated node and combines them into a single node.
    Inputs:
    node(dictionary): A dictionary containing the initial page of the node.
    Returns:
    dictionary: A dictionary containing the combined node data
    '''
    while 'view' in node and 'nextPage' in node['view']:#Checking two conditions: whether the 'view' key exists in the 'node' dictionary and if the first condition is true, it checks if 'nextPage' is a key in the dictionary that the node['view'] points to.
        try:  # Attempting a block of code before throwing an exception
            next_page = download(node['view']['nextPage'])
        except Exception as e: #Throwing an Exception error as a variable e
            print(f"An error occurred while downloading the next page:{e}")
            break #Stopping at this error
        node['edges'] += next_page['edges'] #Adding the edge of 'nextPage' to the existing edges of the node
    del node['view'] #Removing the 'view' key as we no longer need it
    return node

def get_edge(uri):
    '''Takes a URI and returns the edge.
    Inputs:
    uri(string): A string of a URI.
    Returns:
    tuple or None: A tuple containing the page data and a boolean indicating whether we need to save the edge locally if can be fetched, None otherwise.
    '''
    try:# Attempting a block of code before throwing an exception
        return get_page(uri)
    except Exception as e:
        print(f"An error occurred while fetching the edge data:{e}")
        return None
def get_multi_model(info):
    '''Takes information of an entity and return the content of the file.
    Inputs:
    info(dictionary): A string of the information of an entity.
    Returns:
    bytes or None: The content of the file as bytes if it could be read, or None otherwise.
    '''
    filename = transform_filename(info['@id']) #Transform the id of the information with into a complete file path
    filename = filename + '.' + info['form'] #Adding '.' and form of the information to the filename
    try: # Attempting a block of code before throwing an exception
        with open(filename, 'rb') as f: #Openning the file in a binary mode for reading and closing it when done
    except FileNotFoundError: ##Throwing an exception error if FileNotFoundError
        print(f"Error:The file {filename} does not exist.")
        content = None
    except IOError as e: #Throwing an IOError exception as a variable e
        print(f"Error:An I/O error occurred while reading the file {filename}: {e}")
        content = None
        return content





