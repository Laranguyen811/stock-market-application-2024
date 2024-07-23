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

