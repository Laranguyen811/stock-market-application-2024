import random
import os
import pandas as pd


def get_proxy(proxy_list):
    '''Takes a list of proxies and returns username and password corresponding to each proxy.
    Inputs:
    proxy_list(list): a list of proxies
    Returns:
    string: username and password corresponding to each proxy
    '''
    #Load proxies and credentials from environment variables
    proxy_list = os.getenv('PROXY_LIST','').split(',') # Getting the environmental variable called 'PROXY_LIST' and splitting each component at commas
    username = os.getenv('USERNAME','') # Getting the environmental variable called 'USERNAME'
    password = os.getenv('PASSWORD','') # Getting the environmental variable called 'PASSWORD'

    #Check if proxy list is empty
    if not proxy_list:
        raise ValueError("Proxy list is empty")

    #Choose a random proxy
    proxy = random.choice(proxy_list) # A function to pick a random proxy from the proxy list, for load balancing (distributing requests amongst proxies to prevent any proxy from becoming a bottleneck),avoid rate limiting (rotating through proxies without hitting some website's limits), anonymity and privacy (harder for websites to track activity),overcoming geographical restrictions(can access geographically restricted resources if have proxies in different locations)

    return f'{username}:{password}@{proxy_list[0]}'

def get_proxy_list(proxy_list_origin):
    '''Takes a list of proxies'origins and returns a list of proxies corresponding to each proxy.
    Inputs:
    proxy_list_origin(list): A list of proxies' origins
    Returns:
    list: a list of proxies with corresponding username and password
    '''
    proxy_list_origin = os.getenv('PROXY_LIST','').split(',') #Obtaining a list of proxies' origins
    username = os.getenv('USERNAME','')
    password = os.getenv('PASSWORD','')

    if not proxy_list_origin:
        raise ValueError("Proxy list is empty")

    proxy_list = [f'{username}:{password}@{proxy}' for proxy in proxy_list_origin] #Obtaining a list of proxies and corresponding username and password for each proxy
    return proxy_list
    print(f"Proxy list: {proxy_list_origin}")