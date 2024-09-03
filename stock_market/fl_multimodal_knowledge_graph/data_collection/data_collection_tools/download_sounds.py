import requests
from bs4 import BeautifulSoup
import re
import os
import subprocess
import signal
import random
from stock_market.fl_multimodal_knowledge_graph.data_collection.data_collection_tools import proxy_tool

def get_content(url):
    '''Takes a GET request to the specified URL and returns the response content.
    Inputs:
        url(string): A string of the URL.
    Returns:
        bytes: The response content if the request was successful.
    Raises:
    requests.exceptions.RequestException: If the request is not successful.
    '''
    try: #Attempting a block of code before throwing an exception
        #proxy_list = ['proxy1', 'proxy2', 'proxy3']  # Example proxy list
        user_agent = 'Mozilla/5.0 (X11; Linux x86_64; rv:45.0) Gecko/20100101 Firefox/45.0'
        proxy_ip = proxy_tool.get_proxy() # Obtaining proxy IP
        proxies = {'http': 'http://'.format(proxy_ip), 'https':'https://{}'.format(proxy_ip)} #Creating a dictionary of proxies
        response = requests.get(url,headers={'User-Agent':user_agent},proxies=proxies,timeout=10)#Sending a GET request to the specified URL. The headers parameter is used to pass HTTP headers to the request, and the proxies parameter is used to specify the proxies that the request should go through. The timeout parameter specifies the maximum number of seconds to wait for a response before giving up.
        response.raise_for_status() #If the request is unsuccessful, raise an exception
        response.encoding = response.apparent_encoding #Determining the encoding of the webpage for decoding a text.
    except requests.exceptions.RequestException as e: #Throwing a ResquestException exception as variable e
        print(f"An error occurred while sending the request: {e}")
        raise
    else:
        return response.content #Returning the content of the response

def parser_content(html_content):
    '''Takes the given HTML content, parses it and extracts all URLs from 'a' tags with class 'title'.
Inputs:
    html_content(string): the string of HTML content to parse
Returns:
    list: A list of URLs extracted from the HTML content.
    '''
    soup = BeautifulSoup(html_content,'html.parser')#Initiating BeautifulSoup object for easier processing
    sound_url_elements = soup.find_all('a',class_='title')#Extracting required content
    urls = [element.get('href') for element in sound_url_elements]#Extracting 'href' attribute from each element and add to the list

    return urls

def download_sound(search_term, save_path, max_files=5,max_size=3.0,timeout=60.0):
    '''Takes a search term, downloads a sound file from freesound.org and saves it to a path.
    Inputs:
        search_term(string): A string of search term for on freesound.org to
        save_path(string): A string of path to save the downloaded files.
        max_files(int): An integer specifying the maximum number of files to download.
        max_size(float): A float number specifying the maximum size of the files to download, in MiB.
        timeout(float): A float specifying the maximum time to wait for a download to complete, in seconds.

    Returns:
        list: A list of the file extensions of the downloaded files.
    '''
    url = f"https://freesound.org/search/?q={search_term}"
    content = get_content(url)
    if not content:
        return []
    urls = parser_content(content)
    if not urls:
        return []
    download_files = []

    for url in urls[:max_files]:
        file_url = f'https://freesound.org{url}'

        size = get_file_size(file_url) #Check the file size, skip if too large
        if size > max_size:
            continue

        download_file(file_url,save_path,timeout) #Download the file

        file_extension = os.path.splitext(file_url)[1][1:] #Adding the file extension to the list of download a file
        download_files.append(file_extension)
    return download_files

def get_file_size(url):
    '''Takes a URL and returns the file size.
    Inputs:
        url(string): A string of url of the file to download.
    Returns:
        float: The float number of the file size in MiB.
    '''
    info_str = subprocess.check_output(f'you-get --info {url}',shell=True).decode('utf-8') #Running a command specified by the given argument, waiting for it to finish and then returning its output by bytestring in a formatted string literal. Command executed through shell, and output decoded via 'utf-8'.
    size_pattern = re.compile(r'(\d+\.\d*)[\s]MiB') #Matching sequence of one or more digits, followed by zero or more digits, then any whitespace character, with the exact string 'MiB'.
    size = float(size_pattern.findall(info_str)[0]) #Turning the first string of the output of info_str that matches the pattern of size_pattern into a float number
    return size

def download_file(url,save_path,timeout):
    '''Takes a URL and downloads the file to a specified path.
    Inputs:
        url(string): A string of the URL to download.
        save_path(string): A string of the path to save the file
        timeout(float): The float number of the maximum time in seconds to wait
    Returns:
        None
    '''
    cmd = f'you-get --skip-existing-file-size-check -o {save_path.replace("'","\\'")} -O {str(count)} {url}'#Using you-get to download media from a specific URL, skipping file size checks for existing files and saving the downloaded media to a specific directory with a specific filename.
    if random.randint(0,5) < 5: #If random intergers from 0 to 5 is less than 0
        cmd += f' -x {proxy_tool.get_proxy()}' #Adding commands to get proxies
    cmd += '> tmp.txt' #Adding commands to run the ls command and then writing the output into tmp.txt
    process = subprocess.Popen(cmd,start_new_session=True,shell=True) #Starting a new process to run a specific command, in a new session and through the shell.
    try:#Attempting a block of code before throwing an exception
        process.communicate(timeout=timeout) #Sending input to a process, waiting for it to complete(with a specific timeout), and then returning the process's output and error.
    except subprocess.TimeoutExpired: #Throwing an exception provided by subprocess module when a subprocess operation exceeds the specified timeout duration and clearing temp files
        process.kill() #Terminating the process and clearing the temp files immediately
        process.terminate()#Sending signals to the script to terminate the process
        os.killpg(process.pid, signal.SIGTERM) #Sending a SIGTERM signal to the process group identified by process.pid, asking all processes in the group to terminate
        subprocess.run(f'-rm {save_path.replace("'", "\\'")}/*.download',shell=True)
if __name__ == '__main__':
    download_sound('stock market','Sounds/stock_market')




