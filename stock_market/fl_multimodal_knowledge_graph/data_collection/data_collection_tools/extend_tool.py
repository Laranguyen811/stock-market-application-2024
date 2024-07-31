import re
import os
from stock_market.fl_multimodal_knowledge_graph.data_collection.data_collection_tools.uri_tool import uri_to_label
from stock_market.fl_multimodal_knowledge_graph.data_collection.data_collection_tools import fetch_tool,download_sounds,download_images

def multimodal_path(uri,model):
    '''Takes a URI of a node or edge and a specified modal type and returns a path to a file.
    Inputs:
        uri(string): A string of a URI
    Returns:
        path: A path to a file with the specified modal type
    '''
    path = 'm' + fetch_tool.transform_filename(uri)[1:] #Modifying the URI type at the beginning
    path = re.findall('(.*)\..*?',path)[0] #Removing the file type at the end
    path = path + '/' + model

    return path
