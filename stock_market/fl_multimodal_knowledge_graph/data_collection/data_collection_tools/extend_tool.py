import re
import os
from typing import List,Dict,Any
import logging

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

def multimodel_uri(uri,model):
    '''Takes a URI of a node or edge and returns the base URI of multimodal information.
    Inputs:
        uri(string): A string of a URI of a node or edge.
    Returns:
        string: A string of the base URI of the multimodal information.
    '''
    return '/m' + uri[:2] + '/' + model + '/'

def get_text(uri):
    '''Takes a URI and returns the text we use for searching.
    Inputs:
        uri(string): A string of a URI which we can use as a base for searching
    Returns:
        string: A string of txt we use for searching
    '''
    relationship_dictionary= {
        '/r/PartOf/':'',
        '/r/HasProperty/':'',
        '/r/AtLocation/':'',
        '/r/Causes/':'',
        '/r/HasA/':'has',
        '/r/UsedFor/':'used for',
        '/r/CapableOf/':'can',
        '/r/MadeOf/':'is made of',
        '/r/ReceivesAction/':'can be',
        '/r/CreatedBy/':'created by',
        '/r/LocatedNear/':'near',
        '/r/Causes': '',
        '/r/HasSubevent': '',
        '/r/HasFirstSubevent': '',
        '/r/HasLastSubevent': '',
        '/r/HasPrerequisite': '',
        '/r/MotivatedByGoal': '',
        '/r/ObstructedBy': '',
        '/r/Desires': '',
        '/r/CreatedBy': 'created by ',
        '/r/DistinctFrom': 'distinct from ',
        '/r/SymbolOf': 'symbol of ',
        '/r/DefinedAs': 'defined as ',
        '/r/LocatedNear': 'near ',
        '/r/HasContext': 'has context ',
        '/r/SimilarTo': 'similar to ',
        '/r/CausesDesire': 'causes desire ',
        '/r/TradedOn': 'traded on ',
        '/r/HasCEO': 'has CEO ',
        '/r/Industry': 'industry ',
        '/r/HeadquarteredIn': 'headquartered in ',
        '/r/Owns': 'owns ',
        '/r/PartnersWith': 'partners with ',
        '/r/CompetesWith': 'competes with ',
        '/r/ReportedEarnings': 'has reported earnings ',
        '/r/HasMarketCap': 'has market cap ',
        '/r/HasNews': 'has news ',
        '/r/OffersProduct': 'offers product ',
        '/r/HasMarketSentiment': 'has market sentiment ',
        '/r/HasEvent': 'has event ',
        '/r/HasInvestorType': 'has investor type ',
        '/r/InvestorInterest': 'has investor interest ',
        '/r/InvestorBehavior': 'has investor behavior ',
        '/r/InnovatesIn': 'innovates in ',
        '/r/MarketLeaderIn': 'is market leader in ',
        '/r/AcquiredBy': 'is acquired by ',
        '/r/Acquires': 'acquires ',
        '/r/ImplementsStrategy': 'implements strategy ',
        '/r/InvestorActivity': 'investor activity ',
        '/r/HasLiquidity': 'has liquidity ',
        '/r/HasMarketDepth': 'has market depth ',
        '/r/HasAnalysis': 'has analysis ',
        '/r/HasTrend': 'has trend ',
        '/r/ImpactsEconomy': 'impacts economy ',
        '/r/UsesAlgorithmicTrading': 'uses algorithmic trading ',
        '/r/HasTraderType': 'has trader type ',
        '/r/HasSharePrice': 'has share price ',
        '/r/HasCustomerLoyalty': 'has customer loyalty ',
        '/r/HasWorkforce': 'has workforce ',
        '/r/UsesAdvertising': 'uses advertising ',
        '/r/HasProductPrice': 'has product price ',
        '/r/HasCorporateAction': 'has corporate action ',
        '/r/HasEarningsForecast': 'has earnings forecast ',
        '/r/HasUserBase': 'has user base ',
        '/r/SubjectToRegulation': 'subject to regulation ',
        '/r/TraderBehavior': 'trader behavior ',
        '/r/HasPhilanthropy': 'has philanthropy ',
        '/r/HasEthics': 'has ethics ',
        '/r/CommunityInvolvement': 'community involvement ',
        '/r/HasCulture': 'has culture ',
        '/r/CSRMetrics': 'CSR metrics ',
        '/r/CSRKPIs': 'CSR KPIs ',
        '/r/HasCertifications': 'has certifications ',
        '/r/ESGFactors': 'ESG factors '
        } #A dictionary to map each relationship type to a corresponding human-readable relationship text.
    if uri[1] =='c':#Node
        text = uri_to_label(uri)
    else:
        blocks = re.findall(r'^.*\[(.*?)\]', uri)[0].split(',') #Matching a pattern of a URI, getting the first string of it as blocks
        start_node = uri_to_label(blocks[1]) + '' #Converting the second string of blocks to a label and adding a whitespace as a start of a node
        end_node = uri_to_label(blocks[2]) #Converting the third string of blocks to a label as an end of a node

        edge = blocks[0] #Obtaining the first string of the blocks as the edge
        relation = relationship_dictionary.get(edge,'') #Retrieving a value associated with a key called edge from a relationship dictionary a scheme based on a relationship type
        text = (start_node + ' ' + end_node).replace(' ','+')#Converting to text based on the start of a node and the end of a node, replacing a whitespace with a plus sign

    return text

def add_multimodel_node_audio(text : str,path : str) -> List[str]:
    '''Takes a path used for saving audio files and a text used for searching and returns a list of file formats.
    Inputs:
        text(string): A string of a text used for searching for audio files.
        path(string): A string representing the path used for saving audio files.
    Returns:
        list: A list of file formats for the saved audio files,e.g., ['wav', 'wav', 'mp3'].
    '''
    #Setting up logging
    logging.basicConfig(level=logging.INFO) #Configuring the logging systems with basic configs using a convenience function logging.basicConfig

    #Ensuring the directory exists
    if not os.path.exists(path): #If the path does not exist
        os.makedirs(path) #Creating a new directory (path)
        logging.info(f"Created directory at {path}") #Emitting log messages using the logging.info method
    try: #Attempting a block of code before throwing an exception
        file_formats = download_sounds.download_sound(text,path) # Creating a list of file formats from text and path
        logging.info(f"Download audio files: {file_formats}") #Emitting log messages using the logging.info method
        return file_formats
    except Exception as e: #Throwing an exception as variable e
        logging.error(f"An error occurred: {e}")#Logging error messages indicating serious problems using logging.error method
        return [] #Returning an empty list in case of an error to maintain a consistent return type, making it easier to handle the function's output without having to check for different types
def add_multimodel_node_image(text : str,path : str) -> List[str]:
    '''Takes a path to save image files and a text used for searching and returns a list of file formats.
    Inputs:
        string(text): A string of a text used for searching image files.
        string(path): A string representing the path used for saving image files.
    Returns:
        list: A list of file formats for the saved image files,e.g., ['jpg', 'png', 'gif'].
    '''
    #Setting up logging
    logging.basicConfig(level=logging.INFO) #Configuring the logging systems with basic configs using a convenience function logging.basicConfig

    #Ensuring the directory exists
    if not os.path.exists(path): #If the directory(path) does not exist
        os.makedirs(path) #Creating a directory (path)
        logging.info(f"Created directory at {path}") #Emitting log messages using the logging.info method
    try: #Attempting a block of code before throwing an exception
        file_formats = download_images.download_images(text,path)
        logging.info(f"Download image files: {file_formats}") #Emitting log messages using the logging.info method
        return file_formats
    except Exception as e: #Throwing an exception as variable e
        logging.error(f"An error occurred: {e}") #Logging error messages indicating serious problems using logging.error method
        return [] #Returning an empty list in case of an error to maintain a consistent return type, making it easier to handle the function's output without having to check for different types

def add_multimodel_node_visual(text : str,path : str) -> List[str]:
    '''Takes a path to save visual files and a text used for searching and returns a list of file formats.
    Inputs:
        string(text): A string of a text used for searching visual files
        string(path): A string representing the path used for saving visual files
    Returns:
        list: A list of file formats for the saved visual file, e.g, [.mp4,.avi,.mov,.wmv,.mkv,.ai,.eps,.pdf]
    '''
    #Setting up logging
    logging.basicConfig(level=logging.INFO) #Configuring the logging systems with basic configs using a convenience function logging.basicConfig

    #Ensuring the directory exists
    if not os.path.exists(path): #If the path does not exist
        os.makedirs(path) #Creating a path
        logging.info(f"Created directory at {path}") ##Emitting log messages using the logging.info method
    try: #Attempting a block of code before throwing an exception
        file_formats = download_visual.download_visual(text,path)
        logging.info(f"Download visual files: {file_formats}") #Emitting log messages using the logging.info method
        return file_formats
    except Exception as e:#Throwing an exception as variable e
        logging.error(f"An error occurred: {e}") #Logging error messages indicating serious problems using logging.error method
        return []


def add_multimodel(page:Dict[str,Any],model_type:str) -> Dict[str,Any]:
    '''Adds multimodal information to the page.
        Inputs:
            page(Dict[str,Any]):The page dicctionary containing '@id' key.
            model_type(str): The type of model to add (e.g. 'audio','image','visual').
        Returns:
            Dict[str,Any]:The updated page dicctionary with multimodal information.
    '''
    uri = page['@id']
    model_info = {} #Using a temporary dictionary to record the newly added multimodal information fields
    text = get_text(uri)

    logging.basicConfig(level=logging.INFO) #Setting up a basic config for logging
    def add_multimodal_node(node_type:str,add_function):
        newly_added = add_function(text,multimodal_path(uri,node_type)) #Downloading information and returning the list of newly added items
        if len(newly_added) > 0:
            model_info[node_type] = [] #Initialising the list as empty
            for i,form in enumerate(newly_added):
                entry = {'@id': multimodel_uri(uri,node_type) + str(i),'form': form} #Creating an entry using id as key, value is the base URI and the number of iteration and form
                if node_type == 'visual':
                    entry['with-audio'] = form[0]
                    entry['form'] = form[1]
                model_info[node_type].append(entry)
            logging.info(f'{node_type} added')
    #Adding audio, image, and visual nodes
    add_multimodal_node('audio',add_multimodel_node_audio)
    add_multimodal_node('image',add_multimodel_node_image)
    add_multimodal_node('visuals',add_multimodel_node_visual)








