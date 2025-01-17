import os
from dotenv import load_dotenv
import google.generativeai as genai
import torch
import transformers
import logging #a tracing library
#from stock_market_application_data import *
#from stock_data_analysis import stock_data
#from crewai import Agent #a framework for orchestrating role-playing, autonomous AI agents.
import getpass #a module to securely process password prompts when the program interacts with users
import vertexai as v1
import sqlite3
import pandas as pd
#from crewai_tools import (CodeDocsSearchTool,
 #                         CSVSearchTool,
  #                        DirectorySearchTool,
  #                        DOCXSearchTool,
  #                        DirectoryReadTool,
  #                        FileReadTool,
  #                        GithubSearchTool,
  #                        TXTSearchTool,
  #                        JSONSearchTool,
  #                        PDFSearchTool,
  #                        RagTool,
  #                        ScrapeWebsiteTool,
  #                        ScrapeElementFromWebsiteTool,
  #                        WebsiteSearchTool,
  #                        BrowserbaseLoadTool,
   #                       EXASearchTool)
from typing import Annotated
from langgraph.graph.message import add_messages
#from IPython.display import HTML, Markdown, display
from alpha_vantage.timeseries import TimeSeries
import yfinance as yf
import pprint


google_api_key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=google_api_key)
PROJECT_ID = os.getenv('PROJECT_ID')
REGION = "australia-southeast2"
from vertexai.preview.extensions import Extension

def _set_if_undefined (var:str) -> None: #function definition, specify the argument types and the return of the function,
    #'var:str' => variable should be a string, '-> None': does not return a value
    if os.environ.get(var):
        return
    os.environ[var] = getpass.getpass(var) #if a user types in a password, process the password
#Configure tracing to visualise and debug the agent
logging.basicConfig(filename='agent_trace.log', level=logging.DEBUG)
decision_list = ['buy','sell','hold','diversify','manage risk', 'rebalance']
def tracing(stock_data):
    ''' Takes stock data and returns the tracing of the agent's action.
    Input: stock_data
    Returns:
    string: Tracing of the agent's action
    '''
    logging.basicConfig(filename='agent_trace.log', level=logging.DEBUG)
    for action in decision_list:
        decision = action
    logging.info(f"Decision suggested: {decision}")
    return decision
#genai.configure(api_key=os.environ['GEMINI_API_KEY'])
#gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash"

def fetch_real_time_data(symbol):
    '''
    Fetches real-time stock data for a given stock symbol.
    Inputs:
        symbol(str): A string of stock symbol
     Returns:
        DataFrame: A pandas DataFrame of stock data
    '''
    stock = yf.Ticker(symbol)
    stock_data = stock.history(period='1d')
    return stock_data

#Initialising parameters
num_agents = 8 #the number of agents. Agent 1: a world-class stock market investment analyst. Agent 2: An analysis reviewer. Agent 3: An analysis changer. Agent 4: A world-class stock market researcher.
#Agent 5: A research critic. Agent 6: A research changer. Agent 7: A financial literacy educator. Agent 8: An educational content critic. Agent 9: An educational content changer.
num_tokens = 10000 #number of tokens, every 100 tokens represent 60-80 words
max_seq_len = 1024 #maximum sequence length, controlling the number of tokens of the input
dim = 320 #dimension, representing the number of features
depth = 8 #the number of layers in a neural network
dim_head = 32 #referring the dimensionality of key,query, and value vectors in the self-attention mechanism
heads = 6 #A head = the part of the model that makes predictions. Every model has a backbone and a head (backbone network) and a certain amount of prediction heads
use_abs_pos_emb = False #use absolute positional embedding. Positional embeddings inform about the position of tokens in input sequence
attn_flash = True #using flash attention, an efficient and precise Transformer acceleration technique,perceiving memory read and write ops, changing blocks of query,key,and value from the GPU's HBM(main memory) to SRAM.
attn_kv_heads = 2 #key-value caching pair with attention heads
qk_norm = True #normalisation of the query and key vectors
attn_qk_norm = True #normalisation of the query and key vectors of attention mechanism
attn_qk_norm_dim_scale = True # the attention score computed as a dot product of query and key vectors,scaled down by the square root of the dimension of these vectors => applying learned scale across the dimension of features, then normalising the query and key vectors
post_fusion_norm = True # controlling whether we apply normalisation after fusion, fusion refers to combining multiple modalities from different stages
post_modal_transform_norm = True #controlling whether we apply normalisation after multimodal transformation



#def get_agents_with_stock_data(symbol):
 #   '''Takes Gemini model and returns a list of agents.
 #   Input:
 #   model: Gemini model
 #   Returns:
 #   list: a list of agents
 #   '''
 #   stock_data = fetch_real_time_data(symbol)
 #    agents = []
 #   for i in range(num_agents):
 #       agent = Gemini(
 #           num_tokens=num_tokens,
 #           max_seq_len=max_seq_len,
 #           dim=dim,
 #           depth=depth,
 #           dim_head=dim_head,
 #           heads=heads,
 #           use_abs_pos_emb=use_abs_pos_emb,
 #           attn_flash=attn_flash,
 #           attn_kv_heads=attn_kv_heads,
 #           qk_norm=qk_norm,
 #           attn_qk_norm=attn_qk_norm,
 #           attn_qk_norm_dim_scale=attn_qk_norm_dim_scale,
 #           post_fusion_norm=post_fusion_norm,
  #          post_modal_transform_norm=post_modal_transform_norm,
 #       )
 #       agent.stock_data = stock_data
 #       agents.append(agent)
 #   return agents


flash = genai.GenerativeModel('gemini-1.5-flash')
response = flash.generate_content("Explain Socially Responsible Investing to me like I am a 10th grader")
print(response.text)

chat = flash.start_chat(history=[])
response = chat.send_message("Classify the types of investment risks based on high-risk, low-risk and moderate-risk")

#agents = get_agents_with_stock_data('AAPL')

def create_database(db_name):
    """
    Create a database for stock market data
    """
    conn = sqlite3.connect(db_name)
    return conn
def fetch_real_time_data(symbol):
    '''
    Fetches real-time stock data for a given stock symbol.
    Inputs:
        symbol(str): A string of stock symbol
     Returns:
        DataFrame: A pandas DataFrame of stock data
    '''
    stock = yf.Ticker(symbol)
    stock_data = stock.history(period='1d')
    return stock_data

def create_table(conn):
    """
    Create a table for stock market data
    """
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS companies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        industry TEXT NOT NULL,
        )
    ''')
    conn.commit()
    return cursor

def insert_companies(conn, companies):
    """
    Insert companies into the tables

    """
    cursor = conn.cursor()
    cursor.executemany('''
    INSERT INTO companies (name, industry) VALUES (?,?)
    ''', companies)
    conn.commit()
def main():
    db_name = 'companies.db'
    companies = [
        ('Apple', 'Technology'),
        ('Tesla', 'Automotive'),
        ('Microsoft', 'Technology'),
        ('Amazon', 'Retail'),
        ('Google', 'Technology'),
        ('Facebook', 'Technology'),
        ('Netflix', 'Entertainment'),
        ('Twitter', 'Social Media'),
        ('Shopify', 'Retail'),
        ('Zoom', 'Technology'),
        ('Slack', 'Technology'),
        ('Spotify', 'Entertainment'),
        ('Uber', 'Transportation'),
        ('Lyft', 'Transportation'),
        ('Pinterest', 'Social Media'),
        ('Snapchat', 'Social Media'),
        ('TikTok', 'Social Media'),
        ('Instagram', 'Social Media'),
        ('WhatsApp', 'Social Media'),
        ('LinkedIn', 'Social Media'),
        ('Reddit', 'Social Media'),
        ('Twitch', 'Entertainment'),
        ('Tinder', 'Social Media'),
        ('Bumble', 'Social Media'),
        ('Grindr', 'Social Media'),
        ('Hinge', 'Social Media'),
        ('Match', 'Social Media'),
        ('OkCupid', 'Social Media'),
        ('Plenty of Fish', 'Social Media'),
        ('eHarmony', 'Social Media'),
        ('Zoosk', 'Social Media'),
        ('Christian Mingle', 'Social Media'),
        ('JDate', 'Social Media'),
        ('Farmers Only', 'Social Media'),
        ('Alibaba', 'Retail'),
        ('JD.com', 'Retail'),
        ('Tencent', 'Technology'),
        ('Baidu', 'Technology'),
        ('Weibo', 'Social Media'),
        ('WeChat', 'Social Media'),
        ('QQ', 'Social Media'),
        ('Douyin', 'Social Media'),
        ('Kuaishou', 'Social Media'),
        ('Sina Weibo', 'Social Media'),
        ('YY', 'Social Media'),
        ('VK', 'Social Media'),
        ('Odnoklassniki', 'Social Media'),
        ('MySpace', 'Social Media'),
        ('Friendster', 'Social Media'),
        ('Orkut', 'Social Media'),
        ('Hi5', 'Social Media'),
        ('Bebo', 'Social Media'),
        ('Tagged', 'Social Media'),
        ('Xanga', 'Social Media'),
        ('LiveJournal', 'Social Media'),
        ('Renren', 'Social Media'),
        ('Kaixin', 'Social Media'),
        ('Qzone', 'Social Media'),
        ('Mixi', 'Social Media'),
        ('Cyworld', 'Social Media'),
        ('Naver', 'Social Media'),
        ('Daum', 'Social Media'),
        ('Cyworld', 'Social Media'),
        ('Nate', 'Social Media'),
        ('Me2day', 'Social Media'),
        ('KakaoTalk', 'Social Media'),
        ('Line', 'Social Media'),
        ('Kik', 'Social Media'),
        ('Viber', 'Social Media'),
        ('Telegram', 'Social Media'),
        ('Signal', 'Social Media'),
        ('Threema', 'Social Media'),
        ('Wickr', 'Social Media'),
        ('Snapchat', 'Social Media'),
        ('TikTok', 'Social Media'),
        ('Instagram', 'Social Media'),
        ('WhatsApp', 'Social Media'),
        ('LinkedIn', 'Social Media'),
        ('IBM', 'Technology'),
    ]

