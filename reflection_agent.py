import os
from dotenv import load_dotenv
import google.generativeai as genai
import torch
import transformers
import logging #a tracing library
from stock_market_application_data import *
from stock_data_analysis import stock_data
from crewai import Agent #a framework for orchestrating role-playing, autonomous AI agents.
import getpass #a module to securely process password prompts when the program interacts with users
from crewai_tools import (CodeDocsSearchTool,
                          CSVSearchTool,
                          DirectorySearchTool,
                          DOCXSearchTool,
                          DirectoryReadTool,
                          FileReadTool,
                          GithubSearchTool,
                          TXTSearchTool,
                          JSONSearchTool,
                          PDFSearchTool,
                          RagTool,
                          ScrapeWebsiteTool,
                          ScrapeElementFromWebsiteTool,
                          WebsiteSearchTool,
                          BrowserbaseLoadTool,
                          EXASearchTool)
from gemini_torch.model import Gemini

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
#gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")
gemini_model = Gemini(
    num_tokens=10000, #number of tokens, every 100 tokens represent 60-80 words
    max_seq_len=1024,#maximum sequence length, controlling the number of tokens of the input
    dim=320,#dimension, representing the number of features
    depth=8,#the number of layers in a neural network
    dim_head=32,#referring the dimensionality of key,query, and value vectors in the self-attention mechanism
    heads=6,#A head = the part of the model that makes predictions. Every model has a backbone and a head (backbone network) and a certain amount of prediction heads
    use_abs_pos_emb=False,#use absolute positional embedding. Positional embeddings inform about the position of tokens in input sequence
    attn_flash=True,#using flash attention, an efficient and precise Transformer acceleration technique,perceiving memory read and write ops,
    #changing blocks of query,key,and value from the GPU's HBM(main memory) to SRAM.
    attn_kv_heads=2,#key-value caching pair with attention heads
    qk_norm=True,#normalisation of the query and key vectors
    attn_qk_norm=True,#normalisation of the query and key vectors of attention mechanism
    attn_qk_norm_dim_scale=True,# the attention score computed as a dot product of query and key vectors,scaled down by the square root of
    #the dimension of these vectors => applying learned scale across the dimension of features, then normalising the query and key vectors

)

#Actor 1: A Stock Market Investment Analyst
    #1. Creating a Persona: defining the tone and expertise level of the AI.
stock_market_investment_analyst = Agent(
    role="Stock Market Investment Analyst",
    goal="Provide financial analysis and insights about the stock market investment",
    verbose=True,
    memory=True,
    backstory=(
        "You are a world-class stock market investment analyst capable of analyzing both qualitative and quantitative data."
        "You have a deep understanding of financial markets, economic principles and industry trends."
        "You proactively suggest actions based on users' needs and preferences."
        "You communicate in a friendly, respectful, inclusive, truthful and professional manner."
         "You can process and generate text, images, video and audio."
    ),
    tools=[
        CodeDocsSearchTool,
        CSVSearchTool,
        DirectorySearchTool,
        DOCXSearchTool,
        DirectoryReadTool,
        FileReadTool,
        GithubSearchTool,
        TXTSearchTool,
        JSONSearchTool,
        PDFSearchTool,
        RagTool,
        ScrapeWebsiteTool,
        ScrapeElementFromWebsiteTool,
        WebsiteSearchTool,
        BrowserbaseLoadTool,
        EXASearchTool],
    allow_delegation=True,
)







