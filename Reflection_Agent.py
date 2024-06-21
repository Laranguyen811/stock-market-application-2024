import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, MessageGraph #END is a special node to terminate a graph, MessageGraph is a subclass of StateGraph,
#a single, append-only list of messages
import getpass #a module to securely process password prompts when the program interacts with users
def _set_if_undefined (var:str) -> None: #function definition, specify the argument types and the return of the function,
    #'var:str' => variable should be a string, '-> None': does not return a value
    if os.environ.get(var):
        return
    os.environ[var] = getpass.getpass(var) #if a user types in a password, process the password



