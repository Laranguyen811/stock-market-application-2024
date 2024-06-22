import os
from dotenv import load_dotenv
import google.generativeai as genai

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")
#a single, append-only list of messages
import getpass #a module to securely process password prompts when the program interacts with users
def _set_if_undefined (var:str) -> None: #function definition, specify the argument types and the return of the function,
    #'var:str' => variable should be a string, '-> None': does not return a value
    if os.environ.get(var):
        return
    os.environ[var] = getpass.getpass(var) #if a user types in a password, process the password



