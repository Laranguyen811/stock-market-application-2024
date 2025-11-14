# Import libraries
import os
from dotenv import load_dotenv
import logging

from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
import nltk
from readability import FleschReadingEase, GunningFogIndex, KincaidGradeLevel
import pandas as pd
from sentence_transformers import SentenceTransformer,util
from nltk.tokenize import sent_tokenize
import unstructured
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
# from transformers import
gemini_api_key = os.getenv("GOOGLE_API_KEY")

# Obtain Gemini model
from google.api_core import client_options # pass configuration settings to the underlying Google API client
gem_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0,google_api_key=gemini_api_key, client_options={"api_endpoint":"generativelanguage.googleapis.com"})

try:
    loader = UnstructuredWordDocumentLoader("financial_literacy/Plain English Checklist - 1.docx", mode="elements")
    documents = loader.load()
    print(documents)
    raw_text = "\n".join([doc.page_content for doc in documents]) # Process the document to obtain raw text
    check_list_text = "\n".join([line.strip() for line in raw_text.splitlines() if line.strip()]) # Strip empty lines
    print(check_list_text)
except ImportError as e:
    print("Missing dependency:", e)
    print("Try running: Pip install unstructured")

url = 'https://www.gsi-alliance.org/wp-content/uploads/2021/08/GSIR-20201.pdf'
# GET the url
response = requests.get(url)
# Raise error if response fails
response.raise_for_status()

# Print the response
print(response.text)

# Parse the response
soup = BeautifulSoup(response.text, 'html.parser')
paragraphs = soup.find_all('p') # Obtain all paragraphs
clean_text = '.\n'.join(p.get_text() for p in paragraphs) # Clean the text
print(clean_text)

# Chain-of-thought prompt for Plain English response
simplify_prompt_cot = HumanMessagePromptTemplate.from_template(
"""
    Please simplify the content of the text in plain and easy language. Please use home examples that are universal and easy to understand across different cultures to make it easy to understand.
    Please use examples that are non-commercial and from the home. Avoid using examples related to specific toys, brands, or things that might not be common in all households globally.
    Please ensure that your simplification is easy enough for an 8-9 grader to understand.
    Please follow the  ISO 24495-1:2023 Plain Language guidelines.
    Please ensure there are no more than 11 words per sentence. Please make sure that each word has no more than 3 syllables.
    Please use simple, common words to avoid confusion. When unfamility terms are necessary, explain them in context and provide an in-text definition using simple language.
    Please use active voice. Avoid passive voice.
    Please break or chunk information into short sections. Please divide content into short chunks of information with informative, clear headers.
    Please use simple sentence structures. Avoid complex use of parentheses, commas, and semicolons.
    Please use visual cues to draw attention to main points. Signal where to find important information and use visual elements such as arrows, boxes, bullets, bold, and larger font to add emphasis.
    Please use visual aids to support the main message and represent the intended audience. Please use visual aids to reinforce rather than distract from the content.
    Please use numbers that are clear and easy to understand. Please use whole numbers rather than fractions and decimals. Please add context for numbers in the form of words or additional numbers to indicate clear use of numbers if it fits.
    Please clearly identifies at least one action the user can take.
    Please address the user directly when describing actions.
    Please break down any action into manageable, explicit steps. Please avoid using ambiguous terms that users can interpret incorrectly.
    Please use language and examples that would be familiar to the audiences.
    Please use visual aids that are diverse regarding relevant race, ethnicity, age, gender, ability, and other characteristics.
    Please avoid perpetuating stereotypes. Please consider whether any language, examples, or images used in the material could unintentionally reinforce stereotypes.

    EXAMPLE: A ethical investment is an investment that is made with one's ethical or moral principles in mind.
    EXAMPLE: Socially Responsibly Investing is an investing practice based on environmental, social, and governance criteria
    EXAMPLE: Financial literacy relates to a person's competency for managing money. There are 5 categories of financial literacy: knowledge of financial concepts, ability to communicate about financial concepts, aptitude in managing personal finances, skills in making appropriate finance decisions and confidence in planning effectively for future needs.



Let's think step by step:
1. What are the core characteristics of an ethical investment?
2. What are the core characteristics of a Socially Responsible Investment?
3. What are the core characteristics of basic financial literacy?
3. Think of examples of an ethical investment an intersectional woman can make.
4. Think of examples of a Socially Responsible Investment a low-income individual can make.
4. Classify financial skills a marginalised individual need to decide which ethical and/or Socially Responsible Investment to buy.
5. Please explain the social enterprise structure options.
"""
)

