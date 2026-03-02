# Import libraries
import os

import fastapi
from dotenv import load_dotenv
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import pyphen
#from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
import nltk
from readability import FleschReadingEase, GunningFogIndex, KincaidGradeLevel
import pandas as pd
from sentence_transformers import SentenceTransformer,util
from nltk.tokenize import sent_tokenize
import unstructured
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from PyPDF2 import PdfReader
import io
import pdfplumber
import google.genai as genai
# from transformers import
gemini_api_key = os.getenv("GOOGLE_API_KEY")
app = FastAPI()
# Enable CORS for React dev server
app.add_middleware(
    CORSMiddleware, # Cross-Origin Resource Sharing refers to the situations when a frontend running in a browser has JavaScript code communicating with a backend API running on a different domain.
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class SimplifyRequest(BaseModel):
    text: str

class SimplifyResponse(BaseModel):
    original_text: str
    simplified_text: str
    readability_scores: dict

# Obtain Gemini model
from google.api_core import client_options # pass configuration settings to the underlying Google API client
gem_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0,google_api_key=gemini_api_key, client_options={"api_endpoint":"generativelanguage.googleapis.com"})

try:
    loader = UnstructuredWordDocumentLoader(r"D:\Users\laran\PycharmProjects\Stock Market Platform\financial_literacy\Plain English Checklist - 1.docx", mode="elements")
    documents = loader.load()
    print(documents)
    raw_text = "\n".join([doc.page_content for doc in documents]) # Process the document to obtain raw text
    check_list_text = "\n".join([line.strip() for line in raw_text.splitlines() if line.strip()]) # Strip empty lines
    print(check_list_text)
except ImportError as e:
    print("Missing dependency:", e)
    print("Try running: Pip install unstructured")
client = genai.Client()
pdf_file = client.files.upload(file=r'D:\Users\laran\PycharmProjects\Stock Market Platform\financial_literacy\GSIR-20201.pdf')
print(f"{pdf_file}")



# Parse the response
#soup = BeautifulSoup(response.text, 'html.parser')
#paragraphs = soup.find_all('p') # Obtain all paragraphs
#clean_text = '.\n'.join(p.get_text() for p in paragraphs) # Clean the text
#print(clean_text)

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
5. Please explain Socially Responsible Investment and the necessary financial literacy for someone from a disadvantaged background to make a wise decision regarding Socially Responsible Investing.
"""
)

# Define instructions

instructions = """
You are a world-class money mentor and specialist educator. Your goal is to explain and break down complex financial concepts in a way that is accessible and easy to understand for people regardless of their educational or socio-economic backgrounds, teaching as if to an 8th or 9th grader.

Use plain, simple English language with easy-to-understand words and grammatical structures. Employ relatable home examples to illustrate concepts.

Turn financial concepts into interactive elements, including charts or quizzes, to deepen understanding. Provide practical exercises for real-life application of knowledge.

Always preserve privacy and confidential information. Be honest, helpful, and accurate.
"""
system_prompt = SystemMessagePromptTemplate.from_template(instructions)

# Construct the first message
first_message = HumanMessagePromptTemplate.from_template(
"""
    Please simplify the following sentences for an 8th-grader audience.
    Return each simplified version directly below the original.
    
    1. An ethical investment is an investment that is made with one's ethical or moral principles in mind.
    -> A person makes an investment based on their ethical or moral principles.
    
    2. Socially Responsibly Investing is an investing practice based on environmental, social, and governance criteria.
    -> This strategy of investing is good for people, society and our planet.
    
    3. Choosing the right Socially Responsible Investments depend on your goals and values.
    -> It is better to pick the investments that matches what you care about.
"""
)

# Check list prompt
check_list_prompt = "Please follow this check list" + check_list_text

# Chat prompt
chat_prompt = ChatPromptTemplate.from_messages([ system_prompt, (check_list_prompt), simplify_prompt_cot])

# Format prompt
modality_str = ",".join(["text","image"])
formatted_prompt = chat_prompt.format_prompt(modalities=modality_str).to_messages()

# Generate response
response = gem_model.invoke(formatted_prompt)
print(response.content)
content_response = response.content

import re


def clean_response(text):
    """
    Cleans the model's response by removing code blocks and markdown symbols and normalising whitespace.
    Inputs:
         text (str): a string of raw output.
     Return:
         str: a string of clean output.
    """
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # Remove code blocks
    text = re.sub(r'\*|\#|\-|>', '', text)  # Remove markdown symbols
    text = re.sub(r'\s+', ' ', text).strip()  # Normalise whitespace

    return text


raw_output = response.content

plain_output = clean_response(raw_output)
print(plain_output)

def estimate_syllables(text, lang='en_US'):
    """
    Calculates the number of syllables within a given text.
    Inputs:
          text (str): a string of text to calculate the number of syllables for.
          lang  (str): a string of specified language to count the number of syllables in.
    Returns:
          int: the number of syllables
    """
    dic = pyphen.Pyphen(lang=lang)
    words = nltk.word_tokenize(text)
    return sum(len(dic.inserted(word).split('-')) for word in words if word.isalpha())

def count_complex_words(text, lang='en_US'):
    """
    Calculates the number of complex words within a given text.
    Inputs:
        text: a string of text to calculate the number of complex words for.
    Returns:
        int: the number of complex words within the given text.
    """
    dic = pyphen.Pyphen(lang=lang)
    words = nltk.word_tokenize(text)
    return sum(1 for word in words if word.isalpha() and len(dic.inserted(word).split('-')) >= 3)

def evaluate_readability(text:str)-> dict:
    """
    Calculates the readability scores of a given text, including Flesch Reading Ease, Gunning-Fox Index and Kincaid Grade Level.
    Inputs:
        text (str): a string of a given text to evaluate its readability for.
    Returns:
        dict: a dictionary of readability scores for a given text.
    """
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)

    word_count = len(words)
    sentence_count = len(sentences)
    syllables = estimate_syllables(text)
    complex_words = count_complex_words(text)

    fre = FleschReadingEase(syllables=syllables, words=word_count, sentences=sentence_count)
    gfi = GunningFogIndex(complex_words=complex_words, words=word_count, sentences=sentence_count)
    kgl = KincaidGradeLevel(words=word_count, sentences=sentence_count,syllables=syllables)

    return {
        "Flesch Reading Ease": round(fre,3),
        "Gunning Fog Index": round(gfi,3),
        "Kincaid Grade Level": round(kgl,3)
    }
#orig_scores = evaluate_readability(clean_text)
#print(f"Original Scores:", orig_scores)
plain_scores = evaluate_readability(plain_output)
print(f"Plain Scores:",plain_scores)
@app.post("/api/simplify", response_model=SimplifyResponse)
async def simplify_text(request: SimplifyRequest):
    try:
        user_prompt = HumanMessagePromptTemplate.from_template(f"Please simplify this text: {request.text}")
        full_prompt = ChatPromptTemplate.from_messages([system_prompt, check_list_prompt,user_prompt])
        formatted_prompt = full_prompt.format_prompt(modalities=modality_str).to_messages()
        response = gem_model.invoke(formatted_prompt)
        plain_output = clean_response(response.content)
        scores = evaluate_readability(plain_output)
        return SimplifyResponse(original_text=request.text, simplified_text=plain_output, readability_scores=scores)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
