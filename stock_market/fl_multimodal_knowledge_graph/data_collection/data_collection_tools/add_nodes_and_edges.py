import uritools as ut
from urllib.parse import urljoin
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
import rdflib

def preprocess_and_standardize_text(text):
    """Take a string and returns lemmatized tokens
    Input:
    text (string): Text to be standardized
    Returns:
    string: a string of lemmatized tokens
    """
    lemmatizer = WordNetLemmatizer() #lemmatizing text to find the root of words, help agents understand conversations better
    text = text.lower()
    tokens = word_tokenize(text) #tokenizing words
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

def join_uri(*pieces):

