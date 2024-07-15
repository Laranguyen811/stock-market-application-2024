import uritools as ut
from urllib.parse import urljoin,urlparse
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
    """ Takes constituent pieces which we should join with slashes (/) to create a joined url
    Inputs:
    pieces (list): A list of the constituent pieces
    Returns:
    string: a string of a new uri
    """
    uri = '/'+ ('/'.join([piece.strip('/') for piece in pieces])) #stripping off the leading and trailing slashes of constiuent
    #pieces and join them together to create a joined url with a leading slash
    return uri

def concept_url(lang,text,*more):
    """Takes a word or a phrase of a particular language and returns a joined uri to build a concept url (a representation of a concept
    Inputs:
    lang (string): a particular language
    text (string): a string of text to be joined
    more (string): any other string to add to the uri, including part of speech, disambiguation, etc.

    Returns:
    string: a joined uri to build a concept url
    """
    assert ' ' not in text, "%r is not in normalised form" % text #an error exception to see if there are spaces in the text (or
    #whether it is in a normalised form)
    if len(more) > 0:
        if len(more) != 1:
            # We misparsed a part of speech; everything after the text is probably junk
            more = []
        else:
            for dis1 in more[1:]:
                assert '_' not in dis1, "%r is not in normalised" %dis1 #ensuring all disambiguations (the process of making an ambiguous expression clear and understandable are in normalised form
    return urlparse(('','','/c/'+ lang +'/'+text + '/' + '/'.join(more),'','','')) #using urlunparse to build the URI

def compound_uri(op, args):
    """Takes a main operator with the slash included and an arbitrary number of arguments and returns the URI representing the entire compound structure (made up of different data types)
    Inputs:
    op (string): the operator of the compound structure
    args (list): the list of arguments
    Returns:
    string: the compound URI

    """
    assert op.startswith('/'),"Operator must start with '/'" #ensuring the operator starts with a slash
    for arg in args:
        assert ' ' not in arg, "%r is not in normalised" % arg #ensuring the argument is in a normalised form
        items = [op, '['] + [f"{',' if i else ''}{arg}" for i,arg in enumerate(args)] + [']'] #using list comprehension to
        # build an items list
        return join_uri(items)
def split_uri(uri):
    """Takes a URI and splits the string into smaller parts without the slash.
    Inputs:
    uri (string): a string of URI components
    Returns:
    list: the list of parts without the slash
    """
    uri = uri.lstrip('/') #stripping leading slash if present
    if not uri:
        return[] #return an empty if URI is empty
    return uri.split('/') #splitting the URI on slashes and returning the result

def is_absolute_uri(uri):
    """Takes a URI and check if it is an absolute URI (containing all the necessary information to locate a resource on the Internet).
    Inputs:
    uri (string): a string of the URI

    Returns:
        bool: True if the URI is absolute, false otherwise
    """
    return uri.startswith('http') or uri.startswith('cc:')
def uri_prefix(uri,max_pieces=3):
    """ Take the URI and strip off components that might make it too detailed.
    Inputs:
    uri (string): a string of the URI
    max_pieces (string): the number of maximum pieces to keep
    Returns:
    string: the prefix of the URI
    """
    if is_absolute_uri(uri):
        return uri
    pieces = split_uri[:max_pieces]
    return join_uri(*pieces)

