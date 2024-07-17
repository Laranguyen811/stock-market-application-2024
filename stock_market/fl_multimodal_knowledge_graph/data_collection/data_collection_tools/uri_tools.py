import uritools as ut
from urllib.parse import urljoin,urlparse
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
import rdflib
from collections import Counter
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
def uri_prefixes(uri,min_pieces=2):
    """Take the URI and get the URI that are the prefixes of the URI (prefix must have at least 2 components).
    Input:
    uri (string): the string of the URI
    max_pieces (string): the number of maximum pieces
    Returns:
    string: the string of the prefix of the URI
    """
    if is_absolute_uri(uri):
        return [uri]
    pieces = []
    for piece in split_uri(uri):
        counts = Counter(pieces) #counting hashable objects, in this case constiuent pieces
        if len(pieces) >= min_pieces and counts['['] == counts[']']:
            yield join_uri(*pieces) #using yield to create a generator function and produce prefixes on-the-fly, more memory efficient especially when having many prefixes

def parse_compound_uri(uri):
    """ Take the compound URI and extract its operator and list of arguments.
    Inputs:
    URI (string): a string of the compound URI. A compound URI is a URI containing an operator and a list of arguments.
    For example, '/or/[/and/[/s/one/,/s/two/]/,/and/[/s/three/,/s/four/]/]'.
    Returns:
    string: the string of the operator. In the example above, the operator would be '/or'.
    list: the list of arguments. In the example above, the arguments would be '/and/[/s/one/,/s/two/]', '/and/[/s/three/,/s/four/]'].
    """
    pieces = split_uri(uri)
    if pieces[-1] != ']':
        raise ValueError("Compound URIs must end with /]/")
    if '[' not in pieces:
        raise ValueError("Compound URIs must contain /[/ at the beginning of each argument list")
    start_list = pieces.index('[')
    op = join_uri(*pieces[:start_list])

    chunks = []
    current = []
    depth = 0

    #Split on commas unless they are within additional pair of brackets
    for piece in pieces[(start_list+1) : -1]:
        current.append(piece)
        counts = Counter(current)
        if piece == ',' and counts['['] == counts[']']:
            chunks.append('/' + ('/'.join(current[:-1])).strip('/'))
            current = []
        elif piece =='[':
            depth += 1
        elif piece == ']':
            depth -= 1

    assert depth == 0, "Unmatched brackets in %r" % uri
    if current:
        chunks.append('/' + ('/'.join(current)).strip('/'))
    return op,chunks

def parse_possible_compound_uri(op,uri):
    """ Takes a compound URI and returns a list of components in the compound URI if its operator matches 'op' or
    a list containing the URI itself if not.
    Inputs:
    uri(string): the compound URI. For example, it could be /or/[/and/[/s/one/,/s/two/]/
    Returns:
    list: the list of components in the compound URI. In the example above, the list would be: ['/and/[/s/one/,/s/two/]
    """
    if uri.startswith('/' + op + '/'):
        return parse_compound_uri(uri)[1]
    else:
        return [uri]

def conjunction_uri(*sources):
    """ Takes sources and return a URI representing the conjunction of the sources (combining multiple sources to provide an assertion.
    Inputs:
    sources(list): The list of sources to be conjoined. An example is '/s/rule/some_kind_of_parser', '/s/contributor/omcs/dev'
    Returns:
    string: A URI representing the conjunction of the sources. In the example above, a URI would be /and/[/s/contributor/omcs/dev/,/s/rule/some_kind_of_parser/
    """
    if len(sources) == 0:
        raise ValueError('Conjunctions of 0 things are not allowed')
    if len(sources) == 1:
        return sources[0]
    return compound_uri('/and',sorted(set(sources)))

def assertion_uri(rel,start,end):
    """ Takes a URI and returns an assertion (a weighted edge - an edge assigned a numerical value) with its relation, start node and end node as a compound URI.
    Input:
    rel(string): The relation of the URI being asserted. A relation example is /r/CapableOf.
    start(string): The start node of the URI. An example is /c/en/cat
    end(string): The end node of the URI. An example is /c/en/sleep
    Returns:
    string: The assertion URI. In the example above, the assertion URI is /a/[/r/CapableOf/,/c/en/cat/,/c/en/sleep/]
    """
    if not rel.startswith('/r'):
        raise ValueError(f"Invalid relation:{rel}.Relation must start with '/r'.")
    if not start.startswith('/c'):
        raise ValueError(f"Invalid start node:{start}.Start must start with '/c'.")
    if not end.startswith('/c'):
        raise ValueError(f"Invalid end node:{end}.Start must start with '/c'.")
    return compound_uri('/a',(rel,start,end))

def is_concept(uri):
    """ Takes a URI and returns a boolean indicating whether it is a concept or not.
    Inputs:
    uri(string): The URI to check. An example is '/c/sv/klänning
    Returns:
    bool: True if the URI is a concept or False if it is not. In the example above, the returned value is True.
    """
    return uri.startswith('/c/')

def is_relation(uri):
    """ Takes a URI and returns a boolean indicating whether it is a relation or not.
    Inputs:
    uri(string): The URI to check. An example is '/r/IsA'
    Returns:
    bool: True if the URI is a relation or False if it is not. In the example above, the returned value is True.
    """
    return uri.startswith('/r/')

def is_term(uri):
    """ Takes a URI and returns a boolean indicating whether it is a term (a word or phrase) or not.
    Inputs:
    uri(string): The URI to check. An example is/c/sv/kostym.
    Returns:
    bool: True if the URI is a term or False if it is not. In the example above, the returned value is True.
    """
    return uri.startswith('/c/') or uri.startswith('/x/')

def get_uri_language(uri):
    """ Takes a URI and returns its language. If the URI points to an assertion, returns its first concept.
    Inputs:
    uri(string): The string of the URI.An example is /a/[/r/RelatedTo/,/c/en/orchestra/,/c/en/symphony/]
    Returns:
    string: The language of the URI. In the above example, the language would be 'en'
    string: The first concept of the URI. In the above example, there would be no returned value.
    """
    if uri.startswith('/a/'):
        return get_uri_language(parse_possible_compound_uri('a',uri)[1])
    elif is_term(uri):
        return split_uri(uri)[1]
    else:
        return None

def uri_to_label(uri):
    """ Takes a URI and returns a label associated with the URI so that we can use it in its node.
    Inputs:
    uri(string): The string of the URI.An example is /c/en/example.
    Returns:
    string: The label of the URI to be used in a node. In the above example, the returned label would be an example.
    """
    if is_absolute_uri(uri):
        return uri.split('/')[-1].replace('_',' ') #splitting at the slash, take the component before it, removing the slash and replace it with space
    if is_term(uri):
        uri = uri_prefix(uri)
    parts = split_uri(uri)
    if len(parts) < 3 and not is_relation(uri):
        return ''
    return parts[-1].replace('_',' ')

class Licenses:
    cc_attribution = 'cc:by/4.0'
    cc_sharealike = 'cc:by-sa/4.0'











