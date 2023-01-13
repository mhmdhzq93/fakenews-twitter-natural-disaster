import pandas as pd
import spacy
import re
import nltk
import string
import contractions
import warnings
from spacy.lang.char_classes import LIST_PUNCT
from collections import defaultdict
from spellchecker import SpellChecker
from nltk.tokenize import word_tokenize, sent_tokenize, TweetTokenizer, WordPunctTokenizer, RegexpTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

warnings.filterwarnings('ignore')

spell = SpellChecker()
nlp = spacy.load("en_core_web_sm")
lemmatizer = spacy.load("en_core_web_sm", disable = ['parser', 'ner']) # Lemmatization
stemmer = PorterStemmer()
regexp = RegexpTokenizer("[\w']+")

html_entity_dict = {"&amp;": "&",
                    "&lt;" : "<",
                    "&gt;" : ">"}

##### Data cleaning functions #####
# Convert lowercase
def convert_lowercase(text):
    return text.lower()

def remove_extra_whitespaces(text):
    return re.sub("\s\s+" , " ", text.strip())

# Removing punctuations
def remove_punct(text):
    punct_str = string.punctuation
    punct_str = punct_str.replace("'", "") # discarding apostrophe from the string to keep the contractions intact
    table=str.maketrans('','',punct_str)
    return text.translate(table)

# Removing URLs
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

# Removing HTML tags
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

# Remove emojis @ emoticons
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Remove html entities and replace with the right ASCII characters
def remove_html_entity(text):
    for word, initial in html_entity_dict.items():
        text = text.replace(word.lower(), initial)
    return text

# Remove non-ASCII characters
def remove_nonASCII(text):
    return text.encode('ascii',errors='ignore').decode()

# Discardment of non-alphabetic words @ numbers
def remove_non_alpha(text):
    word_list_non_alpha = [word for word in regexp.tokenize(text) if word.isalpha()]
    text_non_alpha = " ".join(word_list_non_alpha)
    return text_non_alpha

def text_cleaning(text: str) -> str:
    cleaned_text = convert_lowercase(text)
    cleaned_text = remove_extra_whitespaces(cleaned_text)
    cleaned_text = remove_punct(cleaned_text)
    cleaned_text = remove_URL(text)
    cleaned_text = remove_html(cleaned_text)
    cleaned_text = remove_emoji(cleaned_text)
    cleaned_text = remove_html_entity(cleaned_text)
    cleaned_text = remove_nonASCII(cleaned_text)
    cleaned_text = remove_non_alpha(cleaned_text) 
    return cleaned_text


##### Data preprocessing functions #####
# Expand contractions
def expand_contractions(text):
    return contractions.fix(text)

# Remove Stopwords
def remove_stopwords(text):
    return " ".join([word for word in regexp.tokenize(text) if word not in stopwords.words("english")])

def correct_spellings(text):
    if text is None:
        return text
    else:
        corrected_text = []
        misspelled_words = spell.unknown(text.split())
        for word in text.split():
            if word in misspelled_words and spell.correction(word) is not None:
                corrected_text.append(spell.correction(word))
            else:
                corrected_text.append(word)
        return " ".join(corrected_text)

def text_lemmatizer(text):
    text_spacy = " ".join([token.lemma_ for token in lemmatizer(text)])
    return text_spacy

def text_preprocessing(text: str) -> str:
    processed_text = expand_contractions(text)
    # processed_text = correct_spellings(processed_text)
    processed_text = remove_stopwords(processed_text)
    processed_text = text_lemmatizer(processed_text)
    return processed_text


