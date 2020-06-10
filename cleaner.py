
import nltk
import string
import contractions
from bs4 import BeautifulSoup



def normalize_document(text):
    stop_words = set( nltk.corpus.stopwords.words('english')+ list(string.punctuation)+["...","*","''","``"])
    text_without_html = BeautifulSoup(text).get_text()
    words = text_without_html.split() 
    words_without_contractions = [contractions.fix(word) for word in words]
    #Join wordlist again to use word tokenize so words can be separated properly without losing meaning
    text_complete = ' '.join(words_without_contractions)
    words_nltk = nltk.word_tokenize(text_complete)
    clean_words = [word.lower() for word in words_nltk if word.lower() not in stop_words]
    clean_text = " ".join(clean_words)
    return clean_text
