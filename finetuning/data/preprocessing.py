import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = ' '.join([word for word in word_tokens if word.lower() not in stop_words])
    return ' '.join(filtered_text.split())

def remove_punctuation(text):
    return ''.join([char for char in text if char.isalnum() or char.isspace()])

def preprocess_text(text):
    text_no_punct = remove_punctuation(text)
    text_no_stopwords = remove_stopwords(text_no_punct)
    return text_no_stopwords