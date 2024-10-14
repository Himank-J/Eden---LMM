import nltk
import re

nltk.download('punkt')
nltk.download('punkt_tab')

def remove_punctuation(text):
    newtext = ''.join([char for char in text if char.isalnum() or char.isspace()])
    newtext = ' '.join(newtext.split())
    return newtext

def preprocess_text(text):
    text_no_punct = remove_punctuation(text)
    return text_no_punct

def getStringAfter(output, start_str):
    if start_str in output:
        answer = output.split(start_str)[1]
    else:
        answer = output

    answer = preprocess_text(answer)
    return answer

def getAnswerPart(output):
    output_words = nltk.word_tokenize(output)
    filtered_words = [word for word in output_words if word.lower() not in [w.lower() for w in input_words]]
    return ' '.join(filtered_words)

start_text = "<|system|> \n You are an assistant good at understanding the context.<|end|> \n <|user|> \n"
end_text = "\n Describe the objects and their relationship in the given context.<|end|> \n <|assistant|> \n"

words = nltk.word_tokenize(start_text) + nltk.word_tokenize(end_text)
input_words = list(set(words))