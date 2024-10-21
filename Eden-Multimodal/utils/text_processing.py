import nltk
from config import tokenizer
import torch
from utils.image_processing import get_clip_embeddings

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
    input_words = nltk.word_tokenize("<|system|> \n You are an assistant good at understanding the context. <|end|> \n <|user|> \n") + nltk.word_tokenize("\n Describe the objects and their relationship in the given context.<|end|> \n <|assistant|> \n")
    output_words = nltk.word_tokenize(output)
    filtered_words = [word for word in output_words if word.lower() not in [w.lower() for w in input_words]]
    return ' '.join(filtered_words)

def getInputs(image_path, question, answer=""):
    image_features = None
    num_image_tokens = 0

    if image_path is not None:
        image_features = get_clip_embeddings(image_path)
        num_image_tokens = image_features.shape[1]

    start_text = f"<|system|>\nYou are an assistant good at understanding the context.<|end|>\n<|user|>\n "
    end_text = f" .\n  Describe the objects and their relationship from the context. <|end|>\n<|assistant|>\n {answer}"

    start_tokens = tokenizer(start_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    end_tokens = tokenizer(end_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    
    start_input_ids = start_tokens['input_ids']
    start_attention_mask = start_tokens['attention_mask']
    end_input_ids = end_tokens['input_ids']
    end_attention_mask = end_tokens['attention_mask']

    if image_path is not None:
        attention_mask = torch.cat([start_attention_mask, torch.ones((1, num_image_tokens), dtype=torch.long), end_attention_mask], dim=1)
    else:
        attention_mask = torch.cat([start_attention_mask, end_attention_mask], dim=1)

    return start_input_ids, end_input_ids, image_features, attention_mask