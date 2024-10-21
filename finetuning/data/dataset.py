import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from .preprocessing import preprocess_text
from config import MAX_LENGTH

class ImageTextDatasetForCausalLM(Dataset):
    def __init__(self, clip_processor, csv_file, image_dir, tokenizer, max_length=MAX_LENGTH):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.clip_processor = clip_processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_name = row['image']
        human_text = row['human']
        gpt_text = row['gpt']

        gpt_text_st = preprocess_text(gpt_text)

        image_name_without_ext = os.path.splitext(image_name)[0]
        image_files = [os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) if f.endswith(f'{image_name_without_ext}.jpg')]
        if not image_files:
            raise FileNotFoundError(f"No image file found for {image_name}")

        image = self.clip_processor(images=Image.open(image_files[0]), return_tensors="pt")

        start_text = "<|system|>\nYou are a helpful assistant good at answering questions based on the given context.<|end|>\n<|user|>\n"
        end_text = f"\n{human_text}<|end|>\n<|assistant|>\n{gpt_text}"
        
        return {
            "image_features": image['pixel_values'],
            "start_text": start_text,
            "end_text": end_text
        }
    
    def shuffle(self, seed=None):
        self.data = self.data.sample(frac=1, random_state=seed).reset_index(drop=True)