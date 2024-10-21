import torch
from PIL import Image
import clip
import os
from tqdm import tqdm
from datasets import load_dataset
import numpy as np

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load Instruct 150k dataset from local JSON file
dataset = load_dataset('json', data_files='llava_instruct_150k.json', split='train')
embeddings = {}
batch_size = 32  

for i in tqdm(range(0, len(dataset), batch_size)):
    batch = dataset[i:i+batch_size]
    images = []
    ids = []
    
    for item in batch:
        image_path = os.path.join('train', item['image'])
        try:
            image = preprocess(Image.open(image_path).convert('RGB'))
            images.append(image)
            ids.append(item['id'])
        except Exception as e:
            print(f"Error processing image {item['image']}: {e}")
    
    if images:
        images = torch.stack(images).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(images)
        
        for id, feature in zip(ids, image_features):
            embeddings[id] = feature.cpu().numpy()

# Save embeddings
torch.save(embeddings, 'clip_embeddings.pt')
