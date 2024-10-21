from PIL import Image
from config import clipprocessor, clipmodel
import torch

def getImageArray(image_path):
    image = Image.open(image_path)
    return image

def get_clip_embeddings(image_path):
    image = clipprocessor(images=Image.open(image_path), return_tensors="pt")
    image_features = clipmodel.get_image_features(**image)
    return torch.stack([image_features])