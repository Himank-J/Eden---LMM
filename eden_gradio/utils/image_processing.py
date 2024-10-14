from PIL import Image
from transformers import CLIPProcessor, CLIPModel

clipmodel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clipprocessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def getImageArray(image_path):
    image = Image.open(image_path)
    return image

def get_clip_embeddings(image):
    processed_image = clipprocessor(images=image, return_tensors="pt")
    image_features = clipmodel.get_image_features(**processed_image)
    return image_features