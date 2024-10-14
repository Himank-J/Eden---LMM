import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_location = "./MM_FT_C1_V2"
model_name = "microsoft/Phi-3.5-mini-instruct"