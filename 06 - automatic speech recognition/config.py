import torch

test = False

device = "cuda" if torch.cuda.is_available() else "cpu"
