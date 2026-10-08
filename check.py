import torch
import transformers
import fastapi

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("fastapi:", fastapi.__version__)
print("GPU available", torch.cuda.is_available())