import torch

embedding = torch.load("speaker_embedding.pt")

print("Type:", type(embedding))
print("Shape:", embedding.shape)
print("Embedding:")
print(embedding)