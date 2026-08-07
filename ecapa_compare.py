from speechbrain.pretrained import EncoderClassifier
import torch
import torchaudio
import torch.nn.functional as F

print("Loading model...")

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)

print("Model loaded!")

import soundfile as sf
import torch

signal1, fs1 = sf.read("input/person1.wav")
signal2, fs2 = sf.read("input/person2.wav")

signal1 = torch.tensor(signal1, dtype=torch.float32)
signal2 = torch.tensor(signal2, dtype=torch.float32)

# Convert stereo to mono
if signal1.ndim == 2:
    signal1 = signal1.mean(dim=1)
if signal2.ndim == 2:
    signal2 = signal2.mean(dim=1)

# Add batch dimension
signal1 = signal1.unsqueeze(0)
signal2 = signal2.unsqueeze(0)

embedding1 = classifier.encode_batch(signal1)

torch.save(embedding1, "output/speaker_embedding.pt")
print("Speaker embedding saved successfully!")

embedding2 = classifier.encode_batch(signal2)

similarity = F.cosine_similarity(
    embedding1.squeeze(),
    embedding2.squeeze(),
    dim=0
)

print("Similarity:", similarity.item())

if similarity > 0.75:
    print("Same Speaker")
else:
    print("Different Speaker")