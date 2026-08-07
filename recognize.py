from speechbrain.pretrained import EncoderClassifier
import soundfile as sf
import torch
import torch.nn.functional as F
import os

print("Loading ECAPA Model...")

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)

print("Model Loaded!")

# Ask for test audio
audio_file = input("Enter WAV filename: ")

# Read audio
signal, fs = sf.read(audio_file)

signal = torch.tensor(signal, dtype=torch.float32)

# Convert stereo to mono
if signal.ndim == 2:
    signal = signal.mean(dim=1)

# Add batch dimension
signal = signal.unsqueeze(0)

# Extract embedding
test_embedding = classifier.encode_batch(signal)

best_score = -1
best_speaker = None

# Compare with every registered speaker
for file in os.listdir("database"):

    if file.endswith(".pt"):

        saved_embedding = torch.load(os.path.join("database", file))

        score = F.cosine_similarity(
            test_embedding.squeeze(),
            saved_embedding.squeeze(),
            dim=0
        )

        print(f"{file[:-3]} Similarity: {score.item():.4f}")

        if score > best_score:
            best_score = score
            best_speaker = file[:-3]

print("\n==============================")
print("Recognized Speaker:", best_speaker)
print("Highest Similarity:", best_score.item())
print("==============================")