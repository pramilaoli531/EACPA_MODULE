from speechbrain.pretrained import EncoderClassifier
import soundfile as sf
import torch
import os

print("Loading ECAPA Model...")

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)

print("Model Loaded!")

# Ask the user
speaker_name = input("Enter Speaker Name: ")

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
embedding = classifier.encode_batch(signal)

# Create folder if it doesn't exist
os.makedirs("database", exist_ok=True)

# Save embedding
save_path = f"database/{speaker_name}.pt"

torch.save(embedding, save_path)

print("Speaker registered successfully!")
print("Saved to:", save_path)