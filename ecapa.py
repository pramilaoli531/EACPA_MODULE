from speechbrain.pretrained import EncoderClassifier
import torch
import torchaudio

print("Loading ECAPA-TDNN model...")

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)

print("Model loaded successfully!")

# Read the audio file and extract the speaker embedding
signal, fs = torchaudio.load("sample.wav")

embedding = classifier.encode_batch(signal)

print("Embedding Shape:", embedding.shape)

# Save the embedding
torch.save(embedding, "speaker_embedding.pt")

print("Speaker embedding saved successfully!")