# ECAPA-TDNN Speaker Recognition Module

## Project Description

This project implements speaker recognition using the pretrained ECAPA-TDNN model from SpeechBrain.

The system extracts speaker embeddings from audio and compares them using cosine similarity for speaker verification and recognition.

---

## Project Structure

```
ECAPA_Module/
│
├── database/
│   ├── Ram.pt
│   └── Sita.pt
│
├── input/
│   ├── person1.wav
│   ├── person2.wav
│   └── sample.wav
│
├── output/
│   └── speaker_embedding.pt
│
├── ecapa.py
├── ecapa_compare.py
├── register.py
├── recognize.py
├── check_embedding.py
└── requirements.txt
```

---

## Requirements

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Register a Speaker

```bash
python register.py
```

---

### 2. Recognize a Speaker

```bash
python recognize.py
```

---

### 3. Compare Two Speakers

```bash
python ecapa_compare.py
```

---

### 4. Check the Saved Embedding

```bash
python check_embedding.py
```

---

## Workflow

```
Input Audio (.wav)
        │
        ▼
ECAPA-TDNN
        │
        ▼
Speaker Embedding (.pt)
        │
        ▼
Database
        │
        ▼
Speaker Recognition
```

---

## Technologies Used

- Python
- PyTorch
- TorchAudio
- SpeechBrain
- ECAPA-TDNN

---

## Dataset

- Pretrained ECAPA-TDNN model trained on VoxCeleb1 and VoxCeleb2.
- Tested using sample WAV audio files.
