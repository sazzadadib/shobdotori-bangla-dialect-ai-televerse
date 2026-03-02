<div align="center">

# 🎙️ SHOBDOTORI — শব্দতরী
### Bridging Dialects to Standard Bangla through Speech Recognition

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper%20Medium-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/research/whisper)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **Team BITWISEMIND** · *Televerse AI Hackathon Competition Entry*

*An Automatic Speech Recognition (ASR) system that transcribes **20 Bangladeshi regional dialects** into standard Bangla using a fine-tuned Whisper Medium model.*

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Repository Structure](#-repository-structure)
- [Training Pipeline](#-training-pipeline)
- [Inference Pipeline](#-inference-pipeline)
- [Technical Configuration](#-technical-configuration)
- [Requirements](#-requirements)
- [Getting Started](#-getting-started)
- [Team](#-team)

---

## 🌟 Overview

**SHOBDOTORI (শব্দতরী)** — meaning *"vessel of words"* in Bengali — is a regional dialect-aware Automatic Speech Recognition (ASR) system built for the **Televerse AI Hackathon**. Bangladesh is home to a rich tapestry of regional dialects that differ significantly in phonology, vocabulary, and prosody from standard Bangla. Existing ASR systems struggle with these variations, leaving millions of dialect speakers underserved.

This project tackles that gap by fine-tuning **OpenAI's Whisper Medium** model on a curated dataset of **20 distinct Bangladeshi regional dialects**, enabling robust transcription of dialectal speech into standard written Bangla.

### 🎯 Problem Statement

Regional dialect speakers face significant barriers when using mainstream Bangla speech recognition technology. SHOBDOTORI bridges this divide by:
- Recognizing phonological variations across 20 Bangladeshi districts
- Transcribing regional speech into standardized Bangla text
- Preserving linguistic accessibility for dialect-speaking communities

---

## 📊 Key Results

| Training Step | Training Loss | Validation Loss | WER (%) |
|:---:|:---:|:---:|:---:|
| 1,000 | 0.0081 | 0.0642 | 10.96 |
| 2,000 | 0.0013 | 0.0551 | 7.11 |
| 3,000 | 0.0009 | 0.0577 | 7.88 |
| **4,000** | **0.0004** | **0.0500** | **6.17** ✅ |

- **Best WER: 6.17%** — achieved at step 4,000 (lower is better)
- Total training time: ~8.5 hours on **NVIDIA Tesla T4 (14.74 GB)**
- Training samples: **3,015** · Evaluation samples: **335**
- Effective batch size: **16** (4 per device × 4 gradient accumulation steps)

---

## 🏗️ Architecture

```
Audio Input (16kHz WAV)
        │
        ▼
┌─────────────────────────────┐
│  Whisper Feature Extractor  │   Log-Mel Spectrogram (80 bins)
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│     Whisper Medium Encoder  │   ~307M parameters
│  (bengaliAI fine-tuned base)│   Transformer encoder layers
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│     Whisper Decoder         │   Autoregressive token generation
│  (Bengali Language Head)    │   task = "transcribe"
└─────────────────────────────┘
        │
        ▼
Standard Bangla Transcription
```

### Base Model

We built upon [`bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium`](https://huggingface.co/bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium) — itself a fine-tune of [`openai/whisper-medium`](https://huggingface.co/openai/whisper-medium) trained on the [BEN10 Kaggle Dataset](https://www.kaggle.com/competitions/ben10). We further fine-tuned this model on our custom 20-dialect regional dataset.

---

## 🗂️ Dataset

### Regional Coverage — 20 Bangladeshi Districts

| Region Group | Districts |
|---|---|
| **Dhaka Division** | Dhaka, Mymensingh |
| **Chittagong Division** | Chittagong, Comilla, Feni, Noakhali, Lakshmipur, Brahmanbaria |
| **Khulna Division** | Khulna, Jessore, Jhenaidah, Kushtia |
| **Rajshahi Division** | Rajshahi, Bogura, Natore, Pabna |
| **Barisal Division** | Barisal, Bhola |
| **Sylhet Division** | Sylhet |
| **Rangpur Division** | Rangpur |

### Dataset Statistics

| Split | Samples | Source |
|---|---|---|
| Train | 3,015 (90%) | `bitwisemind/hackathon` (HuggingFace) |
| Test | 335 (10%) | `bitwisemind/hackathon` (HuggingFace) |
| **Total** | **3,350** | Audio + CSV annotations |

- **Audio format**: WAV files, resampled to 16 kHz
- **Annotation format**: CSV files per district (filename, transcription)
- **Features**: Log-Mel spectrograms (80 bins), tokenized Bengali text labels

---

## 📁 Repository Structure

```
shobdotori-bangla-dialect-ai-televerse/
│
├── 📓 BITWISEMIND_Regional_WhisperMedium_FineTuning_Training.ipynb
│       └── Full fine-tuning pipeline: data loading → preprocessing →
│           model setup → training → evaluation → model saving
│
├── 📓 BITWISEMIND_Regional_WhisperMedium_FineTuning_Inference.ipynb
│       └── Inference pipeline: load fine-tuned model → transcribe audio
│
├── 📄 BITWISEMIND_DOCUMENT.pdf
│       └── Competition project documentation and technical writeup
│
├── 📋 requirements.txt
│       └── Python package dependencies
│
└── 📖 README.md
```

---

## 🔄 Training Pipeline

The training notebook is structured into well-defined stages:

```
1. 🔧  Environment Setup        — Install dependencies, verify GPU
2. 📥  Data Preparation         — Download dataset from HuggingFace Hub
3. 📦  Extract & Configure      — Unzip archives, define 20 dialect regions
4. 📚  Import Libraries         — Load transformers, datasets, evaluate
5. 📊  Dataset Loading          — Custom loader across all 20 regional CSVs
6. ⚙️  Processor Setup          — WhisperFeatureExtractor + WhisperTokenizer
7. 🔀  Data Preprocessing       — Convert audio → Log-Mel + tokenize labels
8. 🧩  Data Collator            — Pad sequences, mask padding with -100
9. 🤖  Model Loading            — Load Whisper Medium (~3.06 GB weights)
10. 📏 Evaluation Metric        — Word Error Rate (WER) via evaluate/jiwer
11. ⚙️  Training Configuration  — Seq2SeqTrainingArguments (4000 steps)
12. 🚀  Fine-Tuning             — Seq2SeqTrainer with custom callbacks
13. 💾  Model Saving            — Save weights + processor to disk
14. 📈  Results Summary         — Final WER and performance report
```

---

## 🔍 Inference Pipeline

Load the saved fine-tuned model and transcribe new audio:

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import librosa

# Load fine-tuned model & processor
model_path = "./whisper-medium-bengali-regional-final"
processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path)
model.eval()

# Load and preprocess audio
audio, sr = librosa.load("your_audio.wav", sr=16000)

# Extract features
input_features = processor(
    audio,
    sampling_rate=16000,
    return_tensors="pt"
).input_features

# Generate transcription
with torch.no_grad():
    predicted_ids = model.generate(input_features)

transcription = processor.batch_decode(
    predicted_ids, skip_special_tokens=True
)[0]

print(f"Transcription: {transcription}")
```

> ⚡ See `BITWISEMIND_Regional_WhisperMedium_FineTuning_Inference.ipynb` for the complete inference notebook.

---

## ⚙️ Technical Configuration

### Training Hyperparameters

| Parameter | Value |
|---|---|
| Base Model | `bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium` |
| Architecture | Whisper Medium (~300M params) |
| Language | Bengali (`bn`) |
| Task | Transcription |
| Batch size (per device) | 4 |
| Gradient accumulation steps | 4 |
| **Effective batch size** | **16** |
| Learning rate | `1e-5` |
| Warmup steps | 200 |
| Max training steps | 4,000 |
| Optimizer | AdamW (`adamw_torch`) |
| Max gradient norm | 1.0 |
| Precision | FP16 mixed precision |
| Gradient checkpointing | ✅ Enabled |
| Evaluation strategy | Every 1,000 steps |
| Best model metric | WER (lower = better) |
| Monitoring | TensorBoard |

### Hardware

| Component | Specification |
|---|---|
| GPU | NVIDIA Tesla T4 (14.74 GB VRAM) |
| CUDA | 12.6 |
| PyTorch | 2.6.0+cu124 |
| Driver | 560.35.03 |

---

## 📦 Requirements

```txt
torch==2.6.0
transformers==4.45.0
datasets==2.16.1
accelerate==1.0.0
evaluate==0.4.3
jiwer==3.0.3
librosa==0.10.0
soundfile==0.12.1
tensorboard==2.15.1
huggingface-hub==0.20.3
numpy==1.26.4
pandas==2.1.4
scipy==1.11.4
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/sazzadadib/shobdotori-bangla-dialect-ai-televerse.git
cd shobdotori-bangla-dialect-ai-televerse
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Training

Open and run the training notebook on a GPU environment (Kaggle, Colab, or local):

```
BITWISEMIND_Regional_WhisperMedium_FineTuning_Training.ipynb
```

> 💡 **Recommended**: Use Kaggle with a T4 GPU accelerator. The training notebook will automatically download the dataset from HuggingFace Hub (`bitwisemind/hackathon`).

### 4. Run Inference

```
BITWISEMIND_Regional_WhisperMedium_FineTuning_Inference.ipynb
```

Load your fine-tuned model and transcribe dialectal Bangla audio files.

---

## 👥 Team

<div align="center">

**Team BITWISEMIND**
*Televerse AI Hackathon · 2025*

</div>

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) — Base ASR architecture
- [bengaliAI](https://huggingface.co/bengaliAI) — Regional Whisper checkpoint used as our starting point
- [Hugging Face](https://huggingface.co/) — Model hub, `transformers`, `datasets`, and `evaluate` libraries
- [Kaggle](https://www.kaggle.com/) — GPU compute environment for training
- [BEN10 Competition](https://www.kaggle.com/competitions/ben10) — Inspired dataset structure

---

<div align="center">

*Made with ❤️ for the Bengali-speaking community*

**শব্দতরী** — *Carrying voices across dialects, one word at a time.*

</div>
