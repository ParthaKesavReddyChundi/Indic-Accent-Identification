# Native Language Identification using HuBERT

## 📌 Project Overview
This project focuses on **Native Language Identification (NLI)** for Indian English speakers. By analyzing speech patterns, the system classifies speakers into regional backgrounds (Kerala, Tamil Nadu, Andhra Pradesh, etc.). 

The study performs a comparative analysis between **Traditional Acoustic Features (MFCC)** and **Self-Supervised Representations (HuBERT)**. The final model achieves state-of-the-art performance on sentence-level speech and includes a real-time GUI for accent-aware cuisine recommendations.

## 🚀 Key Features
- **Layer-wise Analysis:** Proved that HuBERT Layer 9 (paralinguistic) outperforms Layer 12 (semantic) for accent detection.
- **Linguistic Segmentation:** Comparative study of isolated words vs. full sentences.
- **GUI Application:** A `customtkinter` based desktop app for live accent detection and food recommendation.

## 📊 Results

| Model | Feature Type | Accuracy | Observation |
| :--- | :--- | :--- | :--- |
| **Baseline** | MFCC (13 coeffs) | 24.70% | Limited by lack of temporal context. |
| **HuBERT (L12)**| Transformer Embeddings | 11.48% | Overfitted to semantic content. |
| **HuBERT (L9)** | Transformer Embeddings | **25.53%** | **Optimal layer for accent cues.** |

### Linguistic Level Analysis (Using Best Model)
- **Isolated Words (<1.5s):** 91.84% Accuracy
- **Full Sentences (>3.0s):** **99.09% Accuracy** (Demonstrates the importance of prosody).

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Indic-Accent-Identification.git](https://github.com/YOUR_USERNAME/Indic-Accent-Identification.git)
   cd Indic-Accent-Identification


2. Install dependencies:
   pip install -r requirements.txt

3.(Optional) Download the dataset: This project uses the IndicAccentDb from HuggingFace

💻 Usage

1. Feature Extraction
Run the extraction scripts to process audio into embeddings:
python src/extract_layer9.py

2.Training
Train the Linear Probe classifier:
python src/train_linear_probe.py

3. Run the GUI Demo
Launch the interactive cuisine recommender:
python src/app_gui.py

🧠 Technologies
Deep Learning: PyTorch, HuggingFace Transformers (HuBERT Base)

Audio Processing: Librosa, SoundDevice

Interface: CustomTkinter

Project developed as part of academic research on Speech Processing.
