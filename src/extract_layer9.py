import os
import torch
import librosa
import numpy as np
import pandas as pd
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_PATH = "E:/Speech_Data_Cache"
OUTPUT_FILE = "hubert_features_layer9.pkl"  # <--- NEW FILE NAME
MODEL_NAME = "facebook/hubert-base-ls960"
SAMPLE_RATE = 16000
DURATION = 3 
TARGET_LAYER = 9  # <--- CHANGED FROM 12 TO 9 (Middle layers capture accent better)

print(f"--- Extracting HuBERT Layer {TARGET_LAYER} ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = HubertModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

audio_files = []
for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith('.wav'):
            file_path = os.path.join(root, file)
            folder_name = os.path.basename(root)
            audio_files.append((file_path, folder_name, file))

features = []
print(f"Processing {len(audio_files)} files...")

for path, label, filename in tqdm(audio_files):
    try:
        signal, sr = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION)
        target_len = SAMPLE_RATE * DURATION
        if len(signal) < target_len:
            padding = target_len - len(signal)
            signal = np.pad(signal, (0, padding), 'constant')
        else:
            signal = signal[:target_len]

        inputs = feature_extractor(signal, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            outputs = model(input_values, output_hidden_states=True)
        
        # Extract Layer 9
        layer_output = outputs.hidden_states[TARGET_LAYER]
        embedding = torch.mean(layer_output, dim=1).squeeze().cpu().numpy()
        
        parts = filename.replace('.wav', '').split('_')
        speaker_id = "unknown"
        if "speaker" in parts:
            idx = parts.index("speaker")
            if idx + 1 < len(parts):
                speaker_id = parts[idx+1]

        features.append({
            "features": embedding,
            "label": label,
            "speaker_id": speaker_id,
            "filename": filename
        })
        
    except Exception as e:
        print(f"Error on {filename}: {e}")

if features:
    pd.DataFrame(features).to_pickle(OUTPUT_FILE)
    print(f"\nSuccess! Saved Layer {TARGET_LAYER} features.")