import os
import torch
import librosa
import numpy as np
import pandas as pd
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_PATH = "E:/Speech_Data_Cache"
OUTPUT_FILE = "hubert_features_layer12.pkl" 
MODEL_NAME = "facebook/hubert-base-ls960"
SAMPLE_RATE = 16000
DURATION = 3 # Seconds
TARGET_LAYER = 12 # The last layer is usually best for broad linguistic info

print(f"--- Loading HuBERT Model: {MODEL_NAME} ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = HubertModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval() # Set to evaluation mode (freezes weights)

print("--- Model Loaded. Starting Extraction ---")

# 1. Map the Files (Reuse the logic from before)
audio_files = []
for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith('.wav'):
            file_path = os.path.join(root, file)
            folder_name = os.path.basename(root)
            audio_files.append((file_path, folder_name, file))

# 2. Extraction Loop
features = []

# We process one file at a time. 
# (Batching is faster but harder to write. This is safe and stable.)
for path, label, filename in tqdm(audio_files):
    try:
        # Load and resample
        signal, sr = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Padding/Truncating to ensure fixed length
        target_len = SAMPLE_RATE * DURATION
        if len(signal) < target_len:
            padding = target_len - len(signal)
            signal = np.pad(signal, (0, padding), 'constant')
        else:
            signal = signal[:target_len]

        # Prepare input for HuBERT
        # return_tensors="pt" gives us PyTorch tensors
        inputs = feature_extractor(signal, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)

        # Forward Pass (The "Thinking" Step)
        with torch.no_grad():
            outputs = model(input_values, output_hidden_states=True)
        
        # Extract Hidden States
        # outputs.hidden_states is a tuple of 13 tensors (1 embedding + 12 layers)
        # We grab the specific layer we want (Index 12 = Final Layer)
        layer_output = outputs.hidden_states[TARGET_LAYER]
        
        # Average over time (Mean Pooling)
        # Converts shape [1, 149, 768] -> [768]
        # This gives us one vector representing the whole 3-second clip
        embedding = torch.mean(layer_output, dim=1).squeeze().cpu().numpy()
        
        # Extract Speaker ID again for splitting
        parts = filename.replace('.wav', '').split('_')
        speaker_id = "unknown"
        if "speaker" in parts:
            idx = parts.index("speaker")
            if idx + 1 < len(parts):
                speaker_id = parts[idx+1]

        features.append({
            "features": embedding, # This is now a vector of size 768
            "label": label,
            "speaker_id": speaker_id,
            "filename": filename
        })
        
    except Exception as e:
        print(f"Error on {filename}: {e}")

# 3. Save
if features:
    df = pd.DataFrame(features)
    df.to_pickle(OUTPUT_FILE)
    print(f"\nSuccess! Saved {len(df)} HuBERT embeddings to {OUTPUT_FILE}")
else:
    print("Failed to extract any features.")