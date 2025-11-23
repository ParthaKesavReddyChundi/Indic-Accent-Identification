import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm # For a progress bar

# --- CONFIGURATION ---
DATA_PATH = "E:/Speech_Data_Cache"  # Where your language folders are
OUTPUT_FILE = "mfcc_features.pkl"   # Where we save the processed data
SAMPLE_RATE = 16000
DURATION = 3 # Max duration to analyze in seconds (truncates or pads)
N_MFCC = 13  # Standard number of coefficients for speech

print(f"--- Starting MFCC Extraction from {DATA_PATH} ---")

features = []

# 1. Traverse the directory
audio_files = []
for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith('.wav'):
            # We assume folder structure is: .../Language/filename.wav
            # Example: .../kerala/Kerala_speaker_01_1.wav
            file_path = os.path.join(root, file)
            folder_name = os.path.basename(root) 
            audio_files.append((file_path, folder_name, file))

print(f"Found {len(audio_files)} audio files. Processing now...")

# 2. Extraction Loop
for path, label, filename in tqdm(audio_files):
    try:
        # Load audio (automatically resamples to 16kHz)
        signal, sr = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Fix length (Pad if short, Cut if long) - ensures uniform input shape
        target_len = SAMPLE_RATE * DURATION
        if len(signal) < target_len:
            # Pad with zeros
            padding = target_len - len(signal)
            signal = np.pad(signal, (0, padding), 'constant')
        else:
            # Truncate
            signal = signal[:target_len]

        # Extract MFCCs
        # This creates a matrix: (n_mfcc, time_steps)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)
        
        # Take the MEAN across time (simple aggregation for baseline model)
        # This turns the matrix into a single vector of 13 numbers per clip
        mfcc_mean = np.mean(mfcc.T, axis=0)
        
        # Parse Metadata from filename (Heuristic based on your finding)
        # Filename: Kerala_speaker_01_1.wav
        # We try to extract the Speaker ID (01)
        parts = filename.replace('.wav', '').split('_')
        speaker_id = "unknown"
        if "speaker" in parts:
            idx = parts.index("speaker")
            if idx + 1 < len(parts):
                speaker_id = parts[idx+1]

        features.append({
            "features": mfcc_mean,
            "label": label,        # Native Language (e.g., kerala -> Malayalam)
            "speaker_id": speaker_id,
            "filename": filename
        })
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# 3. Save Data
if features:
    df = pd.DataFrame(features)
    print(f"\nSuccessfully extracted features for {len(df)} files.")
    print("Preview of Data:")
    print(df.head())
    
    # Save to disk
    df.to_pickle(OUTPUT_FILE)
    print(f"\nSaved features to {OUTPUT_FILE}")
    print("You can now use this file to train your Classifier!")
else:
    print("No features extracted. Check your paths.")