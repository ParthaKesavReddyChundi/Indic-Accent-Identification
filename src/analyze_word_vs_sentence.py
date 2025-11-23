import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import librosa
import os
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_FILE = "hubert_features_layer9.pkl"
AUDIO_ROOT = "E:/Speech_Data_Cache"

print("--- Phase 4: Word vs Sentence Analysis ---")
df = pd.read_pickle(DATA_FILE)

# 1. ENCODE LABELS FIRST (Moved to top to fix KeyError)
print("Encoding labels...")
label_encoder = LabelEncoder()
df['label_id'] = label_encoder.fit_transform(df['label'])
num_classes = len(label_encoder.classes_)

# 2. Add Duration Column
# (If you already ran this once, the duration might be cached in memory, 
# but we'll recalculate to be safe and robust)
print("Calculating audio durations...")
durations = []
# We use a try-except block just in case a file is missing
for filename, folder in tqdm(zip(df['filename'], df['label']), total=len(df)):
    full_path = os.path.join(AUDIO_ROOT, folder, filename)
    try:
        # get_duration is fast
        d = librosa.get_duration(path=full_path)
        durations.append(d)
    except:
        durations.append(0) 

df['duration'] = durations

# 3. Split into Subsets
# Words = Very short (< 1.5s)
# Sentences = Long (> 3.0s) to ensure clear separation
word_df = df[df['duration'] < 1.5].copy() 
sentence_df = df[df['duration'] > 3.0].copy()

print(f"\nFound {len(word_df)} Isolated Word samples")
print(f"Found {len(sentence_df)} Full Sentence samples")

# 4. Prepare for Training
# We train on EVERYONE to build a strong "General Accent Model"
# Then we test specifically on the short vs long files.
scaler = StandardScaler()
X_all = np.stack(df['features'].values)
X_all_scaled = scaler.fit_transform(X_all)

# Convert all to tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train = torch.tensor(X_all_scaled, dtype=torch.float32).to(device)
y_train = torch.tensor(df['label_id'].values, dtype=torch.long).to(device)

# Define Linear Probe Model
class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearProbe, self).__init__()
        self.layer = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.layer(x)

model = LinearProbe(768, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

print("\nTraining generic model for analysis...")
for epoch in range(40): # Train for 40 epochs
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X_train), y_train)
    loss.backward()
    optimizer.step()

# 5. Evaluate Specific Subsets
model.eval()

def evaluate_subset(subset_df, name):
    if len(subset_df) == 0:
        print(f"{name}: No samples found.")
        return
    
    # Transform features using the SAME scaler as training
    features_raw = np.stack(subset_df['features'].values)
    features_scaled = scaler.transform(features_raw)
    
    X_sub = torch.tensor(features_scaled, dtype=torch.float32).to(device)
    y_sub = torch.tensor(subset_df['label_id'].values, dtype=torch.long).to(device)
    
    with torch.no_grad():
        preds = torch.argmax(model(X_sub), dim=1)
        acc = (preds == y_sub).sum().item() / len(y_sub)
    
    print(f"Accuracy on {name}: {acc*100:.2f}%")

print("\n--- RESULTS: Linguistic Level Analysis ---")
evaluate_subset(word_df, "ISOLATED WORDS (<1.5s)")
evaluate_subset(sentence_df, "FULL SENTENCES (>3.0s)")
print("------------------------------------------")