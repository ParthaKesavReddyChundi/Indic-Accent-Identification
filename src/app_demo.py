import torch
import torch.nn as nn
import numpy as np
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import pandas as pd
import os
import time

# --- CONFIGURATION ---
MODEL_PATH = "hubert_features_layer9.pkl" # We need the label encoder from here
SAVED_MODEL_WEIGHTS = "linear_probe_model.pth" # We will save weights first!
DURATION = 3
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. RE-DEFINE THE MODEL STRUCTURE (Must match training exactly) ---
class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearProbe, self).__init__()
        self.layer = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.layer(x)

# --- 2. SETUP FOOD MENU ---
# Customize this menu!
MENU = {
    "kerala": ["Appam with Stew", "Puttu and Kadala Curry", "Kerala Sadya", "Avial"],
    "tamil": ["Chicken Chettinad", "Dosa with Sambar", "Pongal", "Filter Coffee"],
    "andhra_pradesh": ["Hyderabadi Biryani", "Gongura Pachadi", "Pesarattu", "Spicy Chicken Curry"],
    "karnataka": ["Bisi Bele Bath", "Mysore Masala Dosa", "Ragi Mudde", "Mysore Pak"],
    "gujrat": ["Dhokla", "Thepla", "Undhiyu", "Basundi"],
    "jharkhand": ["Litti Chokha", "Dhuska", "Rugra", "Malpua"]
}

# --- 3. HELPER FUNCTIONS ---
def record_audio(filename="live_input.wav", duration=3):
    print(f"\n🎤 Recording for {duration} seconds... SPEAK NOW!")
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()  # Wait until recording is finished
    write(filename, SAMPLE_RATE, (recording * 32767).astype(np.int16))
    print("✅ Recording saved.")
    return filename

def predict_accent(audio_path, model, feature_extractor, hubert, label_encoder):
    # Load and Preprocess
    signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
    
    # Pad if necessary
    target_len = SAMPLE_RATE * DURATION
    if len(signal) < target_len:
        padding = target_len - len(signal)
        signal = np.pad(signal, (0, padding), 'constant')
    else:
        signal = signal[:target_len]
        
    # Extract HuBERT Features (Layer 9)
    inputs = feature_extractor(signal, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(DEVICE)
    
    with torch.no_grad():
        outputs = hubert(input_values, output_hidden_states=True)
        layer_output = outputs.hidden_states[9] # Layer 9
        embedding = torch.mean(layer_output, dim=1).squeeze() # [768]
    
    # Predict with Classifier
    # Note: We need to scale the input if we used StandardScaler in training.
    # For a quick demo, we might skip scaler if we retrain without it, 
    # BUT for best results, we should load the scaler. 
    # (Simplified here: we feed raw features. It might degrade accuracy slightly but works for demo).
    
    embedding = embedding.unsqueeze(0).to(DEVICE) # Add batch dimension [1, 768]
    logits = model(embedding)
    probs = torch.softmax(logits, dim=1)
    pred_idx = torch.argmax(probs).item()
    confidence = probs[0][pred_idx].item()
    
    predicted_label = label_encoder.inverse_transform([pred_idx])[0]
    return predicted_label, confidence

# --- 4. MAIN APPLICATION ---
def main():
    print("--- 🍽️  LOADING ACCENT-AWARE MENU SYSTEM ... ---")
    
    # A. Load Resources
    # We need to recreate the LabelEncoder from your data file
    df = pd.read_pickle(MODEL_PATH) 
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(df['label'])
    num_classes = len(le.classes_)
    
    # B. Load AI Models
    print("Loading Brain (HuBERT)...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    hubert.to(DEVICE)
    hubert.eval()
    
    print("Loading Classifier...")
    model = LinearProbe(768, num_classes).to(DEVICE)
    # NOTE: In a real app, you would load saved weights like:
    # model.load_state_dict(torch.load("my_best_model.pth"))
    # Since we just trained it in memory in the previous step, 
    # we will just initialize a random one for this code structure demonstration
    # UNLESS you want to quickly retrain it here?
    
    # Quick Retrain for the Demo to ensure it works live
    print("⚡ Quick-Training model on your data for the demo...")
    X = np.stack(df['features'].values)
    y = le.transform(df['label'])
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y_tensor = torch.tensor(y, dtype=torch.long).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(30): # 30 epochs is enough
        optimizer.zero_grad()
        loss = criterion(model(X_tensor), y_tensor)
        loss.backward()
        optimizer.step()
    model.eval()
    print("✅ System Ready!")

    # C. Interaction Loop
    while True:
        print("\n" + "="*40)
        print("1. 🎤 Place Order (Record Voice)")
        print("2. 📁 Upload File (Test existing .wav)")
        print("3. ❌ Exit")
        choice = input("Select option: ")
        
        if choice == "1":
            filename = record_audio()
            process_request(filename, model, feature_extractor, hubert, le)
            
        elif choice == "2":
            path = input("Enter full path to .wav file: ").strip('"')
            if os.path.exists(path):
                process_request(path, model, feature_extractor, hubert, le)
            else:
                print("File not found!")
                
        elif choice == "3":
            print("Goodbye!")
            break

def process_request(filepath, model, fe, hubert, le):
    print("\n🎧 Analyzing Accent...")
    region, conf = predict_accent(filepath, model, fe, hubert, le)
    
    print(f"\n✨ DETECTED ACCENT: {region.upper()} (Confidence: {conf*100:.1f}%)")
    print(f"📍 Inferred Region: {region}")
    
    if region in MENU:
        print("\n🍛 RECOMMENDED SPECIALTIES FOR YOU:")
        for item in MENU[region]:
            print(f"   - {item}")
    else:
        print("No specific menu for this region yet.")

if __name__ == "__main__":
    main()