import customtkinter as ctk
import threading
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
MODEL_PATH = "hubert_features_layer9.pkl" 
SAMPLE_RATE = 16000
DURATION = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MENU DATA ---
MENU = {
    "kerala": ["Appam with Stew", "Puttu and Kadala Curry", "Kerala Sadya", "Avial"],
    "tamil": ["Chicken Chettinad", "Dosa with Sambar", "Pongal", "Filter Coffee"],
    "andhra_pradesh": ["Hyderabadi Biryani", "Gongura Pachadi", "Pesarattu", "Spicy Chicken Curry"],
    "karnataka": ["Bisi Bele Bath", "Mysore Masala Dosa", "Ragi Mudde", "Mysore Pak"],
    "gujrat": ["Dhokla", "Thepla", "Undhiyu", "Basundi"],
    "jharkhand": ["Litti Chokha", "Dhuska", "Rugra", "Malpua"]
}

# --- MODEL DEFINITION (Must match training) ---
class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearProbe, self).__init__()
        self.layer = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.layer(x)

class AccentApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Setup
        self.title("Accent-Aware Cuisine Recommender")
        self.geometry("600x700")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # Load AI Models in Background
        self.status_label = ctk.CTkLabel(self, text="Loading AI Models... Please Wait", font=("Arial", 16))
        self.status_label.pack(pady=20)
        
        # Variables
        self.model = None
        self.feature_extractor = None
        self.hubert = None
        self.label_encoder = None
        self.is_recording = False

        # Start loading thread
        threading.Thread(target=self.load_models, daemon=True).start()

        # --- GUI LAYOUT ---
        
        # 1. Header
        self.header = ctk.CTkLabel(self, text="🍽️ Native Taste", font=("Roboto", 32, "bold"))
        self.header.pack(pady=(40, 10))
        
        self.sub_header = ctk.CTkLabel(self, text="Speak to get personalized food recommendations", font=("Arial", 14), text_color="gray")
        self.sub_header.pack(pady=(0, 30))

        # 2. Record Section
        self.record_btn = ctk.CTkButton(self, text="🎤 Tap to Record", width=200, height=60, 
                                        font=("Arial", 20, "bold"), corner_radius=30, 
                                        fg_color="#E53935", hover_color="#D32F2F",
                                        command=self.toggle_recording)
        self.record_btn.pack(pady=20)
        self.record_btn.configure(state="disabled") # Disabled until models load

        self.timer_label = ctk.CTkLabel(self, text="", font=("Arial", 14))
        self.timer_label.pack(pady=5)

        # 3. Results Card (Initially Hidden or Empty)
        self.result_frame = ctk.CTkFrame(self, width=500, height=300, corner_radius=20)
        self.result_frame.pack(pady=30, padx=20, fill="both", expand=True)
        
        self.result_title = ctk.CTkLabel(self.result_frame, text="Waiting for input...", font=("Arial", 18, "bold"))
        self.result_title.pack(pady=(20, 10))
        
        self.accent_label = ctk.CTkLabel(self.result_frame, text="", font=("Arial", 16), text_color="#4FC3F7")
        self.accent_label.pack(pady=5)

        self.menu_text = ctk.CTkTextbox(self.result_frame, width=400, height=150, font=("Arial", 14))
        self.menu_text.pack(pady=10)
        self.menu_text.insert("0.0", "Recommendations will appear here.")
        self.menu_text.configure(state="disabled")

    def load_models(self):
        """Loads the heavy AI models without freezing the GUI"""
        try:
            # Load Encoder
            df = pd.read_pickle(MODEL_PATH)
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(df['label'])
            num_classes = len(self.label_encoder.classes_)

            # Load HuBERT
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
            self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
            self.hubert.to(DEVICE)
            self.hubert.eval()

            # Load Classifier & Quick Train (Same logic as before)
            self.model = LinearProbe(768, num_classes).to(DEVICE)
            
            # Quick Retrain on data
            X = np.stack(df['features'].values)
            y = self.label_encoder.transform(df['label'])
            X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            y_tensor = torch.tensor(y, dtype=torch.long).to(DEVICE)
            
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
            criterion = nn.CrossEntropyLoss()
            self.model.train()
            for _ in range(30):
                optimizer.zero_grad()
                loss = criterion(self.model(X_tensor), y_tensor)
                loss.backward()
                optimizer.step()
            self.model.eval()

            # Enable GUI
            self.status_label.configure(text="System Ready", text_color="green")
            self.record_btn.configure(state="normal")
            
        except Exception as e:
            self.status_label.configure(text=f"Error: {e}", text_color="red")
            print(e)

    def toggle_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.record_btn.configure(text="🛑 Recording...", fg_color="gray")
            self.result_title.configure(text="Listening...", text_color="white")
            
            # Start recording in a separate thread so GUI doesn't freeze
            threading.Thread(target=self.record_and_process).start()

    def record_and_process(self):
        filename = "live_gui_input.wav"
        
        # Record
        recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
        for i in range(DURATION):
            self.timer_label.configure(text=f"Recording... {DURATION-i}s")
            time.sleep(1)
        sd.wait()
        write(filename, SAMPLE_RATE, (recording * 32767).astype(np.int16))
        
        self.timer_label.configure(text="Processing...")
        
        # Process
        try:
            region, conf = self.predict_accent(filename)
            
            # Update UI (Must be done on main thread essentially, but CTK handles this mostly ok)
            self.update_results(region, conf)
            
        except Exception as e:
            print(f"Prediction Error: {e}")

        # Reset Button
        self.is_recording = False
        self.record_btn.configure(text="🎤 Tap to Record", fg_color="#E53935")
        self.timer_label.configure(text="")

    def predict_accent(self, audio_path):
        signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
        target_len = SAMPLE_RATE * DURATION
        if len(signal) < target_len:
            padding = target_len - len(signal)
            signal = np.pad(signal, (0, padding), 'constant')
        else:
            signal = signal[:target_len]
            
        inputs = self.feature_extractor(signal, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(DEVICE)
        
        with torch.no_grad():
            outputs = self.hubert(input_values, output_hidden_states=True)
            layer_output = outputs.hidden_states[9]
            embedding = torch.mean(layer_output, dim=1).squeeze()
        
        embedding = embedding.unsqueeze(0).to(DEVICE)
        logits = self.model(embedding)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs).item()
        confidence = probs[0][pred_idx].item()
        
        return self.label_encoder.inverse_transform([pred_idx])[0], confidence

    def update_results(self, region, conf):
        # Title
        self.result_title.configure(text=f"Detected: {region.upper()}", text_color="#4CAF50")
        self.accent_label.configure(text=f"Confidence: {conf*100:.1f}%")
        
        # Food Recommendations
        self.menu_text.configure(state="normal")
        self.menu_text.delete("0.0", "end")
        
        if region in MENU:
            text = "Recommended Specialties:\n\n"
            for item in MENU[region]:
                text += f"• {item}\n"
        else:
            text = "No menu available for this region."
            
        self.menu_text.insert("0.0", text)
        self.menu_text.configure(state="disabled")

if __name__ == "__main__":
    app = AccentApp()
    app.mainloop()